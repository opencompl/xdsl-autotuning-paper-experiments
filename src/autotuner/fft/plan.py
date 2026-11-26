"""Scheduling decisions for the FFT expansions.

A DFT larger than one register kernel is split by Cooley-Tukey, and each
split can pick any radix: the order of the factors decides the strides of
every pass, and with them how each pass vectorizes. ``choose_split`` follows
an explicit ``FftStrategy.schedule`` when there is one, and otherwise searches
the factor orders with a cost model that counts instructions.
"""

import math
from dataclasses import dataclass
from functools import cache

from autotuner.dialects.fft import DftOp, RegisterLayout
from autotuner.fft.kernels import BASE_RADICES

SUPPORTED_PRIMES = (2, 3, 5, 7)

# Instructions of each base-case kernel, per vector of transforms, by layout.
KERNEL_INSTRUCTIONS = {
    RegisterLayout.SPLIT: {1: 0, 2: 4, 3: 12, 4: 16, 5: 36, 7: 72, 8: 52},
    RegisterLayout.INTERLEAVED: {1: 0, 2: 2, 3: 6, 4: 9, 5: 18, 7: 36, 8: 28},
}
# A twiddle multiplies and adds both parts, or swaps and multiplies twice.
TWIDDLE_INSTRUCTIONS = {RegisterLayout.SPLIT: 4, RegisterLayout.INTERLEAVED: 3}
# Compact twiddle factors load each factor twice, duplicating its parts.
COMPACT_TWIDDLE_INSTRUCTIONS = 2
# A split stores every point and reloads it in the next pass, which costs more
# than the two instructions that appear: the reload waits on store forwarding.
MEMORY_PASS_PER_POINT = 2
# Waiting for stores to commit because the store buffer cannot forward them,
# in instructions: about 20 cycles.
FORWARDING_STALL = 50


@dataclass(frozen=True)
class FftStrategy:
    """The knobs of the FFT schedule."""

    max_leaf: int = 8
    """The most points one register kernel transforms; larger DFTs are split
    in memory. Each point takes two vector registers."""

    vector_bits: tuple[int, ...] = (512, 256, 128)
    """The vector widths the kernels may use, widest first."""

    layout: RegisterLayout = RegisterLayout.SPLIT
    """How the register kernels hold complex numbers. Interleaved ones take one
    register per point, so they reach twice the points of split ones."""

    schedule: tuple[int, ...] = ()
    """The size of the DFTs in every pass over memory, in execution order, or
    empty to let the cost model choose. ``(64, 16)`` is FFTW's
    ``dft-ct-dit/16`` over ``dft-direct-64``: 16 DFTs of 64 points, then 64
    DFTs of 16 points with twiddles. Each size is one register kernel."""

    l1_bytes: int = 32768
    """The size of the L1 data cache, which the input, the output, and the
    twiddle factors of the interleaved layout should fit in together."""


@dataclass(frozen=True)
class DftShape:
    """The properties of an ``fft.dft`` that scheduling depends on."""

    n: int
    istride: int
    ostride: int
    batch: tuple[int, ...] = ()
    idist: tuple[int, ...] = ()
    odist: tuple[int, ...] = ()
    twiddle_n: int = 0
    twiddle_dim: int = 0

    @staticmethod
    def of(op: DftOp) -> "DftShape":
        return DftShape(
            op.n.value.data,
            op.istride.value.data,
            op.ostride.value.data,
            op.batch_values,
            op.idist_values,
            op.odist_values,
            op.twiddle_n.value.data,
            op.twiddle_dim.value.data,
        )

    def split(self, r: int) -> tuple["DftShape", "DftShape"]:
        """Decimation in time, ``n = r * m``: ``r`` DFTs of ``m`` points over
        the residues of the input index modulo ``r``, written to the output,
        then ``m`` DFTs of ``r`` points in place, with twiddle factors, over
        the output strided by ``m``."""
        assert not self.twiddle_n, "only a DFT without twiddles can be split"
        m = self.n // r
        first = DftShape(
            m,
            r * self.istride,
            self.ostride,
            (r, *self.batch),
            (self.istride, *self.idist),
            (m * self.ostride, *self.odist),
        )
        second = DftShape(
            r,
            m * self.ostride,
            m * self.ostride,
            (m, *self.batch),
            (self.ostride, *self.odist),
            (self.ostride, *self.odist),
            twiddle_n=self.n,
            twiddle_dim=0,
        )
        return first, second


@dataclass(frozen=True)
class LeafVectorization:
    """How a register kernel maps its transforms onto vector lanes.

    ``lanes`` transforms run side by side, one per lane, along batch dimension
    ``dimension`` (``None`` when ``lanes`` is 1). A side is ``transposed`` when
    its points, not its transforms, are contiguous in memory, so the kernel
    loads or stores whole blocks and transposes them in registers. In the
    interleaved ``layout`` a lane is a pair of elements, a side whose
    transforms are neither contiguous nor transposable is ``lanewise``,
    loading or storing each lane's complex number on its own, and a batch
    that ``lanes`` does not divide ends in narrower vectors (see ``chunks``).
    """

    lanes: int
    dimension: int | None
    input_transposed: bool
    output_transposed: bool
    layout: RegisterLayout = RegisterLayout.SPLIT
    input_lanewise: bool = False
    output_lanewise: bool = False


def chunks(extent: int, lanes: int) -> list[int]:
    """The lanes of each vector covering ``extent`` transforms: as many full
    vectors as fit, then the remainder in halving powers of two."""
    result = [lanes] * (extent // lanes)
    width = lanes // 2
    while width:
        if extent % lanes & width:
            result.append(width)
        width //= 2
    return result


def compact_twiddles(shape: DftShape, strategy: FftStrategy, element_bits: int) -> bool:
    """Whether interleaved twiddle factors are stored compactly.

    Duplicated, the factors of a DFT take about as many bytes as its input
    and output together (two vectors of 32 B per twiddled vector of 16 B
    points in f64), so they fit in L1 beside the data only while the data
    fills at most half of it. Beyond that, the misses cost more than the
    shuffles that compact factors need.
    """
    if strategy.layout is not RegisterLayout.INTERLEAVED:
        return False
    data_bytes = 2 * shape.n * math.prod(shape.batch) * element_bits // 4
    return 2 * data_bytes > strategy.l1_bytes


def factorize(n: int) -> list[int]:
    factors: list[int] = []
    for p in SUPPORTED_PRIMES:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n != 1:
        raise ValueError(f"DFT sizes must factor into {SUPPORTED_PRIMES}")
    return factors


def register_kernel_supported(n: int, strategy: FftStrategy) -> bool:
    if n > strategy.max_leaf:
        return False
    try:
        factors = factorize(n)
    except ValueError:
        return False
    return all(f in BASE_RADICES for f in factors)


def vectorizations(
    shape: DftShape, strategy: FftStrategy, element_bits: int
) -> list[LeafVectorization]:
    """Every legal vectorization of a register kernel, scalar last."""
    layout = strategy.layout
    lane_bits = element_bits * (2 if layout is RegisterLayout.INTERLEAVED else 1)
    result: list[LeafVectorization] = []
    for bits in strategy.vector_bits:
        lanes = bits // lane_bits
        if lanes < 2:
            continue
        for dimension, extent in enumerate(shape.batch):
            # Only interleaved kernels can end a batch in narrower vectors or
            # move lanes one at a time.
            flexible = layout is RegisterLayout.INTERLEAVED
            if extent % lanes and not (flexible and extent > lanes):
                continue
            input_direct = shape.idist[dimension] == 1
            output_direct = shape.odist[dimension] == 1
            transposable = shape.n % lanes == 0
            input_transposed = not input_direct and shape.istride == 1 and transposable
            output_transposed = (
                not output_direct and shape.ostride == 1 and transposable
            )
            input_lanewise = not (input_direct or input_transposed)
            output_lanewise = not (output_direct or output_transposed)
            if (input_lanewise or output_lanewise) and not flexible:
                continue
            result.append(
                LeafVectorization(
                    lanes,
                    dimension,
                    input_transposed,
                    output_transposed,
                    layout,
                    input_lanewise,
                    output_lanewise,
                )
            )
    result.append(LeafVectorization(1, None, False, False, layout))
    return result


def _kernel_instructions(n: int, layout: RegisterLayout) -> int:
    if n in KERNEL_INSTRUCTIONS[layout]:
        return KERNEL_INSTRUCTIONS[layout][n]
    r = next(r for r in (4, 2, 3, 5) if n % r == 0)
    m = n // r
    return (
        r * _kernel_instructions(m, layout)
        + m * _kernel_instructions(r, layout)
        + (r - 1) * (m - 1) * TWIDDLE_INSTRUCTIONS[layout]
    )


def leaf_cost(shape: DftShape, plan: LeafVectorization, compact: bool) -> float:
    """The instructions a register kernel executes."""
    n, layout = shape.n, plan.layout
    registers = layout.registers_per_point

    def vector_cost(lanes: int) -> int:
        cost = _kernel_instructions(n, layout)
        if shape.twiddle_n:
            twiddle = TWIDDLE_INSTRUCTIONS[layout]
            if compact and lanes > 1 and shape.twiddle_dim == plan.dimension:
                twiddle += COMPACT_TWIDDLE_INSTRUCTIONS
            cost += (n - 1) * twiddle
        if layout is RegisterLayout.SPLIT:
            # Each point: two loads (or a re and an im) and two stores, plus
            # two shuffles to (de)interleave each way with more than one lane.
            cost += n * (4 + (4 if lanes > 1 else 0))
        else:
            # Each point: one load and one store, as in memory.
            cost += n * 2
        if lanes > 1:
            for transposed in (plan.input_transposed, plan.output_transposed):
                if transposed:
                    cost += registers * n * int(math.log2(lanes))
            for lanewise in (plan.input_lanewise, plan.output_lanewise):
                if lanewise:
                    # One load or store per lane instead of per vector.
                    cost += n * (lanes - 1)
        return cost

    if plan.dimension is None:
        return math.prod(shape.batch) * vector_cost(1)
    extent = shape.batch[plan.dimension]
    others = math.prod(shape.batch) // extent
    return others * sum(vector_cost(lanes) for lanes in chunks(extent, plan.lanes))


@cache
def best_leaf(
    shape: DftShape, strategy: FftStrategy, element_bits: int
) -> tuple[float, LeafVectorization]:
    compact = compact_twiddles(shape, strategy, element_bits)
    return min(
        (
            (leaf_cost(shape, plan, compact), plan)
            for plan in vectorizations(shape, strategy, element_bits)
        ),
        key=lambda candidate: candidate[0],
    )


@cache
def plan_cost(shape: DftShape, strategy: FftStrategy, element_bits: int) -> float:
    """The cost of the cheapest schedule of ``shape``."""
    return min(
        (cost for cost, _ in _options(shape, strategy, element_bits)),
        default=math.inf,
    )


def _options(
    shape: DftShape, strategy: FftStrategy, element_bits: int
) -> list[tuple[float, int | None]]:
    """``(cost, radix)`` for every schedule, ``None`` meaning a leaf."""
    options: list[tuple[float, int | None]] = []
    if register_kernel_supported(shape.n, strategy):
        options.append((best_leaf(shape, strategy, element_bits)[0], None))
    if shape.twiddle_n:
        return options
    for r in range(2, shape.n):
        if shape.n % r or not register_kernel_supported(r, strategy):
            continue
        first, second = shape.split(r)
        cost = (
            plan_cost(first, strategy, element_bits)
            + plan_cost(second, strategy, element_bits)
            + MEMORY_PASS_PER_POINT * shape.n * math.prod(shape.batch)
            + _forwarding_stall(first, second, strategy, element_bits)
        )
        options.append((cost, r))
    return options


def _forwarding_stall(
    first: DftShape, second: DftShape, strategy: FftStrategy, element_bits: int
) -> float:
    """The cost of the second pass reloading what the first just stored.

    A load can only take its data from the store buffer when one store wrote
    all of it. When the first pass stores fewer bytes at a time than the
    second loads, as a pass storing lane by lane does, the second pass waits
    for the latest stores to commit. Other work hides that wait in all but
    the smallest DFTs, so it counts once, not per load.
    """
    if strategy.layout is not RegisterLayout.INTERLEAVED:
        return 0
    split = choose_split(first, strategy, element_bits)
    last = first if split is None else first.split(split)[1]
    stored = best_leaf(last, strategy, element_bits)[1]
    loaded = best_leaf(second, strategy, element_bits)[1]
    store_lanes = 1 if stored.output_lanewise else stored.lanes
    load_lanes = 1 if loaded.input_lanewise else loaded.lanes
    return FORWARDING_STALL if load_lanes > store_lanes else 0


def choose_split(
    shape: DftShape, strategy: FftStrategy, element_bits: int
) -> int | None:
    """The radix of the split of ``shape``, or ``None`` for a leaf."""
    if strategy.schedule and not shape.twiddle_n:
        return _scheduled_split(shape, strategy)
    options = _options(shape, strategy, element_bits)
    if not options:
        raise ValueError(f"cannot schedule a DFT of {shape.n} points")
    return min(options, key=lambda option: option[0])[1]


def _scheduled_split(shape: DftShape, strategy: FftStrategy) -> int | None:
    """The split ``strategy.schedule`` gives ``shape``.

    Splitting off the last pass leaves the passes before it for the first
    half, so a DFT of ``n`` points runs the passes whose sizes multiply to
    ``n`` and splits off the last of them.
    """
    schedule = strategy.schedule
    unsupported = [
        size
        for size in schedule
        if size < 2 or not register_kernel_supported(size, strategy)
    ]
    if unsupported:
        raise ValueError(
            f"no register kernel for a pass of {unsupported[0]} points "
            f"(max_leaf={strategy.max_leaf})"
        )
    for count in range(1, len(schedule) + 1):
        if math.prod(schedule[:count]) == shape.n:
            return None if count == 1 else schedule[count - 1]
    raise ValueError(f"the schedule {schedule} has no DFT of {shape.n} points")
