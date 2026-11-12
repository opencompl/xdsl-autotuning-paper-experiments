"""Scheduling decisions for the FFT expansions.

A DFT larger than one register kernel is split by Cooley-Tukey, and each
split can pick any radix: the order of the factors decides the strides of
every pass, and with them how each pass vectorizes. ``choose_split`` searches
the factor orders with a cost model that counts instructions.
"""

import math
from dataclasses import dataclass
from functools import cache

from autotuner.dialects.fft import DftOp
from autotuner.fft.kernels import BASE_RADICES

SUPPORTED_PRIMES = (2, 3, 5, 7)

# Instructions of each base-case kernel, per vector of transforms.
KERNEL_INSTRUCTIONS = {1: 0, 2: 4, 3: 12, 4: 16, 5: 36, 7: 72, 8: 52}
TWIDDLE_INSTRUCTIONS = 4
# A split stores every point and reloads it in the next pass, which costs more
# than the two instructions that appear: the reload waits on store forwarding.
MEMORY_PASS_PER_POINT = 2


@dataclass(frozen=True)
class FftStrategy:
    """The knobs of the FFT schedule."""

    max_leaf: int = 8
    """The most points one register kernel transforms; larger DFTs are split
    in memory. Each point takes two vector registers."""

    vector_bits: tuple[int, ...] = (512, 256, 128)
    """The vector widths the kernels may use, widest first."""


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
    loads or stores whole blocks and transposes them in registers.
    """

    lanes: int
    dimension: int | None
    input_transposed: bool
    output_transposed: bool


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
    result: list[LeafVectorization] = []
    for bits in strategy.vector_bits:
        lanes = bits // element_bits
        if lanes < 2:
            continue
        for dimension, extent in enumerate(shape.batch):
            if extent % lanes:
                continue
            input_direct = shape.idist[dimension] == 1
            output_direct = shape.odist[dimension] == 1
            transposable = shape.n % lanes == 0
            if not (input_direct or (shape.istride == 1 and transposable)):
                continue
            if not (output_direct or (shape.ostride == 1 and transposable)):
                continue
            result.append(
                LeafVectorization(lanes, dimension, not input_direct, not output_direct)
            )
    result.append(LeafVectorization(1, None, False, False))
    return result


def _kernel_instructions(n: int) -> int:
    if n in KERNEL_INSTRUCTIONS:
        return KERNEL_INSTRUCTIONS[n]
    r = next(r for r in (4, 2, 3, 5) if n % r == 0)
    m = n // r
    return (
        r * _kernel_instructions(m)
        + m * _kernel_instructions(r)
        + (r - 1) * (m - 1) * TWIDDLE_INSTRUCTIONS
    )


def leaf_cost(shape: DftShape, plan: LeafVectorization) -> float:
    """The instructions a register kernel executes."""
    lanes, n = plan.lanes, shape.n
    groups = math.prod(shape.batch) // lanes
    per_group = _kernel_instructions(n)
    if shape.twiddle_n:
        per_group += (n - 1) * TWIDDLE_INSTRUCTIONS
    # Each point: two loads (or a re and an im) and two stores, plus two
    # shuffles to (de)interleave each way when there is more than one lane.
    per_group += n * (4 + (4 if lanes > 1 else 0))
    for transposed in (plan.input_transposed, plan.output_transposed):
        if transposed:
            per_group += 2 * n * int(math.log2(lanes))
    return groups * per_group


@cache
def best_leaf(
    shape: DftShape, strategy: FftStrategy, element_bits: int
) -> tuple[float, LeafVectorization]:
    return min(
        (
            (leaf_cost(shape, plan), plan)
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
        )
        options.append((cost, r))
    return options


def choose_split(
    shape: DftShape, strategy: FftStrategy, element_bits: int
) -> int | None:
    """The radix of the cheapest split of ``shape``, or ``None`` for a leaf."""
    options = _options(shape, strategy, element_bits)
    if not options:
        raise ValueError(f"cannot schedule a DFT of {shape.n} points")
    return min(options, key=lambda option: option[0])[1]
