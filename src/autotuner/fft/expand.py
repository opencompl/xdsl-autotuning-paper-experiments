"""The rewrite patterns that expand ``fft.dft`` down to x86 instructions."""

import itertools
import math
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects import x86_func
from xdsl.dialects.x86.registers import X86VectorRegisterType
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.fft import DftOp, DftRegOp, TransposeOp, TwiddleOp
from autotuner.dialects.x86ext import FmaKind
from autotuner.fft import kernels
from autotuner.fft.constants import ConstantPool
from autotuner.fft.lanes import (
    complex_lane_order,
    deinterleave_shuffles,
    interleave_shuffles,
    transpose_network,
)
from autotuner.fft.plan import (
    DftShape,
    FftStrategy,
    LeafVectorization,
    best_leaf,
    choose_split,
)
from autotuner.fft.values import Datatype, Emitter, Real

Pools = dict[x86_func.FuncOp, ConstantPool]
OpT = TypeVar("OpT", bound=Operation)


def _emitter(
    rewriter: PatternRewriter,
    op: DftOp | DftRegOp | TwiddleOp | TransposeOp,
    datatype: Datatype,
    pools: Pools,
) -> Emitter:
    func = op.parent_op()
    while not isinstance(func, x86_func.FuncOp):
        if func is None:
            raise PassFailedException("FFT operations must be inside x86_func.func")
        func = func.parent_op()
    if func not in pools:
        pools[func] = ConstantPool(func, datatype)
    return Emitter(rewriter, op, datatype, pools[func])


def _dft_op(shape: DftShape, input: SSAValue, output: SSAValue, like: DftOp) -> DftOp:
    return DftOp(
        input,
        output,
        n=shape.n,
        sign=like.sign.value.data,
        istride=shape.istride,
        ostride=shape.ostride,
        batch=shape.batch,
        idist=shape.idist,
        odist=shape.odist,
        twiddle_n=shape.twiddle_n,
        twiddle_dim=shape.twiddle_dim,
        datatype=like.datatype,
    )


@dataclass
class SplitDftPattern(RewritePattern):
    """Cooley-Tukey in memory, decimating in time (see ``DftShape.split``).

    The radix comes from the cost model in ``choose_split``. The second pass
    reads each point once, so it is a register kernel; the first recurses.
    """

    strategy: FftStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftOp, rewriter: PatternRewriter, /) -> None:
        shape = DftShape.of(op)
        try:
            r = choose_split(shape, self.strategy, op.datatype.bitwidth)
        except ValueError as error:
            raise PassFailedException(str(error)) from error
        if r is None:
            return
        first, second = shape.split(r)
        rewriter.replace(
            op,
            (
                _dft_op(first, op.input, op.output, op),
                _dft_op(second, op.output, op.output, op),
            ),
            (),
        )


@dataclass
class DftToRegistersPattern(RewritePattern):
    """Load a batch of small DFTs into registers, transform, and store them."""

    strategy: FftStrategy
    pools: Pools

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftOp, rewriter: PatternRewriter, /) -> None:
        shape = DftShape.of(op)
        if choose_split(shape, self.strategy, op.datatype.bitwidth) is not None:
            return
        e = _emitter(rewriter, op, op.datatype, self.pools)
        _, plan = best_leaf(shape, self.strategy, e.element_bits)
        _lower_leaf(e, op, plan)
        rewriter.erase(op)


def _lower_leaf(e: Emitter, op: DftOp, plan: LeafVectorization) -> None:
    n, sign = op.n.value.data, op.sign.value.data
    batch = op.batch_values
    idist, odist = op.idist_values, op.odist_values
    istride, ostride = op.istride.value.data, op.ostride.value.data
    twiddle_n, twiddle_dim = op.twiddle_n.value.data, op.twiddle_dim.value.data
    lanes = plan.lanes
    bits = lanes * e.element_bits
    complex_bytes = 2 * e.element_size
    order = complex_lane_order(lanes, e.element_bits)

    looped = [d for d in range(len(batch)) if d != plan.dimension]
    # Only a vectorized side can be transposed, so the offsets of transposed
    # lanes always run along a batch dimension.
    dimension = 0 if plan.dimension is None else plan.dimension
    groups = 1 if plan.dimension is None else batch[plan.dimension] // lanes
    for index in itertools.product(*(range(batch[d]) for d in looped)):
        in_base = sum(i * idist[d] for i, d in zip(index, looped))
        out_base = sum(i * odist[d] for i, d in zip(index, looped))
        for group in range(groups):
            # The batch index each lane transforms along the lane dimension.
            lane_batch = [group * lanes + order[i] for i in range(lanes)]

            # Load.
            if plan.input_transposed:
                points = _load_transposed(
                    e,
                    op.input,
                    n,
                    lanes,
                    order,
                    [in_base + lane_batch[p] * idist[dimension] for p in range(lanes)],
                    complex_bytes,
                )
            else:
                start = in_base + group * lanes
                points = [
                    _load_point(
                        e, op.input, (start + j * istride) * complex_bytes, bits
                    )
                    for j in range(n)
                ]

            # Twiddle.
            if twiddle_n:
                for j in range(1, n):
                    if twiddle_dim == plan.dimension:
                        exponents = [sign * j * b for b in lane_batch]
                    else:
                        b = index[looped.index(twiddle_dim)]
                        exponents = [sign * j * b]
                    if all(x % twiddle_n == 0 for x in exponents):
                        continue
                    twiddle = e.rewriter.insert(
                        TwiddleOp(
                            points[j],
                            n=twiddle_n,
                            exponents=exponents,
                            datatype=op.datatype,
                        ),
                        e.insertion_point(),
                    )
                    points[j] = (twiddle.outputs[0], twiddle.outputs[1])

            # Transform.
            dft = e.rewriter.insert(
                DftRegOp(points, sign=sign, datatype=op.datatype), e.insertion_point()
            )
            results = dft.output_points()

            # Store.
            if plan.output_transposed:
                _store_transposed(
                    e,
                    op.output,
                    results,
                    lanes,
                    order,
                    [out_base + lane_batch[k] * odist[dimension] for k in range(lanes)],
                    complex_bytes,
                )
            else:
                start = out_base + group * lanes
                for k, point in enumerate(results):
                    _store_point(
                        e, op.output, (start + k * ostride) * complex_bytes, point, bits
                    )


def _load_point(
    e: Emitter, pointer: SSAValue, offset: int, bits: int
) -> tuple[SSAValue, SSAValue]:
    """Load consecutive complex numbers, one per lane, as real and imaginary parts."""
    if bits == e.element_bits:
        return (
            e.load(pointer, offset, bits),
            e.load(pointer, offset + e.element_size, bits),
        )
    first = e.load(pointer, offset, bits)
    second = e.load(pointer, offset + bits // 8, bits)
    low, high = deinterleave_shuffles(e.element_bits)
    return (
        e.shuffle(low, first, second, e.element_bits),
        e.shuffle(high, first, second, e.element_bits),
    )


def _store_point(
    e: Emitter,
    pointer: SSAValue,
    offset: int,
    point: tuple[SSAValue, SSAValue],
    bits: int,
) -> None:
    re, im = point
    if bits == e.element_bits:
        e.store(pointer, offset, re, bits)
        e.store(pointer, offset + e.element_size, im, bits)
        return
    low, high = interleave_shuffles(e.element_bits)
    e.store(pointer, offset, e.shuffle(low, re, im, e.element_bits), bits)
    e.store(pointer, offset + bits // 8, e.shuffle(high, re, im, e.element_bits), bits)


def _transpose(
    e: Emitter, values: list[SSAValue], datatype: Datatype
) -> list[SSAValue]:
    op = e.rewriter.insert(TransposeOp(values, datatype=datatype), e.insertion_point())
    return list(op.outputs)


def _load_transposed(
    e: Emitter,
    pointer: SSAValue,
    n: int,
    lanes: int,
    order: tuple[int, ...],
    lane_bases: list[int],
    complex_bytes: int,
) -> list[tuple[SSAValue, SSAValue]]:
    """Load ``lanes`` transforms whose points are contiguous, one per lane.

    Lane ``p`` of every result transforms the batch element at complex index
    ``lane_bases[p]``. Loading a block of that element's points puts point
    ``k0 + order[l]`` in lane ``l``, so result ``k`` of the transpose holds
    point ``k0 + order[k]``.
    """
    bits = lanes * 8 * complex_bytes // 2
    points: list[tuple[SSAValue, SSAValue] | None] = [None] * n
    for k0 in range(0, n, lanes):
        loaded = [
            _load_point(e, pointer, (lane_bases[p] + k0) * complex_bytes, bits)
            for p in range(lanes)
        ]
        datatype = e.datatype
        re = _transpose(e, [point[0] for point in loaded], datatype)
        im = _transpose(e, [point[1] for point in loaded], datatype)
        for k in range(lanes):
            points[k0 + order[k]] = (re[k], im[k])
    assert all(point is not None for point in points)
    return points  # type: ignore[return-value]


def _store_transposed(
    e: Emitter,
    pointer: SSAValue,
    results: list[tuple[SSAValue, SSAValue]],
    lanes: int,
    order: tuple[int, ...],
    lane_bases: list[int],
    complex_bytes: int,
) -> None:
    """The inverse of ``_load_transposed``."""
    bits = lanes * 8 * complex_bytes // 2
    for k0 in range(0, len(results), lanes):
        block = [results[k0 + order[p]] for p in range(lanes)]
        re = _transpose(e, [point[0] for point in block], e.datatype)
        im = _transpose(e, [point[1] for point in block], e.datatype)
        for k in range(lanes):
            _store_point(
                e, pointer, (lane_bases[k] + k0) * complex_bytes, (re[k], im[k]), bits
            )


@dataclass
class SplitDftRegPattern(RewritePattern):
    """Cooley-Tukey in registers: renames registers instead of moving data."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftRegOp, rewriter: PatternRewriter, /) -> None:
        n = op.n.value.data
        if n in kernels.BASE_RADICES or n == 1:
            return
        r = next((r for r in (4, 2, 3, 5, 8) if n % r == 0 and r < n), None)
        if r is None:
            raise PassFailedException(f"no register kernel for {n} points")
        m = n // r
        sign = op.sign.value.data
        x = op.input_points()

        def insert(new_op: OpT) -> OpT:
            return rewriter.insert(new_op, InsertPoint.before(op))

        inner = [
            insert(
                DftRegOp(
                    [x[j2 + r * j1] for j1 in range(m)], sign=sign, datatype=op.datatype
                )
            ).output_points()
            for j2 in range(r)
        ]
        results: list[SSAValue] = [op.inputs[0]] * (2 * n)
        for k1 in range(m):
            column = []
            for j2 in range(r):
                point = inner[j2][k1]
                if (j2 * k1) % n:
                    twiddle = insert(
                        TwiddleOp(
                            point, n=n, exponents=[sign * j2 * k1], datatype=op.datatype
                        )
                    )
                    point = (twiddle.outputs[0], twiddle.outputs[1])
                column.append(point)
            outer = insert(DftRegOp(column, sign=sign, datatype=op.datatype))
            for k2, (re, im) in enumerate(outer.output_points()):
                results[2 * (k1 + m * k2)] = re
                results[2 * (k1 + m * k2) + 1] = im
        rewriter.replace(op, (), results)


@dataclass
class DftRegBaseCasePattern(RewritePattern):
    """The nano-kernels: radix 2, 3, 4, 5, and 8 DFTs as adds and FMAs."""

    pools: Pools

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftRegOp, rewriter: PatternRewriter, /) -> None:
        n = op.n.value.data
        if n not in kernels.BASE_RADICES and n != 1:
            return
        e = _emitter(rewriter, op, op.datatype, self.pools)
        points = [(Real(re), Real(im)) for re, im in op.input_points()]
        results = kernels.dft(e, points, op.sign.value.data)
        values = [e.materialize(part) for point in results for part in point]
        rewriter.replace(op, (), values)


@dataclass
class TwiddlePattern(RewritePattern):
    """Multiply by twiddle factors: lazily for one shared factor, through
    vector constants for one factor per lane."""

    pools: Pools

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TwiddleOp, rewriter: PatternRewriter, /) -> None:
        e = _emitter(rewriter, op, op.datatype, self.pools)
        n = op.n.value.data
        exponents = op.exponents.get_values()
        re, im = op.inputs
        if len(set(exponents)) == 1:
            result = e.times_root((Real(re), Real(im)), exponents[0], n)
            rewriter.replace(op, (), [e.materialize(part) for part in result])
            return

        assert isinstance(re.type, X86VectorRegisterType)
        lanes = re.type.bitwidth() // e.element_bits
        assert len(exponents) == lanes, "expected one exponent per lane"
        angles = [2 * math.pi * x / n for x in exponents]
        cosines = e.pool.offset([math.cos(a) for a in angles])
        sines = e.pool.offset([math.sin(a) for a in angles])
        # (a + ib)(c + is) = (ac - bs) + i(as + bc)
        bs = e.mul_constant_raw(im, sines, broadcast=False)
        new_re = e.fma_constant_raw(
            re, bs, cosines, broadcast=False, kind=FmaKind.FMSUB
        )
        bc = e.mul_constant_raw(im, cosines, broadcast=False)
        new_im = e.fma_constant_raw(re, bc, sines, broadcast=False, kind=FmaKind.FMADD)
        rewriter.replace(op, (), [new_re, new_im])


@dataclass
class TransposePattern(RewritePattern):
    """Transpose a square block of vectors with a network of shuffles."""

    pools: Pools

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter, /) -> None:
        e = _emitter(rewriter, op, op.datatype, self.pools)
        lanes = len(op.inputs)
        network = transpose_network(lanes, e.element_bits)
        registers = [op.inputs[source] for source in network.source_of_slot]
        for stage_start in range(0, len(network.steps), lanes):
            previous = list(registers)
            for step, bits in network.steps[stage_start : stage_start + lanes]:
                registers[step.destination] = e.shuffle(
                    step.shuffle, previous[step.a], previous[step.b], bits
                )
        results: list[SSAValue] = [op.inputs[0]] * lanes
        for slot, result in enumerate(network.result_of_slot):
            results[result] = registers[slot]
        rewriter.replace(op, (), results)
