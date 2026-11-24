"""The rewrite patterns that expand ``fft.dft`` down to x86 instructions."""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from xdsl.dialects import x86, x86_func
from xdsl.dialects.x86.registers import SSERegisterType, X86VectorRegisterType
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects import x86ext
from autotuner.dialects.fft import (
    DftOp,
    DftRegOp,
    Point,
    RegisterLayout,
    TransposeOp,
    TwiddleOp,
)
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
    chunks,
    compact_twiddles,
)
from autotuner.fft.values import (
    Datatype,
    Emitter,
    InterleavedAlgebra,
    Packed,
    Real,
    vector_type,
)

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
        compact = compact_twiddles(shape, self.strategy, e.element_bits)
        _lower_leaf(e, op, plan, compact)
        rewriter.erase(op)


def _lower_leaf(e: Emitter, op: DftOp, plan: LeafVectorization, compact: bool) -> None:
    n, sign = op.n.value.data, op.sign.value.data
    batch = op.batch_values
    idist, odist = op.idist_values, op.odist_values
    istride, ostride = op.istride.value.data, op.ostride.value.data
    twiddle_n, twiddle_dim = op.twiddle_n.value.data, op.twiddle_dim.value.data
    layout = plan.layout
    complex_bytes = 2 * e.element_size

    looped = [d for d in range(len(batch)) if d != plan.dimension]
    # Only a vectorized side can be transposed, so the offsets of transposed
    # lanes always run along a batch dimension.
    dimension = 0 if plan.dimension is None else plan.dimension
    widths = [1] if plan.dimension is None else chunks(batch[dimension], plan.lanes)
    for index in itertools.product(*(range(batch[d]) for d in looped)):
        in_base = sum(i * idist[d] for i, d in zip(index, looped))
        out_base = sum(i * odist[d] for i, d in zip(index, looped))
        first = 0
        for lanes in widths:
            if layout is RegisterLayout.INTERLEAVED:
                bits = lanes * 8 * complex_bytes
                order = tuple(range(lanes))
            else:
                bits = lanes * e.element_bits
                order = complex_lane_order(lanes, e.element_bits)
            # The batch index each lane transforms along the lane dimension.
            lane_batch = [first + order[i] for i in range(lanes)]
            if plan.dimension is None:
                in_lanes, out_lanes = [in_base], [out_base]
            else:
                in_lanes = [in_base + b * idist[dimension] for b in lane_batch]
                out_lanes = [out_base + b * odist[dimension] for b in lane_batch]
            # Where a direct side's first lane starts.
            in_start, out_start = in_lanes[0], out_lanes[0]

            # Load.
            points: list[Point]
            if lanes > 1 and plan.input_transposed:
                points = _load_transposed(
                    e, op.input, n, lanes, order, in_lanes, complex_bytes, bits, layout
                )
            elif lanes > 1 and plan.input_lanewise:
                points = [
                    (
                        e.load_lanewise(
                            op.input,
                            [(b + j * istride) * complex_bytes for b in in_lanes],
                            bits,
                        ),
                    )
                    for j in range(n)
                ]
            else:
                points = [
                    _load_point(
                        e,
                        op.input,
                        (in_start + j * istride) * complex_bytes,
                        bits,
                        layout,
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
                            compact=compact,
                        ),
                        e.insertion_point(),
                    )
                    points[j] = tuple(twiddle.outputs)

            # Transform.
            dft = e.rewriter.insert(
                DftRegOp(points, sign=sign, layout=layout, datatype=op.datatype),
                e.insertion_point(),
            )
            results = dft.output_points()

            # Store.
            if lanes > 1 and plan.output_transposed:
                _store_transposed(
                    e,
                    op.output,
                    results,
                    lanes,
                    order,
                    out_lanes,
                    complex_bytes,
                    bits,
                    layout,
                )
            elif lanes > 1 and plan.output_lanewise:
                for k, (point,) in enumerate(results):
                    e.store_lanewise(
                        op.output,
                        [(b + k * ostride) * complex_bytes for b in out_lanes],
                        point,
                        bits,
                    )
            else:
                for k, point in enumerate(results):
                    _store_point(
                        e,
                        op.output,
                        (out_start + k * ostride) * complex_bytes,
                        point,
                        bits,
                        layout,
                    )
            first += lanes


def _load_point(
    e: Emitter, pointer: SSAValue, offset: int, bits: int, layout: RegisterLayout
) -> Point:
    """Load consecutive complex numbers, one per lane: as they are in the
    interleaved layout, as real and imaginary parts in the split one."""
    if layout is RegisterLayout.INTERLEAVED:
        return (e.load(pointer, offset, bits),)
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
    point: Point,
    bits: int,
    layout: RegisterLayout,
) -> None:
    if layout is RegisterLayout.INTERLEAVED:
        e.store(pointer, offset, point[0], bits)
        return
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
    bits: int,
    layout: RegisterLayout,
) -> list[Point]:
    """Load ``lanes`` transforms whose points are contiguous, one per lane.

    Lane ``p`` of every result transforms the batch element at complex index
    ``lane_bases[p]``. Loading a block of that element's points puts point
    ``k0 + order[l]`` in lane ``l``, so result ``k`` of the transpose holds
    point ``k0 + order[k]``. Each register of a point transposes separately.
    """
    points: list[Point | None] = [None] * n
    for k0 in range(0, n, lanes):
        loaded = [
            _load_point(e, pointer, (lane_bases[p] + k0) * complex_bytes, bits, layout)
            for p in range(lanes)
        ]
        parts = [
            _transpose(e, [point[w] for point in loaded], e.datatype)
            for w in range(layout.registers_per_point)
        ]
        for k in range(lanes):
            points[k0 + order[k]] = tuple(part[k] for part in parts)
    assert all(point is not None for point in points)
    return points  # type: ignore[return-value]


def _store_transposed(
    e: Emitter,
    pointer: SSAValue,
    results: list[Point],
    lanes: int,
    order: tuple[int, ...],
    lane_bases: list[int],
    complex_bytes: int,
    bits: int,
    layout: RegisterLayout,
) -> None:
    """The inverse of ``_load_transposed``."""
    for k0 in range(0, len(results), lanes):
        block = [results[k0 + order[p]] for p in range(lanes)]
        parts = [
            _transpose(e, [point[w] for point in block], e.datatype)
            for w in range(layout.registers_per_point)
        ]
        for k in range(lanes):
            _store_point(
                e,
                pointer,
                (lane_bases[k] + k0) * complex_bytes,
                tuple(part[k] for part in parts),
                bits,
                layout,
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
        sign, layout = op.sign.value.data, op.register_layout
        x = op.input_points()

        def insert(new_op: OpT) -> OpT:
            return rewriter.insert(new_op, InsertPoint.before(op))

        inner = [
            insert(
                DftRegOp(
                    [x[j2 + r * j1] for j1 in range(m)],
                    sign=sign,
                    layout=layout,
                    datatype=op.datatype,
                )
            ).output_points()
            for j2 in range(r)
        ]
        results: list[Point] = [()] * n
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
                    point = tuple(twiddle.outputs)
                column.append(point)
            outer = insert(
                DftRegOp(column, sign=sign, layout=layout, datatype=op.datatype)
            )
            for k2, point in enumerate(outer.output_points()):
                results[k1 + m * k2] = point
        rewriter.replace(op, (), [value for point in results for value in point])


@dataclass
class DftRegBaseCasePattern(RewritePattern):
    """The nano-kernels: radix 2, 3, 4, 5, 7, 8, and 16 DFTs as adds and FMAs."""

    pools: Pools

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftRegOp, rewriter: PatternRewriter, /) -> None:
        n = op.n.value.data
        if n not in kernels.BASE_RADICES and n != 1:
            return
        e = _emitter(rewriter, op, op.datatype, self.pools)
        sign = op.sign.value.data
        if op.register_layout is RegisterLayout.INTERLEAVED:
            algebra = InterleavedAlgebra(e)
            packed = [Packed(value) for (value,) in op.input_points()]
            values = [
                value
                for point in kernels.dft(algebra, packed, sign)
                for value in algebra.cmaterialize(point)
            ]
        else:
            split = [(Real(re), Real(im)) for re, im in op.input_points()]
            values = [
                value
                for point in kernels.dft(e, split, sign)
                for value in e.cmaterialize(point)
            ]
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
        if len(op.inputs) == 1:
            self._interleaved(e, op, rewriter)
            return
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

    @staticmethod
    def _interleaved(e: Emitter, op: TwiddleOp, rewriter: PatternRewriter) -> None:
        n = op.n.value.data
        exponents = op.exponents.get_values()
        (value,) = op.inputs
        if len(set(exponents)) == 1:
            algebra = InterleavedAlgebra(e)
            result = algebra.times_root(Packed(value), exponents[0], n)
            rewriter.replace(op, (), list(algebra.cmaterialize(result)))
            return

        assert isinstance(value.type, X86VectorRegisterType)
        lanes = value.type.bitwidth() // (2 * e.element_bits)
        assert len(exponents) == lanes, "expected one exponent per lane"
        angles = [2 * math.pi * x / n for x in exponents]
        if op.compact is None:
            cosines = e.pool.offset([math.cos(a) for a in angles for _ in range(2)])
            sines = e.pool.offset(
                [v for a in angles for v in (-math.sin(a), math.sin(a))]
            )
            # (c + is)(a + ib) = (c, c) * (a, b) + (-s, s) * (b, a)
            rotated = e.mul_constant_raw(e.swap(value), sines, broadcast=False)
            result = e.fma_constant_raw(
                value, rotated, cosines, broadcast=False, kind=FmaKind.FMADD
            )
        else:
            pairs = e.pool.offset(
                [v for a in angles for v in (math.cos(a), math.sin(a))]
            )
            bits = value.type.bitwidth()
            cosines = e.load_pair_duplicated(pairs, bits, 0)
            sines = e.load_pair_duplicated(pairs, bits, 1)
            # (c + is)(a + ib) = (c, c) * (a, b) -/+ (s, s) * (b, a)
            rotated = e.mul_raw(e.swap(value), sines)
            result = e.fma_raw(value, cosines, rotated, kind=FmaKind.MADDSUB)
        rewriter.replace(op, (), [result])


@dataclass
class TransposePattern(RewritePattern):
    """Transpose a square block of vectors with a network of shuffles."""

    pools: Pools

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter, /) -> None:
        e = _emitter(rewriter, op, op.datatype, self.pools)
        lanes = len(op.inputs)
        register_type = op.inputs[0].type
        assert isinstance(register_type, X86VectorRegisterType)
        network = transpose_network(lanes, register_type.bitwidth() // lanes)
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


# How many stores back a reload can still find its stores uncommitted, by
# measurement on Zen 4: splitting reloads of older stores only adds work.
RECENT_STORES = 16

# Stores and loads a reload can be matched against, by the bytes they move.
_VECTOR_LOADS = (x86.ops.DM_VmovupdOp, x86.ops.DM_VmovupsOp)
_STORE_BYTES = {
    x86.ops.MS_VmovsdOp: 8,
    x86ext.MS_VmovssOp: 4,
}


def _stored_bytes(op: Operation) -> int | None:
    if isinstance(op, (x86.ops.MS_VmovupdOp, x86.ops.MS_VmovupsOp)):
        value_type = op.source.type
        assert isinstance(value_type, X86VectorRegisterType)
        return value_type.bitwidth() // 8
    if isinstance(op, x86ext.StoreChunkOp):
        return 16 if op.half is None else 8
    return _STORE_BYTES.get(type(op))


def match_reloads_to_stores(func: x86_func.FuncOp) -> None:
    """Split vector loads that no single recent store covers.

    A load takes its data from the store buffer only when the youngest store
    it overlaps wrote all of it; otherwise it waits for the stores to commit,
    which costs more than a DFT this small does. A pass whose stores are
    narrower than the next pass's loads, because its batch ends in narrower
    vectors or it stores lane by lane, makes such loads. Each becomes a load
    of what the youngest store at its start wrote, then 128-bit chunks
    inserted from memory, or built from two 64-bit halves where 64-bit
    stores wrote them.

    Stores commit in order, so a reload of stores older than the last
    ``RECENT_STORES`` waits little, and only once for all such reloads; it
    stays as it is.
    """
    # (pointer, offset, bytes) of every store so far, youngest last.
    stores: list[tuple[SSAValue, int, int]] = []

    def forwards(pointer: SSAValue, start: int, end: int) -> bool:
        for age, store in enumerate(reversed(stores)):
            if store[0] is pointer and store[1] < end and start < store[1] + store[2]:
                return age >= RECENT_STORES or (
                    store[1] <= start and end <= store[1] + store[2]
                )
        return True

    for block in func.body.blocks:
        for op in list(block.ops):
            if (size := _stored_bytes(op)) is not None:
                stores.append((op.memory, op.memory_offset.value.data, size))  # pyright: ignore
                continue
            if not isinstance(op, _VECTOR_LOADS):
                continue
            result_type = op.destination.type
            assert isinstance(result_type, X86VectorRegisterType)
            start = op.memory_offset.value.data
            end = start + result_type.bitwidth() // 8
            if not forwards(op.memory, start, end):
                _split_load(op, result_type, forwards)


def _split_load(
    op: x86.ops.DM_VmovupdOp | x86.ops.DM_VmovupsOp,
    result_type: X86VectorRegisterType,
    forwards: Callable[[SSAValue, int, int], bool],
) -> None:
    """Load ``op``'s bytes in pieces that each forward from one store."""
    pointer, start = op.memory, op.memory_offset.value.data
    size = result_type.bitwidth() // 8
    new_ops: list[Operation] = []

    def chunk(offset: int) -> SSAValue | None:
        """A 128-bit chunk built from two halves, or ``None`` to insert it
        straight from memory."""
        if forwards(pointer, offset, offset + 16):
            return None
        if not all(forwards(pointer, h, h + 8) for h in (offset, offset + 8)):
            raise PassFailedException("cannot match a reload to its stores")
        low = x86ext.DM_VmovsdOp(
            pointer, offset, destination=SSERegisterType.unallocated()
        )
        high = x86ext.VmovhpdMemOp(
            low.destination,
            pointer,
            offset + 8,
            destination=SSERegisterType.unallocated(),
        )
        new_ops.extend((low, high))
        return high.destination

    # The widest prefix that forwards from one store.
    first = next(
        (w for w in (32, 16) if w < size and forwards(pointer, start, start + w)), 16
    )
    if first == 16 and (built := chunk(start)) is not None:
        value = built
    else:
        load = type(op)(
            pointer, start, destination=vector_type(8 * first).unallocated()
        )
        new_ops.append(load)
        value = load.destination
    for offset in range(start + first, start + size, 16):
        index = (offset - start) // 16
        destination = type(result_type).unallocated()
        built = chunk(offset)
        insert = (
            x86ext.InsertChunkMemOp(
                value, pointer, offset, index, destination=destination
            )
            if built is None
            else x86ext.InsertChunkOp(value, built, index, destination=destination)
        )
        new_ops.append(insert)
        value = insert.destination
    Rewriter.insert_op(new_ops, InsertPoint.before(op))
    op.destination.replace_all_uses_with(value)
    Rewriter.erase_op(op)
