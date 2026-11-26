"""The expansions of the ``fft`` operations, each one rewrite of one operation.

Every rewrite replaces one operation in place with operations whose external
inputs are that operation's operands, and whose results replace its results.
None of the ``fft`` operations has an inout pair, and no expansion overwrites
an operand (see ``emit.Emitter``), so each rewrite keeps the register
allocator's structural conditions; the orchestrator in ``plan`` picks the
parameters that keep register demand within the register file.

- ``SplitDft``: Cooley-Tukey in memory, one ``fft.dft`` into two passes,
  or the decision to run it in register kernels.
- ``TileDft``: a pass into vectors of transforms, each one ``fft.dft``.
- ``DftToRegisters``: a vector of transforms into loads, ``fft.transpose``,
  ``fft.twiddle``, ``fft.dft_reg``, and stores.
- ``SplitDftReg``: Cooley-Tukey in registers, renaming instead of moving.
- ``DftRegKernel``: the base cases, as adds and fused multiply-adds.
- ``Twiddle`` and ``Transpose``: multiplies and shuffles.
"""

import math
from dataclasses import dataclass

from xdsl.dialects.x86.registers import X86VectorRegisterType
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.fft import DftOp, DftRegOp, TransposeOp, TwiddleOp
from autotuner.dialects.x86ext import FmaKind
from autotuner.fft import kernels
from autotuner.fft.emit import Emitter, InterleavedAlgebra, Labels, Packed, replace
from autotuner.fft.lanes import transpose_network
from autotuner.fft.plan import (
    DftShape,
    FftStrategy,
    choose_split,
    compact_twiddles,
    is_leaf,
    register_split_radix,
    side_modes,
    tile,
)


def _shape(op: DftOp) -> DftShape:
    return DftShape(
        op.n.value.data,
        op.istride.value.data,
        op.ostride.value.data,
        op.batch.get_values(),
        op.idist.get_values(),
        op.odist.get_values(),
        op.twiddle_n.value.data,
        op.twiddle_dim.value.data,
    )


def _dft(
    like: DftOp,
    shape: DftShape,
    input: SSAValue,
    output: SSAValue,
    *,
    ioffset: int,
    ooffset: int,
    twiddle_offset: int,
    compact: bool,
    registers: bool,
) -> DftOp:
    return DftOp(
        input,
        output,
        n=shape.n,
        sign=like.sign.value.data,
        istride=shape.istride,
        ostride=shape.ostride,
        ioffset=ioffset,
        ooffset=ooffset,
        batch=shape.batch,
        idist=shape.idist,
        odist=shape.odist,
        twiddle_n=shape.twiddle_n,
        twiddle_dim=shape.twiddle_dim,
        twiddle_offset=twiddle_offset,
        compact_twiddles=compact,
        registers=registers,
        datatype=like.datatype,
    )


@dataclass
class SplitDft(RewritePattern):
    """Cooley-Tukey in memory, decimating in time (see ``DftShape.split``):
    ``r`` DFTs of ``n / r`` points from the input to the output, then
    ``n / r`` twiddled DFTs of ``r`` points in place in the output, which
    run in register kernels. When the orchestrator picks no radix, the DFT
    is marked to run in register kernels as it is.
    """

    strategy: FftStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftOp, rewriter: PatternRewriter, /) -> None:
        if op.registers is not None:
            return
        shape = _shape(op)
        if shape.twiddle_n:
            r = None
        else:
            try:
                r = choose_split(shape, self.strategy, op.datatype.bitwidth)
            except ValueError as error:
                raise PassFailedException(str(error)) from error
        if r is None:
            marked = _dft(
                op,
                shape,
                op.input,
                op.output,
                ioffset=op.ioffset.value.data,
                ooffset=op.ooffset.value.data,
                compact=op.compact_twiddles is not None,
                registers=True,
                twiddle_offset=op.twiddle_offset.value.data,
            )
            rewriter.replace(op, (marked,), ())
            return
        first, second = shape.split(r)
        ooffset = op.ooffset.value.data
        rewriter.replace(
            op,
            (
                _dft(
                    op,
                    first,
                    op.input,
                    op.output,
                    ioffset=op.ioffset.value.data,
                    ooffset=ooffset,
                    twiddle_offset=0,
                    compact=False,
                    registers=False,
                ),
                _dft(
                    op,
                    second,
                    op.output,
                    op.output,
                    ioffset=ooffset,
                    ooffset=ooffset,
                    twiddle_offset=0,
                    compact=False,
                    registers=True,
                ),
            ),
            (),
        )


@dataclass
class TileDft(RewritePattern):
    """A pass of register-sized DFTs as a sequence of vectors of transforms,
    each one ``fft.dft`` with no batch or a batch of one vector's lanes."""

    strategy: FftStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftOp, rewriter: PatternRewriter, /) -> None:
        if op.registers is None:
            return
        shape, bits = _shape(op), op.datatype.bitwidth
        if is_leaf(shape, self.strategy, bits):
            return
        compact = op.compact_twiddles is not None or compact_twiddles(
            shape, self.strategy, bits
        )
        rewriter.replace(
            op,
            [
                _dft(
                    op,
                    t.shape,
                    op.input,
                    op.output,
                    ioffset=op.ioffset.value.data + t.ioffset,
                    ooffset=op.ooffset.value.data + t.ooffset,
                    twiddle_offset=op.twiddle_offset.value.data + t.twiddle_offset,
                    compact=compact and bool(t.shape.twiddle_n),
                    registers=True,
                )
                for t in tile(shape, self.strategy, bits)
            ],
            (),
        )


@dataclass
class DftToRegisters(RewritePattern):
    """Load a vector of small DFTs into registers, twiddle, transform, and
    store them."""

    strategy: FftStrategy
    labels: Labels

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftOp, rewriter: PatternRewriter, /) -> None:
        if op.registers is None:
            return
        if not is_leaf(_shape(op), self.strategy, op.datatype.bitwidth):
            return
        e = Emitter(op.datatype, self.labels)
        n, sign = op.n.value.data, op.sign.value.data
        istride, ostride = op.istride.value.data, op.ostride.value.data
        ioffset, ooffset = op.ioffset.value.data, op.ooffset.value.data
        batch = op.batch.get_values()
        lanes = batch[0] if batch else 1
        idist = op.idist.get_values()[0] if batch else 0
        odist = op.odist.get_values()[0] if batch else 0
        complex_bytes = 2 * e.element_size
        bits = lanes * 8 * complex_bytes
        in_lanes = [ioffset + lane * idist for lane in range(lanes)]
        out_lanes = [ooffset + lane * odist for lane in range(lanes)]

        # Load.
        transposed, lanewise = side_modes(n, lanes, istride, idist)
        points: list[SSAValue] = []
        if transposed:
            for k0 in range(0, n, lanes):
                block = [
                    e.load(op.input, (base + k0) * complex_bytes, bits)
                    for base in in_lanes
                ]
                points += e.add_op(TransposeOp(block, datatype=op.datatype)).results
        else:
            for j in range(n):
                offsets = [(base + j * istride) * complex_bytes for base in in_lanes]
                if lanewise:
                    points.append(e.load_lanewise(op.input, offsets, bits))
                else:
                    points.append(e.load(op.input, offsets[0], bits))

        # Twiddle.
        twiddle_n = op.twiddle_n.value.data
        if twiddle_n:
            offset = op.twiddle_offset.value.data
            per_lane = op.twiddle_dim.value.data == 0
            for j in range(1, n):
                exponents = [
                    sign * j * (offset + (lane if per_lane else 0))
                    for lane in range(lanes)
                ]
                if all(x % twiddle_n == 0 for x in exponents):
                    continue
                twiddle = TwiddleOp(
                    points[j],
                    n=twiddle_n,
                    exponents=exponents,
                    compact=op.compact_twiddles is not None,
                    datatype=op.datatype,
                )
                points[j] = e.add_op(twiddle).results[0]

        # Transform.
        results = list(
            e.add_op(DftRegOp(points, sign=sign, datatype=op.datatype)).results
        )

        # Store.
        transposed, lanewise = side_modes(n, lanes, ostride, odist)
        if transposed:
            for k0 in range(0, n, lanes):
                block = e.add_op(
                    TransposeOp(results[k0 : k0 + lanes], datatype=op.datatype)
                ).results
                for base, value in zip(out_lanes, block):
                    e.store(op.output, (base + k0) * complex_bytes, value, bits)
        else:
            for k, value in enumerate(results):
                offsets = [(base + k * ostride) * complex_bytes for base in out_lanes]
                if lanewise:
                    e.store_lanewise(op.output, offsets, value, bits)
                else:
                    e.store(op.output, offsets[0], value, bits)

        replace(rewriter, op, e)


@dataclass
class SplitDftReg(RewritePattern):
    """Cooley-Tukey in registers: renames registers instead of moving data."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftRegOp, rewriter: PatternRewriter, /) -> None:
        n = len(op.inputs)
        r = register_split_radix(n)
        if r is None:
            return
        m = n // r
        sign, datatype = op.sign.value.data, op.datatype
        x = list(op.inputs)
        new_ops = []

        def add(new_op):
            new_ops.append(new_op)
            return new_op.results

        inner = [
            add(
                DftRegOp(
                    [x[j2 + r * j1] for j1 in range(m)], sign=sign, datatype=datatype
                )
            )
            for j2 in range(r)
        ]
        results: list[SSAValue] = [x[0]] * n
        for k1 in range(m):
            column = []
            for j2 in range(r):
                point = inner[j2][k1]
                if (j2 * k1) % n:
                    point = add(
                        TwiddleOp(
                            point, n=n, exponents=[sign * j2 * k1], datatype=datatype
                        )
                    )[0]
                column.append(point)
            outer = add(DftRegOp(column, sign=sign, datatype=datatype))
            for k2, point in enumerate(outer):
                results[k1 + m * k2] = point
        rewriter.replace(op, new_ops, results)


@dataclass
class DftRegKernel(RewritePattern):
    """The nano-kernels: radix 2, 3, 4, 5, 7, 8, and 16 DFTs as adds and FMAs."""

    labels: Labels

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DftRegOp, rewriter: PatternRewriter, /) -> None:
        if register_split_radix(len(op.inputs)) is not None:
            return
        e = Emitter(op.datatype, self.labels)
        algebra = InterleavedAlgebra(e)
        points = [Packed(value) for value in op.inputs]
        results = [
            algebra.materialize(point)
            for point in kernels.dft(algebra, points, op.sign.value.data)
        ]
        replace(rewriter, op, e, results)


def _lanes(value: SSAValue, element_bits: int) -> int:
    register_type = value.type
    assert isinstance(register_type, X86VectorRegisterType)
    return register_type.bitwidth() // (2 * element_bits)


@dataclass
class Twiddle(RewritePattern):
    """Multiply by twiddle factors from vector constants: a swap, a multiply,
    and a fused multiply-add."""

    labels: Labels

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TwiddleOp, rewriter: PatternRewriter, /) -> None:
        e = Emitter(op.datatype, self.labels)
        n, value = op.n.value.data, op.input
        lanes = _lanes(value, e.element_bits)
        exponents = list(op.exponents.get_values())
        if len(exponents) == 1:
            exponents *= lanes
        exponents += [0] * (lanes - len(exponents))
        if len(set(exponents)) == 1:
            # One factor for every lane: multiples of i and of (1 + i)/sqrt(2)
            # come cheaper than a general complex product.
            algebra = InterleavedAlgebra(e)
            result = algebra.times_root(Packed(value), exponents[0], n)
            replace(rewriter, op, e, [algebra.materialize(result)])
            return
        angles = [2 * math.pi * x / n for x in exponents]
        bits = value.type.bitwidth()  # pyright: ignore
        if op.compact is None:
            # (c + is)(a + ib) = (c, c) * (a, b) + (-s, s) * (b, a)
            sines = [v for a in angles for v in (-math.sin(a), math.sin(a))]
            cosines = [math.cos(a) for a in angles for _ in range(2)]
            rotated = e.mul_constant(e.swap(value), sines, broadcast=False)
            result = e.fma_constant(
                value, rotated, cosines, broadcast=False, kind=FmaKind.FMADD
            )
        else:
            pairs = [v for a in angles for v in (math.cos(a), math.sin(a))]
            cosines = e.load_duplicated(pairs, bits, 0)
            sines = e.load_duplicated(pairs, bits, 1)
            # (c + is)(a + ib) = (c, c) * (a, b) -/+ (s, s) * (b, a)
            rotated = e.mul(e.swap(value), sines)
            result = e.fma(value, cosines, rotated, kind=FmaKind.MADDSUB)
        replace(rewriter, op, e, [result])


@dataclass
class Transpose(RewritePattern):
    """Transpose a square block of registers with a network of shuffles."""

    labels: Labels

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter, /) -> None:
        e = Emitter(op.datatype, self.labels)
        lanes = len(op.inputs)
        network = transpose_network(lanes, 2 * e.element_bits)
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
        replace(rewriter, op, e, results)
