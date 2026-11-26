"""Building the replacement of one operation during expansion.

An ``Emitter`` collects the operations that replace one operation, in order,
in a block of their own; ``finish`` hands them back for the rewrite to put in
the operation's place. Everything it knows comes from that operation: the
values it reads are the operation's operands, which it never overwrites, and
the constants it needs it emits itself, beside the code that uses them (see
``x86ext.DataOp``).

Fused multiply-adds are the one destructive instruction the expansions use.
They are staged without naming the input they overwrite, and ``finish`` picks
one that the replacement defined and does not use afterwards, or overwrites a
copy of the addend when there is none. Since only the replacement's own
values are overwritten, its operands stay read-only and every inout pair it
creates links a value to its last use.

``Packed`` and ``InterleavedAlgebra`` are the complex arithmetic the
base-case kernels are written in (see ``kernels``), with signs, real scales,
and factors of ``i`` kept lazy so that they fold into the adds and FMAs that
consume them.
"""

import math
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects import x86
from xdsl.dialects.builtin import (
    I64,
    Float32Type,
    Float64Type,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    i64,
)
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512RegisterType,
    SSERegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint, Rewriter

from autotuner.dialects import x86ext
from autotuner.dialects.x86ext import Address, DataOp, FmaKind
from autotuner.fft.lanes import Shuffle

Datatype = Float32Type | Float64Type


def vector_type(bits: int) -> type[X86VectorRegisterType]:
    return {128: SSERegisterType, 256: AVX2RegisterType, 512: AVX512RegisterType}[bits]


def _vector(value: SSAValue) -> SSAValue[X86VectorRegisterType]:
    return SSAValue.get(value, type=X86VectorRegisterType)


def _like(value: SSAValue) -> X86VectorRegisterType:
    value_type = value.type
    assert isinstance(value_type, X86VectorRegisterType)
    return type(value_type).unallocated()


def _bitwidth(value: SSAValue) -> int:
    value_type = value.type
    assert isinstance(value_type, X86VectorRegisterType)
    return value_type.bitwidth()


class Labels:
    """Labels for constants, unique within a module."""

    PREFIX = ".Lfft"

    def __init__(self, module: ModuleOp):
        used = [
            int(match.group(1))
            for op in module.walk()
            if isinstance(op, DataOp)
            and (match := re.fullmatch(rf"\{self.PREFIX}(\d+)", op.label.data))
        ]
        self._next = max(used, default=-1) + 1

    def fresh(self) -> str:
        label = f"{self.PREFIX}{self._next}"
        self._next += 1
        return label


# region: staged fused multiply-adds


@irdl_op_definition
class _StagedFmaOp(IRDLOperation):
    """``±(a * b) ± c``, before ``finish`` picks the input it overwrites."""

    name = "fft_staged.fma"

    a = operand_def(X86VectorRegisterType)
    b = operand_def(X86VectorRegisterType)
    c = operand_def(X86VectorRegisterType)
    result = result_def(X86VectorRegisterType)
    kind = prop_def(StringAttr)


@irdl_op_definition
class _StagedFmaConstantOp(IRDLOperation):
    """``±(a * [symbol]) ± c``, before ``finish`` picks the input it
    overwrites."""

    name = "fft_staged.fma_constant"

    a = operand_def(X86VectorRegisterType)
    c = operand_def(X86VectorRegisterType)
    result = result_def(X86VectorRegisterType)
    kind = prop_def(StringAttr)
    symbol = prop_def(StringAttr)
    broadcast = prop_def(IntegerAttr[I64])


# endregion


class Emitter:
    """Collects the operations replacing one operation."""

    def __init__(self, datatype: Datatype, labels: Labels):
        self.block = Block()
        self.datatype = datatype
        self.labels = labels
        self._constants: dict[tuple[int, ...], Address] = {}

    @property
    def double(self) -> bool:
        return isinstance(self.datatype, Float64Type)

    @property
    def element_bits(self) -> int:
        return self.datatype.bitwidth

    @property
    def element_size(self) -> int:
        return self.element_bits // 8

    def add_op(self, op: Operation) -> Operation:
        self.block.add_op(op)
        return op

    def _result(self, op: Operation) -> SSAValue:
        return self.add_op(op).results[0]

    # region: constants

    def constant(self, values: Sequence[float]) -> Address:
        """The address of a vector of constants (or of one, to broadcast),
        emitted where this replacement goes unless it already has it."""
        if self.double:
            words = tuple(struct.unpack("<Q", struct.pack("<d", v))[0] for v in values)
        else:
            words = tuple(struct.unpack("<I", struct.pack("<f", v))[0] for v in values)
        if words not in self._constants:
            size = len(words) * self.element_size
            label = self.labels.fresh()
            # The linker keeps one copy of equal constants of a .rodata.cst
            # section, so other expansions emitting the same are free.
            self.add_op(
                DataOp(
                    label,
                    words,
                    word_bits=self.element_bits,
                    alignment=int(math.log2(size)),
                    section=f".rodata.cst{size}",
                )
            )
            self._constants[words] = Address(None, symbol=label)
        return self._constants[words]

    # endregion

    # region: arithmetic

    def add(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op_type = x86.ops.DSS_VaddpdOp if self.double else x86.ops.DSS_VaddpsOp
        return self._result(op_type(_vector(a), _vector(b), destination=_like(a)))

    def sub(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op_type = x86ext.VsubpdOp if self.double else x86ext.VsubpsOp
        return self._result(op_type(_vector(a), _vector(b), destination=_like(a)))

    def mul(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op_type = x86ext.VmulpdOp if self.double else x86ext.VmulpsOp
        return self._result(op_type(_vector(a), _vector(b), destination=_like(a)))

    def mul_constant(
        self, a: SSAValue, values: Sequence[float], *, broadcast: bool
    ) -> SSAValue:
        op_type = x86ext.VmulpdMemOp if self.double else x86ext.VmulpsMemOp
        return self._result(
            op_type(a, self.constant(values), broadcast=broadcast, destination=_like(a))
        )

    def fma(self, a: SSAValue, b: SSAValue, c: SSAValue, *, kind: FmaKind) -> SSAValue:
        """``±(a * b) ± c``."""
        return self._result(
            _StagedFmaOp.build(
                operands=[a, b, c],
                result_types=[a.type],
                properties={"kind": StringAttr(kind)},
            )
        )

    def fma_constant(
        self,
        a: SSAValue,
        c: SSAValue,
        values: Sequence[float],
        *,
        broadcast: bool,
        kind: FmaKind,
    ) -> SSAValue:
        """``±(a * constant) ± c``."""
        symbol = self.constant(values).symbol
        assert symbol is not None
        return self._result(
            _StagedFmaConstantOp.build(
                operands=[a, c],
                result_types=[a.type],
                properties={
                    "kind": StringAttr(kind),
                    "symbol": StringAttr(symbol),
                    "broadcast": IntegerAttr(int(broadcast), i64),
                },
            )
        )

    # endregion

    # region: moving lanes

    def swap(self, a: SSAValue) -> SSAValue:
        """Swap the real and imaginary part of every complex number."""
        op = (
            x86ext.VpermilpdOp(a, 0x55, destination=_like(a))
            if self.double
            else x86ext.VpermilpsOp(a, 0xB1, destination=_like(a))
        )
        return self._result(op)

    def load_duplicated(self, values: Sequence[float], bits: int, part: int):
        """Load constant pairs and duplicate element ``part`` (0 or 1) of each
        pair into both of its lanes."""
        destination = vector_type(bits).unallocated()
        if self.double:
            op = x86ext.DMI_VpermilpdOp(
                self.constant(values), 0xFF * part, destination=destination
            )
        else:
            op = x86ext.DMI_VpermilpsOp(
                self.constant(values), 0xF5 if part else 0xA0, destination=destination
            )
        return self._result(op)

    def shuffle(self, shuffle: Shuffle, a: SSAValue, b: SSAValue, bits: int):
        """Emit ``shuffle`` on elements of ``bits`` bits."""
        destination = _like(a)
        a, b = _vector(a), _vector(b)
        match shuffle.kind, bits:
            case "unpack_low", 64:
                op = x86ext.VunpcklpdOp(a, b, destination=destination)
            case "unpack_high", 64:
                op = x86ext.VunpckhpdOp(a, b, destination=destination)
            case "unpack_low", 32:
                op = x86ext.VunpcklpsOp(a, b, destination=destination)
            case "unpack_high", 32:
                op = x86ext.VunpckhpsOp(a, b, destination=destination)
            case "shuffle_ps", _:
                op = x86ext.VshufpsOp(a, b, shuffle.immediate, destination=destination)
            case "shuffle_128", _:
                op = x86ext.Vshuff64x2Op(
                    a, b, shuffle.immediate, destination=destination
                )
            case _:
                raise ValueError(f"no instruction for {shuffle} on {bits} bits")
        return self._result(op)

    # endregion

    # region: memory

    def load(self, pointer: SSAValue, offset: int, bits: int) -> SSAValue:
        """Load ``bits`` bits, into the low lanes if that is less than a
        register."""
        destination = vector_type(max(bits, 128)).unallocated()
        if bits < 128:
            op_type = x86ext.DM_VmovsdOp if bits == 64 else x86ext.DM_VmovssOp
        else:
            op_type = x86.ops.DM_VmovupdOp if self.double else x86.ops.DM_VmovupsOp
        return self._result(op_type(pointer, offset, destination=destination))

    def store(self, pointer: SSAValue, offset: int, value: SSAValue, bits: int):
        if bits < 128:
            op_type = x86.ops.MS_VmovsdOp if bits == 64 else x86ext.MS_VmovssOp
        else:
            op_type = x86.ops.MS_VmovupdOp if self.double else x86.ops.MS_VmovupsOp
        self.add_op(op_type(pointer, value, offset))

    def load_lanewise(
        self, pointer: SSAValue, offsets: Sequence[int], bits: int
    ) -> SSAValue:
        """Load one complex number per lane, from each of ``offsets``, into a
        register of ``bits`` bits."""
        if len(offsets) == 1:
            return self.load(pointer, offsets[0], bits)
        destination = vector_type(bits)

        def chunk(k: int) -> SSAValue:
            """The 128-bit chunk ``k``: one double or two float complexes."""
            if self.double:
                return self.load(pointer, offsets[k], 128)
            low = self.load(pointer, offsets[2 * k], 64)
            return self._result(
                x86ext.VmovhpdMemOp(
                    low, Address(pointer, offsets[2 * k + 1]), destination=_like(low)
                )
            )

        value = chunk(0)
        for k in range(1, bits // 128):
            if self.double:
                op = x86ext.InsertChunkMemOp(
                    value, pointer, offsets[k], k, destination=destination.unallocated()
                )
            else:
                op = x86ext.InsertChunkOp(
                    value, chunk(k), k, destination=destination.unallocated()
                )
            value = self._result(op)
        return value

    def store_lanewise(
        self, pointer: SSAValue, offsets: Sequence[int], value: SSAValue, bits: int
    ) -> None:
        """Store each lane's complex number to its own offset."""
        if len(offsets) == 1:
            self.store(pointer, offsets[0], value, bits)
            return
        for k in range(bits // 128):
            if self.double:
                self.add_op(x86ext.StoreChunkOp(value, pointer, offsets[k], k))
                continue
            chunk = value if k == 0 else self._result(x86ext.ExtractChunkOp(value, k))
            for half in (0, 1):
                self.add_op(
                    x86ext.StoreChunkOp(chunk, pointer, offsets[2 * k + half], 0, half)
                )

    # endregion

    def finish(
        self, results: Sequence[SSAValue] = ()
    ) -> tuple[list[Operation], list[SSAValue]]:
        """The replacement and the values replacing the operation's results,
        given those values as emitted: they live on after it, as its operands
        may, so they are never overwritten."""
        results = list(results)
        ops = list(self.block.ops)
        position = {op: i for i, op in enumerate(ops)}
        live_out = set(results)

        def dies_at(value: SSAValue, op: Operation) -> bool:
            """Whether ``op`` is the last use of a value defined here."""
            return (
                isinstance(value, OpResult)
                and value.op in position
                and value not in live_out
                and all(
                    use.operation in position
                    and position[use.operation] <= position[op]
                    for use in value.uses
                )
            )

        def copy(value: SSAValue) -> Operation:
            op_type = x86.ops.DS_VmovapdOp if self.double else x86.ops.DS_VmovapsOp
            return op_type(value, destination=_like(value))

        for op in ops:
            new_ops: list[Operation] = []
            if isinstance(op, _StagedFmaOp):
                kind = FmaKind(op.kind.data)
                a, b, c = op.a, op.b, op.c
                if dies_at(c, op):
                    register, s1, s2, form = c, a, b, 231
                elif dies_at(a, op):
                    register, s1, s2, form = a, b, c, 213
                elif dies_at(b, op):
                    register, s1, s2, form = b, a, c, 213
                else:
                    new_ops.append(copy(c))
                    register, s1, s2, form = new_ops[-1].results[0], a, b, 231
                new = x86ext.FmaOp(
                    register,
                    s1,
                    s2,
                    kind=kind,
                    form=form,
                    lane_bitwidth=self.element_bits,
                )
            elif isinstance(op, _StagedFmaConstantOp):
                kind = FmaKind(op.kind.data)
                a, c = op.a, op.c
                if dies_at(c, op):
                    register, s1, form = c, a, 231
                elif dies_at(a, op):
                    register, s1, form = a, c, 132
                else:
                    new_ops.append(copy(c))
                    register, s1, form = new_ops[-1].results[0], a, 231
                new = x86ext.FmaMemOp(
                    register,
                    s1,
                    Address(None, symbol=op.symbol.data),
                    broadcast=bool(op.broadcast.value.data),
                    kind=kind,
                    form=form,
                    lane_bitwidth=self.element_bits,
                )
            else:
                continue
            new_ops.append(new)
            for new_op in new_ops:
                position[new_op] = position[op]
            Rewriter.replace_op(op, new_ops, [new.register_out])
            results = [new.register_out if r is op.result else r for r in results]

        ops = list(self.block.ops)
        for op in ops:
            self.block.detach_op(op)
        return ops, results


def replace(
    rewriter: PatternRewriter,
    op: Operation,
    e: Emitter,
    results: Sequence[SSAValue] = (),
) -> None:
    """Replace ``op`` with what ``e`` emitted, ``results`` replacing its
    results."""
    ops, results = e.finish(results)
    rewriter.insert(ops, InsertPoint.before(op))
    rewriter.replace(op, (), results)


# region: complex arithmetic for the kernels


@dataclass(frozen=True)
class Packed:
    """``coefficient * i**rotated * value``, where ``value`` holds complex
    numbers."""

    value: SSAValue
    coefficient: float = 1.0
    rotated: bool = False

    def __neg__(self) -> "Packed":
        return Packed(self.value, -self.coefficient, self.rotated)

    def scaled(self, factor: float) -> "Packed":
        return Packed(self.value, self.coefficient * factor, self.rotated)

    @property
    def unit(self) -> bool:
        return abs(self.coefficient) == 1.0


def _same_magnitude(a: float, b: float) -> bool:
    return math.isclose(abs(a), abs(b), rel_tol=1e-14)


class InterleavedAlgebra:
    """Lazy complex arithmetic on ``Packed`` values.

    Real coefficients act on both parts alike: adding values whose
    coefficients have the same magnitude subtracts for opposite signs, and
    adding any others is a fused multiply-add by a constant. A factor of
    ``i``, which swaps the parts, stays lazy too: adding a rotated value to an
    unrotated one swaps it once (shared by every sum that needs it) and lets
    ``vfmaddsub`` apply the signs, ``a + i*b`` being ``a - swap(b)`` in the
    real lanes and ``a + swap(b)`` in the imaginary.
    """

    def __init__(self, emitter: Emitter):
        self.e = emitter
        self._swapped: dict[SSAValue, SSAValue] = {}

    def swap(self, value: SSAValue) -> SSAValue:
        if value not in self._swapped:
            self._swapped[value] = self.e.swap(value)
        return self._swapped[value]

    def _unit(self, a: Packed) -> Packed:
        if a.unit:
            return a
        value = self.e.mul_constant(a.value, [abs(a.coefficient)], broadcast=True)
        return Packed(value, math.copysign(1.0, a.coefficient), a.rotated)

    def cadd(self, a: Packed, b: Packed) -> Packed:
        if a.rotated != b.rotated:
            plain, rotated = (a, b) if b.rotated else (b, a)
            # cp*p + i*cq*q = cq * (ratio*p + i*q) = ±cq * (|ratio|*p ± i*q)
            ratio = plain.coefficient / rotated.coefficient
            value = self.e.fma_constant(
                plain.value,
                self.swap(rotated.value),
                [abs(ratio)],
                broadcast=True,
                kind=FmaKind.MADDSUB if ratio > 0 else FmaKind.MSUBADD,
            )
            return Packed(
                value, rotated.coefficient if ratio > 0 else -rotated.coefficient
            )
        rotated = a.rotated
        if _same_magnitude(a.coefficient, b.coefficient):
            scale = abs(a.coefficient)
            match a.coefficient > 0, b.coefficient > 0:
                case True, True:
                    return Packed(self.e.add(a.value, b.value), scale, rotated)
                case True, False:
                    return Packed(self.e.sub(a.value, b.value), scale, rotated)
                case False, True:
                    return Packed(self.e.sub(b.value, a.value), scale, rotated)
                case _:
                    return Packed(self.e.add(a.value, b.value), -scale, rotated)
        if not b.unit:
            a, b = b, a
        # a carries the non-unit coefficient; make b a unit if it is not.
        b = self._unit(b)
        value = self.e.fma_constant(
            a.value,
            b.value,
            [abs(a.coefficient)],
            broadcast=True,
            kind=FmaKind.of(a.coefficient < 0, b.coefficient < 0),
        )
        return Packed(value, 1.0, rotated)

    def csub(self, a: Packed, b: Packed) -> Packed:
        return self.cadd(a, -b)

    @staticmethod
    def cscale(a: Packed, factor: float) -> Packed:
        return a.scaled(factor)

    def csum(self, values: Sequence[Packed]) -> Packed:
        result = values[0]
        for value in values[1:]:
            result = self.cadd(result, value)
        return result

    @staticmethod
    def times_i(a: Packed, sign: int) -> Packed:
        """``a * sign * i``, which is lazy."""
        if a.rotated:
            return Packed(a.value, -sign * a.coefficient)
        return Packed(a.value, sign * a.coefficient, True)

    def times_root(self, a: Packed, exponent: int, n: int) -> Packed:
        """``a * exp(2*pi*i * exponent / n)``: one swap and one FMA,
        ``(c + i*s) * a = s * ((c/s) * a + i*a)``."""
        exponent %= n
        if exponent == 0:
            return a
        if (4 * exponent) % n == 0:
            quarter = 4 * exponent // n
            if quarter == 2:
                return -a
            return self.times_i(a, 1 if quarter == 1 else -1)
        angle = 2 * math.pi * exponent / n
        c, s = math.cos(angle), math.sin(angle)
        if (8 * exponent) % n == 0:
            c = math.copysign(math.sqrt(0.5), c)
            s = math.copysign(math.sqrt(0.5), s)
        return self.cadd(a.scaled(c), self.times_i(a, 1).scaled(s))

    def materialize(self, a: Packed) -> SSAValue:
        """Emit what ``a`` still owes its coefficient and rotation."""
        if not a.rotated:
            if a.coefficient == 1.0:
                return a.value
            return self.e.mul_constant(a.value, [a.coefficient], broadcast=True)
        # c*i*(re + i*im) = (-c*im) + i*(c*re): the swap times (-c, c).
        pairs = _bitwidth(a.value) // (2 * self.e.element_bits)
        return self.e.mul_constant(
            self.swap(a.value), [-a.coefficient, a.coefficient] * pairs, broadcast=False
        )


# endregion
