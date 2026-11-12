"""Emitting floating-point vector arithmetic, with lazy signs and scales.

The base-case kernels compute with ``Real`` values, a register times a known
real coefficient. Negating or scaling a value only changes its coefficient;
adding two values then picks a subtraction for a negative coefficient and a
fused multiply-add for any other, so that, for instance, a multiplication by
``-i`` or by ``(1 - i) / sqrt(2)`` costs no instruction of its own. A value
leaves the algebra through ``materialize``, which emits whatever the
coefficient still owes.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects import x86
from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.dialects.x86.registers import (
    AVX2RegisterType,
    AVX512RegisterType,
    SSERegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects import x86ext
from autotuner.dialects.x86ext import FmaKind
from autotuner.fft.constants import ConstantPool
from autotuner.fft.lanes import Shuffle

Datatype = Float32Type | Float64Type


def as_vector(value: SSAValue) -> SSAValue[X86VectorRegisterType]:
    """``value``, checked to be a vector register."""
    return SSAValue.get(value, type=X86VectorRegisterType)


def vector_type(bits: int) -> type[X86VectorRegisterType]:
    return {128: SSERegisterType, 256: AVX2RegisterType, 512: AVX512RegisterType}[bits]


@dataclass(frozen=True)
class Real:
    """``coefficient * value``, where ``value`` is a register of lanes."""

    value: SSAValue
    coefficient: float = 1.0

    def __neg__(self) -> "Real":
        return Real(self.value, -self.coefficient)

    def scaled(self, factor: float) -> "Real":
        return Real(self.value, self.coefficient * factor)

    @property
    def unit(self) -> bool:
        return abs(self.coefficient) == 1.0


Complex = tuple[Real, Real]


def _same_magnitude(a: float, b: float) -> bool:
    return math.isclose(abs(a), abs(b), rel_tol=1e-14)


class Emitter:
    """Emits the instructions of one expansion before a fixed operation."""

    def __init__(
        self,
        rewriter: PatternRewriter,
        before: Operation,
        datatype: Datatype,
        pool: ConstantPool,
    ):
        self.rewriter = rewriter
        self.insert_point = InsertPoint.before(before)
        self.datatype = datatype
        self.pool = pool
        self.double = isinstance(datatype, Float64Type)

    @property
    def element_bits(self) -> int:
        return 64 if self.double else 32

    @property
    def element_size(self) -> int:
        return self.element_bits // 8

    def _insert(self, op: Operation) -> Operation:
        return self.rewriter.insert(op, self.insertion_point())

    def insertion_point(self) -> InsertPoint:
        return self.insert_point

    @staticmethod
    def _like(value: SSAValue) -> X86VectorRegisterType:
        value_type = value.type
        assert isinstance(value_type, X86VectorRegisterType)
        return type(value_type).unallocated()

    # region: plain instructions

    def add_raw(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op_type = x86.ops.DSS_VaddpdOp if self.double else x86.ops.DSS_VaddpsOp
        return self._insert(
            op_type(as_vector(a), as_vector(b), destination=self._like(a))
        ).results[0]

    def sub_raw(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op_type = x86ext.VsubpdOp if self.double else x86ext.VsubpsOp
        return self._insert(
            op_type(as_vector(a), as_vector(b), destination=self._like(a))
        ).results[0]

    def mul_raw(self, a: SSAValue, b: SSAValue) -> SSAValue:
        op_type = x86ext.VmulpdOp if self.double else x86ext.VmulpsOp
        return self._insert(
            op_type(as_vector(a), as_vector(b), destination=self._like(a))
        ).results[0]

    def mul_constant_raw(
        self, a: SSAValue, offset: int, *, broadcast: bool
    ) -> SSAValue:
        op_type = x86ext.VmulpdMemOp if self.double else x86ext.VmulpsMemOp
        return self._insert(
            op_type(
                a,
                self.pool.pointer,
                offset,
                broadcast=broadcast,
                destination=self._like(a),
            )
        ).results[0]

    def fma_constant_raw(
        self,
        a: SSAValue,
        c: SSAValue,
        offset: int,
        *,
        broadcast: bool,
        kind: FmaKind,
    ) -> SSAValue:
        """``±(a * constant) ± c``."""
        return self._insert(
            x86ext.VirtualFmaMemOp(
                a,
                c,
                self.pool.pointer,
                offset,
                broadcast=broadcast,
                kind=kind,
                lane_bitwidth=self.element_bits,
            )
        ).results[0]

    def neg_raw(self, a: SSAValue) -> SSAValue:
        return self._insert(x86ext.VnegOp(a, lane_bitwidth=self.element_bits)).results[
            0
        ]

    def load(self, pointer: SSAValue, offset: int, bits: int) -> SSAValue:
        """Load ``bits`` bits, or one element into the low lane if that is less
        than a register."""
        destination = vector_type(max(bits, 128)).unallocated()
        if bits == self.element_bits:
            op_type = x86ext.DM_VmovsdOp if self.double else x86ext.DM_VmovssOp
        else:
            op_type = x86.ops.DM_VmovupdOp if self.double else x86.ops.DM_VmovupsOp
        return self._insert(op_type(pointer, offset, destination=destination)).results[
            0
        ]

    def store(self, pointer: SSAValue, offset: int, value: SSAValue, bits: int):
        if bits == self.element_bits:
            op_type = x86.ops.MS_VmovsdOp if self.double else x86ext.MS_VmovssOp
        else:
            op_type = x86.ops.MS_VmovupdOp if self.double else x86.ops.MS_VmovupsOp
        self._insert(op_type(pointer, value, offset))

    def shuffle(self, shuffle: Shuffle, a: SSAValue, b: SSAValue, bits: int):
        """Emit ``shuffle`` on elements of ``bits`` bits."""
        destination = self._like(a)
        match shuffle.kind, bits:
            case "unpack_low", 64:
                op = x86ext.VunpcklpdOp(
                    as_vector(a), as_vector(b), destination=destination
                )
            case "unpack_high", 64:
                op = x86ext.VunpckhpdOp(
                    as_vector(a), as_vector(b), destination=destination
                )
            case "unpack_low", 32:
                op = x86ext.VunpcklpsOp(
                    as_vector(a), as_vector(b), destination=destination
                )
            case "unpack_high", 32:
                op = x86ext.VunpckhpsOp(
                    as_vector(a), as_vector(b), destination=destination
                )
            case "shuffle_ps", _:
                op = x86ext.VshufpsOp(
                    as_vector(a),
                    as_vector(b),
                    shuffle.immediate,
                    destination=destination,
                )
            case "shuffle_128", _:
                op = x86ext.Vshuff64x2Op(
                    as_vector(a),
                    as_vector(b),
                    shuffle.immediate,
                    destination=destination,
                )
            case _:
                raise ValueError(f"no instruction for {shuffle} on {bits} bits")
        return self._insert(op).results[0]

    # endregion

    # region: lazy algebra

    def materialize(self, a: Real) -> SSAValue:
        """Emit what ``a`` still owes its coefficient."""
        if a.coefficient == 1.0:
            return a.value
        if a.coefficient == -1.0:
            return self.neg_raw(a.value)
        value = self.mul_constant_raw(
            a.value, self.pool.offset([abs(a.coefficient)]), broadcast=True
        )
        return value if a.coefficient > 0 else self.neg_raw(value)

    def _unit(self, a: Real) -> Real:
        if a.unit:
            return a
        value = self.mul_constant_raw(
            a.value, self.pool.offset([abs(a.coefficient)]), broadcast=True
        )
        return Real(value, math.copysign(1.0, a.coefficient))

    def add(self, a: Real, b: Real) -> Real:
        if _same_magnitude(a.coefficient, b.coefficient):
            scale = abs(a.coefficient)
            match a.coefficient > 0, b.coefficient > 0:
                case True, True:
                    return Real(self.add_raw(a.value, b.value), scale)
                case True, False:
                    return Real(self.sub_raw(a.value, b.value), scale)
                case False, True:
                    return Real(self.sub_raw(b.value, a.value), scale)
                case _:
                    return Real(self.add_raw(a.value, b.value), -scale)
        if not b.unit:
            a, b = b, a
        # a carries the non-unit coefficient; make b a unit if it is not.
        b = self._unit(b)
        kind = FmaKind.of(a.coefficient < 0, b.coefficient < 0)
        value = self.fma_constant_raw(
            a.value,
            b.value,
            self.pool.offset([abs(a.coefficient)]),
            broadcast=True,
            kind=kind,
        )
        return Real(value)

    def sub(self, a: Real, b: Real) -> Real:
        return self.add(a, -b)

    def sum(self, values: Sequence[Real]) -> Real:
        result = values[0]
        for value in values[1:]:
            result = self.add(result, value)
        return result

    # endregion

    # region: complex helpers

    def cadd(self, a: Complex, b: Complex) -> Complex:
        return self.add(a[0], b[0]), self.add(a[1], b[1])

    def csub(self, a: Complex, b: Complex) -> Complex:
        return self.sub(a[0], b[0]), self.sub(a[1], b[1])

    @staticmethod
    def times_i(a: Complex, sign: int) -> Complex:
        """``a * sign * i``, which only renames and negates."""
        return (-a[1], a[0]) if sign > 0 else (a[1], -a[0])

    def times_root(self, a: Complex, exponent: int, n: int) -> Complex:
        """``a * exp(2*pi*i * exponent / n)`` for a uniform exponent."""
        exponent %= n
        if exponent == 0:
            return a
        if (4 * exponent) % n == 0:
            quarter = 4 * exponent // n
            if quarter == 2:
                return -a[0], -a[1]
            return self.times_i(a, 1 if quarter == 1 else -1)
        angle = 2 * math.pi * exponent / n
        if (8 * exponent) % n == 0:
            # A multiple of (1 + i) / sqrt(2): add first, scale lazily.
            c = math.copysign(math.sqrt(0.5), math.cos(angle))
            s = math.copysign(math.sqrt(0.5), math.sin(angle))
            re = self.add(a[0].scaled(c / abs(c)), a[1].scaled(-s / abs(s)))
            im = self.add(a[0].scaled(s / abs(s)), a[1].scaled(c / abs(c)))
            return re.scaled(abs(c)), im.scaled(abs(c))
        c, s = math.cos(angle), math.sin(angle)
        re = self.add(a[0].scaled(c), a[1].scaled(-s))
        im = self.add(a[0].scaled(s), a[1].scaled(c))
        return re, im

    # endregion
