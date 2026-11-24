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

    def fma_raw(
        self, a: SSAValue, b: SSAValue, c: SSAValue, *, kind: FmaKind
    ) -> SSAValue:
        """``±(a * b) ± c``."""
        return self._insert(
            x86ext.VirtualFmaOp(a, b, c, kind=kind, lane_bitwidth=self.element_bits)
        ).results[0]

    def load_pair_duplicated(self, offset: int, bits: int, part: int) -> SSAValue:
        """Load pool pairs at ``offset`` and duplicate element ``part`` (0 or 1)
        of each pair into both of its lanes."""
        destination = vector_type(bits).unallocated()
        if self.double:
            op = x86ext.DMI_VpermilpdOp(
                self.pool.pointer, 0xFF * part, offset, destination=destination
            )
        else:
            op = x86ext.DMI_VpermilpsOp(
                self.pool.pointer,
                0xF5 if part else 0xA0,
                offset,
                destination=destination,
            )
        return self._insert(op).results[0]

    def neg_raw(self, a: SSAValue) -> SSAValue:
        return self._insert(x86ext.VnegOp(a, lane_bitwidth=self.element_bits)).results[
            0
        ]

    def load(self, pointer: SSAValue, offset: int, bits: int) -> SSAValue:
        """Load ``bits`` bits, into the low lanes if that is less than a
        register."""
        destination = vector_type(max(bits, 128)).unallocated()
        if bits < 128:
            op_type = x86ext.DM_VmovsdOp if bits == 64 else x86ext.DM_VmovssOp
        else:
            op_type = x86.ops.DM_VmovupdOp if self.double else x86.ops.DM_VmovupsOp
        return self._insert(op_type(pointer, offset, destination=destination)).results[
            0
        ]

    def store(self, pointer: SSAValue, offset: int, value: SSAValue, bits: int):
        if bits < 128:
            op_type = x86.ops.MS_VmovsdOp if bits == 64 else x86ext.MS_VmovssOp
        else:
            op_type = x86.ops.MS_VmovupdOp if self.double else x86.ops.MS_VmovupsOp
        self._insert(op_type(pointer, value, offset))

    def load_lanewise(
        self, pointer: SSAValue, offsets: Sequence[int], bits: int
    ) -> SSAValue:
        """Load one interleaved complex number per lane, from each of
        ``offsets``, into a register of ``bits`` bits."""
        lanes = len(offsets)
        if lanes == 1:
            return self.load(pointer, offsets[0], bits)
        destination = vector_type(bits)

        def chunk(k: int) -> SSAValue:
            """The 128-bit chunk ``k``: one double or two float complexes."""
            if self.double:
                return self.load(pointer, offsets[k], 128)
            low = self.load(pointer, offsets[2 * k], 64)
            return self._insert(
                x86ext.VmovhpdMemOp(
                    low, pointer, offsets[2 * k + 1], destination=self._like(low)
                )
            ).results[0]

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
            value = self._insert(op).results[0]
        return value

    def store_lanewise(
        self, pointer: SSAValue, offsets: Sequence[int], value: SSAValue, bits: int
    ) -> None:
        """Store each lane's interleaved complex number to its own offset."""
        if len(offsets) == 1:
            self.store(pointer, offsets[0], value, bits)
            return
        for k in range(bits // 128):
            if self.double:
                self._insert(x86ext.StoreChunkOp(value, pointer, offsets[k], k))
                continue
            chunk = (
                value
                if k == 0
                else self._insert(x86ext.ExtractChunkOp(value, k)).results[0]
            )
            for half in (0, 1):
                self._insert(
                    x86ext.StoreChunkOp(chunk, pointer, offsets[2 * k + half], 0, half)
                )

    def swap(self, a: SSAValue) -> SSAValue:
        """Swap the real and imaginary part of every interleaved complex number."""
        op = (
            x86ext.VpermilpdOp(a, 0x55, destination=self._like(a))
            if self.double
            else x86ext.VpermilpsOp(a, 0xB1, destination=self._like(a))
        )
        return self._insert(op).results[0]

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
    def cscale(a: Complex, factor: float) -> Complex:
        return a[0].scaled(factor), a[1].scaled(factor)

    def csum(self, values: Sequence[Complex]) -> Complex:
        return self.sum([v[0] for v in values]), self.sum([v[1] for v in values])

    def cmaterialize(self, a: Complex) -> tuple[SSAValue, ...]:
        return self.materialize(a[0]), self.materialize(a[1])

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


@dataclass(frozen=True)
class Packed:
    """``coefficient * i**rotated * value``, where ``value`` holds interleaved
    complex numbers: (real, imaginary) pairs of adjacent lanes."""

    value: SSAValue
    coefficient: float = 1.0
    rotated: bool = False

    def __neg__(self) -> "Packed":
        return Packed(self.value, -self.coefficient, self.rotated)

    def scaled(self, factor: float) -> "Packed":
        return Packed(self.value, self.coefficient * factor, self.rotated)


class InterleavedAlgebra:
    """The complex arithmetic of the base-case kernels on ``Packed`` values.

    Real coefficients act on both parts alike, so they stay lazy as in
    ``Real``. So does a factor of ``i``, which would swap the parts: adding a
    rotated value to an unrotated one swaps it once (shared by every sum that
    needs it) and lets ``vfmaddsub`` apply the signs, ``a + i*b`` being
    ``a - swap(b)`` in the real lanes and ``a + swap(b)`` in the imaginary.
    """

    def __init__(self, emitter: Emitter):
        self.e = emitter
        self._swapped: dict[SSAValue, SSAValue] = {}

    def swap(self, value: SSAValue) -> SSAValue:
        if value not in self._swapped:
            self._swapped[value] = self.e.swap(value)
        return self._swapped[value]

    def cadd(self, a: Packed, b: Packed) -> Packed:
        if a.rotated == b.rotated:
            result = self.e.add(
                Real(a.value, a.coefficient), Real(b.value, b.coefficient)
            )
            return Packed(result.value, result.coefficient, a.rotated)
        plain, rotated = (a, b) if b.rotated else (b, a)
        # cp*p + i*cq*q = cq * (ratio*p + i*q) = ±cq * (|ratio|*p ± i*q)
        ratio = plain.coefficient / rotated.coefficient
        value = self.e.fma_constant_raw(
            plain.value,
            self.swap(rotated.value),
            self.e.pool.offset([abs(ratio)]),
            broadcast=True,
            kind=FmaKind.MADDSUB if ratio > 0 else FmaKind.MSUBADD,
        )
        return Packed(value, rotated.coefficient if ratio > 0 else -rotated.coefficient)

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
        """``a * exp(2*pi*i * exponent / n)`` for a uniform exponent: one swap
        and one FMA, ``(c + i*s) * a = s * ((c/s) * a + i*a)``."""
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

    def cmaterialize(self, a: Packed) -> tuple[SSAValue, ...]:
        if not a.rotated:
            return (self.e.materialize(Real(a.value, a.coefficient)),)
        # c*i*(re + i*im) = (-c*im) + i*(c*re): the swap times (-c, c).
        value_type = a.value.type
        assert isinstance(value_type, X86VectorRegisterType)
        pairs = value_type.bitwidth() // (2 * self.e.element_bits)
        offset = self.e.pool.offset([-a.coefficient, a.coefficient] * pairs)
        return (self.e.mul_constant_raw(self.swap(a.value), offset, broadcast=False),)
