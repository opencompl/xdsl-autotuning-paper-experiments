"""Operations for generating discrete Fourier transforms by recursive expansion.

The DFT of ``n`` points with sign ``s`` is ``X[k] = sum_j x[j] * w^(s*j*k)``
with ``w = exp(2*pi*i/n)``; ``s = -1`` is the forward transform. Nothing is
normalised.

The operations form three levels, each expanded into the next:

- ``fft.dft`` transforms complex numbers in memory, possibly many at once.
  Cooley-Tukey splits it into a smaller DFT and an in-place pass of small
  DFTs with twiddle factors, until every DFT is small enough to hold in
  registers.
- ``fft.dft_reg`` transforms complex numbers held in vector registers. Each
  vector lane is an independent transform, so a vector of ``L`` lanes holds one
  point of ``L`` transforms. The real and imaginary parts sit either in
  separate registers (the split layout) or in adjacent lanes of one register,
  as in memory (the interleaved layout, whose ``L`` lanes are pairs of
  elements). Cooley-Tukey splits it further, renaming registers instead of
  moving data, down to radix-2, 3, 4, 5, 7, 8, and 16 base cases.
- ``fft.twiddle`` and ``fft.transpose`` are the register-level glue: the
  twiddle factors between Cooley-Tukey stages, and the transposes that move a
  transform's points between lanes and registers so a single transform can be
  vectorized.
"""

from collections.abc import Sequence
from enum import StrEnum

from xdsl.dialects.builtin import (
    I64,
    DenseArrayBase,
    Float32Type,
    Float64Type,
    IntegerAttr,
    StringAttr,
    UnitAttr,
    i64,
)
from xdsl.dialects.x86.registers import GeneralRegisterType, X86VectorRegisterType
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import MemoryReadEffect, MemoryWriteEffect, Pure
from xdsl.utils.exceptions import VerifyException


class DftLayout(StrEnum):
    INTERLEAVED = "interleaved"
    """Each complex number is a (real, imaginary) pair of adjacent elements."""


class RegisterLayout(StrEnum):
    """How complex numbers sit in vector registers."""

    SPLIT = "split"
    """Real parts in one register, imaginary parts in another."""

    INTERLEAVED = "interleaved"
    """(real, imaginary) pairs of adjacent lanes, as in memory."""

    @property
    def registers_per_point(self) -> int:
        return 2 if self is RegisterLayout.SPLIT else 1


Point = tuple[SSAValue, ...]
"""One complex value per lane: a real and an imaginary register in the split
layout, a single register in the interleaved one."""


def _i64(value: int) -> IntegerAttr:
    return IntegerAttr(value, i64)


def _array(values: Sequence[int]) -> DenseArrayBase:
    return DenseArrayBase.from_list(i64, tuple(values))


@irdl_op_definition
class DftOp(IRDLOperation):
    """A batch of DFTs over complex numbers in memory.

    For every batch index ``b`` (one index per entry of ``batch``) this computes

        out[o(b) + k*ostride] = sum_j in[i(b) + j*istride] * t(j, b) * w^(sign*j*k)

    where ``i(b) = sum_d b[d] * idist[d]`` and ``o(b) = sum_d b[d] * odist[d]``.
    Strides and distances count complex numbers, not scalars.

    ``t(j, b)`` is 1 unless ``twiddle_n`` is non-zero, in which case it is the
    twiddle factor ``exp(2*pi*i * sign*j*b[twiddle_dim] / twiddle_n)``. The
    in-place passes of a Cooley-Tukey split use it, as FFTW's twiddle codelets
    do.

    The input and output may be the same buffer, as long as every transform
    reads exactly the locations it writes.
    """

    name = "fft.dft"

    input = operand_def(GeneralRegisterType)
    output = operand_def(GeneralRegisterType)

    n = prop_def(IntegerAttr[I64])
    sign = prop_def(IntegerAttr[I64])
    istride = prop_def(IntegerAttr[I64])
    ostride = prop_def(IntegerAttr[I64])
    batch = prop_def(DenseArrayBase[I64])
    idist = prop_def(DenseArrayBase[I64])
    odist = prop_def(DenseArrayBase[I64])
    twiddle_n = prop_def(IntegerAttr[I64])
    twiddle_dim = prop_def(IntegerAttr[I64])
    layout = prop_def(StringAttr)
    datatype = prop_def(Float32Type | Float64Type)

    irdl_options = (ParsePropInAttrDict(),)
    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())

    assembly_format = (
        "$input `,` $output attr-dict `:` `(` type($input) `,` type($output) `)`"
    )

    def __init__(
        self,
        input: SSAValue,
        output: SSAValue,
        *,
        n: int,
        sign: int,
        istride: int,
        ostride: int,
        batch: Sequence[int] = (),
        idist: Sequence[int] = (),
        odist: Sequence[int] = (),
        twiddle_n: int = 0,
        twiddle_dim: int = 0,
        layout: DftLayout = DftLayout.INTERLEAVED,
        datatype: Float32Type | Float64Type,
    ):
        super().__init__(
            operands=[input, output],
            properties={
                "n": _i64(n),
                "sign": _i64(sign),
                "istride": _i64(istride),
                "ostride": _i64(ostride),
                "batch": _array(batch),
                "idist": _array(idist),
                "odist": _array(odist),
                "twiddle_n": _i64(twiddle_n),
                "twiddle_dim": _i64(twiddle_dim),
                "layout": StringAttr(layout),
                "datatype": datatype,
            },
        )

    @property
    def batch_values(self) -> tuple[int, ...]:
        return self.batch.get_values()

    @property
    def idist_values(self) -> tuple[int, ...]:
        return self.idist.get_values()

    @property
    def odist_values(self) -> tuple[int, ...]:
        return self.odist.get_values()

    def verify_(self) -> None:
        if self.n.value.data <= 0:
            raise VerifyException("n must be positive")
        if self.sign.value.data not in (-1, 1):
            raise VerifyException("sign must be -1 or 1")
        if not (
            len(self.batch_values) == len(self.idist_values) == len(self.odist_values)
        ):
            raise VerifyException("batch, idist, and odist must have equal lengths")
        if any(extent <= 0 for extent in self.batch_values):
            raise VerifyException("batch extents must be positive")
        if self.twiddle_n.value.data < 0:
            raise VerifyException("twiddle_n must be non-negative")
        if self.twiddle_n.value.data and not (
            0 <= self.twiddle_dim.value.data < len(self.batch_values)
        ):
            raise VerifyException("twiddle_dim must name a batch dimension")
        DftLayout(self.layout.data)


@irdl_op_definition
class DftRegOp(IRDLOperation):
    """A DFT of ``n`` points per vector lane, held in registers.

    The operands and results are ``n`` points each. In the split layout they
    are ``re0, im0, re1, im1, ...``: point ``j`` of the transform in lane ``l``
    is lane ``l`` of ``re_j + i * im_j``. In the interleaved layout each point
    is one register, whose lane pair ``l`` is that point of transform ``l``.
    """

    name = "fft.dft_reg"

    inputs = var_operand_def(X86VectorRegisterType)
    outputs = var_result_def(X86VectorRegisterType)

    n = prop_def(IntegerAttr[I64])
    sign = prop_def(IntegerAttr[I64])
    layout = prop_def(StringAttr)
    datatype = prop_def(Float32Type | Float64Type)

    irdl_options = (ParsePropInAttrDict(),)
    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict `:` functional-type($inputs, $outputs)"

    def __init__(
        self,
        points: Sequence[Point],
        *,
        sign: int,
        layout: RegisterLayout = RegisterLayout.SPLIT,
        datatype: Float32Type | Float64Type,
    ):
        inputs = [value for point in points for value in point]
        super().__init__(
            operands=[inputs],
            result_types=[[value.type for value in inputs]],
            properties={
                "n": _i64(len(points)),
                "sign": _i64(sign),
                "layout": StringAttr(layout),
                "datatype": datatype,
            },
        )

    @property
    def register_layout(self) -> RegisterLayout:
        return RegisterLayout(self.layout.data)

    def _points(self, values: Sequence[SSAValue]) -> list[Point]:
        width = self.register_layout.registers_per_point
        return [tuple(values[i : i + width]) for i in range(0, len(values), width)]

    def input_points(self) -> list[Point]:
        return self._points(list(self.inputs))

    def output_points(self) -> list[Point]:
        return self._points(list(self.outputs))

    def verify_(self) -> None:
        width = RegisterLayout(self.layout.data).registers_per_point
        if len(self.inputs) != width * self.n.value.data:
            raise VerifyException(f"expected {width} registers per point")
        if self.sign.value.data not in (-1, 1):
            raise VerifyException("sign must be -1 or 1")


@irdl_op_definition
class TwiddleOp(IRDLOperation):
    """Multiply lane ``l`` of a complex vector by ``exp(2*pi*i * e[l] / n)``.

    The vector is a point of either register layout: a real and an imaginary
    register, or one interleaved register. ``exponents`` holds one exponent
    per lane, or a single exponent shared by every lane.

    Per-lane factors of an interleaved vector come from the constant pool,
    where ``compact`` stores each factor once as a complex number instead of
    as two vectors of duplicated parts: half the bytes, two more shuffles.
    """

    name = "fft.twiddle"

    inputs = var_operand_def(X86VectorRegisterType)
    outputs = var_result_def(X86VectorRegisterType)

    n = prop_def(IntegerAttr[I64])
    exponents = prop_def(DenseArrayBase[I64])
    datatype = prop_def(Float32Type | Float64Type)
    compact = opt_prop_def(UnitAttr)

    irdl_options = (ParsePropInAttrDict(),)
    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict `:` functional-type($inputs, $outputs)"

    def __init__(
        self,
        point: Point,
        *,
        n: int,
        exponents: Sequence[int],
        datatype: Float32Type | Float64Type,
        compact: bool = False,
    ):
        super().__init__(
            operands=[point],
            result_types=[[value.type for value in point]],
            properties={
                "n": _i64(n),
                "exponents": _array([e % n for e in exponents]),
                "datatype": datatype,
                "compact": UnitAttr() if compact else None,
            },
        )

    def verify_(self) -> None:
        if len(self.inputs) not in (1, 2):
            raise VerifyException(
                "expected a real and an imaginary register, or one interleaved"
            )
        if self.n.value.data <= 0:
            raise VerifyException("n must be positive")


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """Transpose ``L`` vectors of ``L`` lanes: lane ``i`` of result ``k`` is lane
    ``k`` of operand ``i``. A lane is a ``1/L`` of the register, so it is a
    complex number in the interleaved layout."""

    name = "fft.transpose"

    inputs = var_operand_def(X86VectorRegisterType)
    outputs = var_result_def(X86VectorRegisterType)

    datatype = prop_def(Float32Type | Float64Type)

    irdl_options = (ParsePropInAttrDict(),)
    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict `:` functional-type($inputs, $outputs)"

    def __init__(
        self, inputs: Sequence[SSAValue], *, datatype: Float32Type | Float64Type
    ):
        super().__init__(
            operands=[inputs],
            result_types=[[value.type for value in inputs]],
            properties={"datatype": datatype},
        )

    def verify_(self) -> None:
        if len(self.inputs) != len(self.outputs):
            raise VerifyException("expected as many results as operands")


FFT = Dialect("fft", [DftOp, DftRegOp, TwiddleOp, TransposeOp], [])
