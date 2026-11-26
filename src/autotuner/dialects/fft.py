"""Operations for generating discrete Fourier transforms by recursive expansion.

The DFT of ``n`` points with sign ``s`` is ``X[k] = sum_j x[j] * w^(s*j*k)``
with ``w = exp(2*pi*i/n)``; ``s = -1`` is the forward transform. Nothing is
normalised. Complex numbers are interleaved (real, imaginary) pairs, in memory
and in vector registers alike, so a register of ``L`` lanes holds ``L``
complex numbers.

Every operation only reads its operands and defines fresh results: none has an
inout pair. An expansion of one of them therefore keeps the register
allocator's structural conditions as long as it defines what it adds before
using it, leaves its operands untouched, and only overwrites registers that it
defined itself and no longer needs.

- ``fft.dft`` transforms a batch of DFTs in memory. Cooley-Tukey splits it
  into two passes over memory, tiling splits its batch into vectors of
  transforms, and a vector of transforms small enough for registers expands
  into loads, ``fft.twiddle``, ``fft.dft_reg``, and stores.
- ``fft.dft_reg`` transforms points held in registers, one transform per lane.
  Cooley-Tukey splits it into smaller ``fft.dft_reg`` and ``fft.twiddle``,
  which only renames registers, down to base cases that expand into adds and
  fused multiply-adds.
- ``fft.twiddle`` multiplies each lane by its own root of unity.
- ``fft.transpose`` exchanges lanes and registers, so that transforms whose
  points are contiguous in memory can each take a lane.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I64,
    DenseArrayBase,
    Float32Type,
    Float64Type,
    IntegerAttr,
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
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import MemoryReadEffect, MemoryWriteEffect, Pure
from xdsl.utils.exceptions import VerifyException


def _i64(value: int) -> IntegerAttr:
    return IntegerAttr(value, i64)


def _array(values: Sequence[int]) -> DenseArrayBase:
    return DenseArrayBase.from_list(i64, tuple(values))


@irdl_op_definition
class DftOp(IRDLOperation):
    """A batch of DFTs over complex numbers in memory.

    For every batch index ``b`` (one index per entry of ``batch``) this computes

        out[o(b) + k*ostride] = sum_j in[i(b) + j*istride] * t(j, b) * w^(sign*j*k)

    where ``i(b) = ioffset + sum_d b[d] * idist[d]`` and ``o(b) = ooffset +
    sum_d b[d] * odist[d]``. Offsets, strides, and distances count complex
    numbers, not scalars.

    ``t(j, b)`` is 1 unless ``twiddle_n`` is non-zero, in which case it is
    ``exp(2*pi*i * sign*j*(twiddle_offset + b[twiddle_dim]) / twiddle_n)``,
    leaving out ``b[twiddle_dim]`` when ``twiddle_dim`` is -1. The in-place
    second pass of a Cooley-Tukey split uses it, as FFTW's twiddle codelets do.
    ``compact_twiddles`` stores each twiddle factor once instead of with its
    parts duplicated: half the bytes, two more shuffles.

    ``registers`` marks a DFT the orchestrator decided to run in register
    kernels, one vector of transforms at a time, rather than to split further
    in memory.

    The input and output may be the same buffer, as long as every transform
    reads exactly the locations it writes. Both pointers are only read.
    """

    name = "fft.dft"

    input = operand_def(GeneralRegisterType)
    output = operand_def(GeneralRegisterType)

    n = prop_def(IntegerAttr[I64])
    sign = prop_def(IntegerAttr[I64])
    istride = prop_def(IntegerAttr[I64])
    ostride = prop_def(IntegerAttr[I64])
    ioffset = prop_def(IntegerAttr[I64])
    ooffset = prop_def(IntegerAttr[I64])
    batch = prop_def(DenseArrayBase[I64])
    idist = prop_def(DenseArrayBase[I64])
    odist = prop_def(DenseArrayBase[I64])
    twiddle_n = prop_def(IntegerAttr[I64])
    twiddle_dim = prop_def(IntegerAttr[I64])
    twiddle_offset = prop_def(IntegerAttr[I64])
    compact_twiddles = opt_prop_def(UnitAttr)
    registers = opt_prop_def(UnitAttr)
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
        ioffset: int = 0,
        ooffset: int = 0,
        batch: Sequence[int] = (),
        idist: Sequence[int] = (),
        odist: Sequence[int] = (),
        twiddle_n: int = 0,
        twiddle_dim: int = -1,
        twiddle_offset: int = 0,
        compact_twiddles: bool = False,
        registers: bool = False,
        datatype: Float32Type | Float64Type,
    ):
        super().__init__(
            operands=[input, output],
            properties={
                "n": _i64(n),
                "sign": _i64(sign),
                "istride": _i64(istride),
                "ostride": _i64(ostride),
                "ioffset": _i64(ioffset),
                "ooffset": _i64(ooffset),
                "batch": _array(batch),
                "idist": _array(idist),
                "odist": _array(odist),
                "twiddle_n": _i64(twiddle_n),
                "twiddle_dim": _i64(twiddle_dim),
                "twiddle_offset": _i64(twiddle_offset),
                "compact_twiddles": UnitAttr() if compact_twiddles else None,
                "registers": UnitAttr() if registers else None,
                "datatype": datatype,
            },
        )

    def verify_(self) -> None:
        if self.n.value.data <= 0:
            raise VerifyException("n must be positive")
        if self.sign.value.data not in (-1, 1):
            raise VerifyException("sign must be -1 or 1")
        batch = self.batch.get_values()
        if (
            not len(batch)
            == len(self.idist.get_values())
            == len(self.odist.get_values())
        ):
            raise VerifyException("batch, idist, and odist must have equal lengths")
        if any(extent <= 0 for extent in batch):
            raise VerifyException("batch extents must be positive")
        if self.twiddle_n.value.data < 0:
            raise VerifyException("twiddle_n must be non-negative")
        if not -1 <= self.twiddle_dim.value.data < len(batch):
            raise VerifyException("twiddle_dim must name a batch dimension or be -1")


@irdl_op_definition
class DftRegOp(IRDLOperation):
    """A DFT of ``n`` points per lane, held in registers.

    Operand ``j`` holds point ``j`` and result ``k`` point ``k`` of every
    transform, the transform in lane ``l`` taking lane ``l`` of each register.
    """

    name = "fft.dft_reg"

    inputs = var_operand_def(X86VectorRegisterType)
    outputs = var_result_def(X86VectorRegisterType)

    sign = prop_def(IntegerAttr[I64])
    datatype = prop_def(Float32Type | Float64Type)

    irdl_options = (ParsePropInAttrDict(),)
    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict `:` functional-type($inputs, $outputs)"

    def __init__(
        self,
        points: Sequence[SSAValue],
        *,
        sign: int,
        datatype: Float32Type | Float64Type,
    ):
        super().__init__(
            operands=[points],
            result_types=[[value.type for value in points]],
            properties={"sign": _i64(sign), "datatype": datatype},
        )

    def verify_(self) -> None:
        if len(self.inputs) != len(self.outputs):
            raise VerifyException("expected one result per point")
        if self.sign.value.data not in (-1, 1):
            raise VerifyException("sign must be -1 or 1")


@irdl_op_definition
class TwiddleOp(IRDLOperation):
    """Multiply lane ``l`` of a register by ``exp(2*pi*i * exponents[l] / n)``.

    Lanes past the end of ``exponents`` are multiplied by 1. ``compact``
    stores each factor once, as a complex number, instead of as two vectors of
    duplicated parts.
    """

    name = "fft.twiddle"

    input = operand_def(X86VectorRegisterType)
    output = result_def(X86VectorRegisterType)

    n = prop_def(IntegerAttr[I64])
    exponents = prop_def(DenseArrayBase[I64])
    compact = opt_prop_def(UnitAttr)
    datatype = prop_def(Float32Type | Float64Type)

    irdl_options = (ParsePropInAttrDict(),)
    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` functional-type($input, $output)"

    def __init__(
        self,
        point: SSAValue,
        *,
        n: int,
        exponents: Sequence[int],
        compact: bool = False,
        datatype: Float32Type | Float64Type,
    ):
        super().__init__(
            operands=[point],
            result_types=[point.type],
            properties={
                "n": _i64(n),
                "exponents": _array([e % n for e in exponents]),
                "compact": UnitAttr() if compact else None,
                "datatype": datatype,
            },
        )

    def verify_(self) -> None:
        if self.n.value.data <= 0:
            raise VerifyException("n must be positive")
        register_type = self.input.type
        assert isinstance(register_type, X86VectorRegisterType)
        lanes = register_type.bitwidth() // (2 * self.datatype.bitwidth)
        if len(self.exponents.get_values()) > max(lanes, 1):
            raise VerifyException("expected at most one exponent per lane")


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """Transpose ``L`` registers of ``L`` complex lanes: lane ``i`` of result
    ``k`` is lane ``k`` of operand ``i``."""

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
        register_type = self.inputs[0].type
        assert isinstance(register_type, X86VectorRegisterType)
        if register_type.bitwidth() != len(self.inputs) * 2 * self.datatype.bitwidth:
            raise VerifyException("expected as many registers as complex lanes")


FFT = Dialect("fft", [DftOp, DftRegOp, TwiddleOp, TransposeOp], [])
