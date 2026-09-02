from collections.abc import Sequence
from enum import StrEnum

from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.dialects.builtin import (
    BoolAttr,
    Float32Type,
    Float64Type,
    IntegerAttr,
    StringAttr,
    i64,
)
from xdsl.dialects.x86.ops import DSS_Operation
from xdsl.dialects.x86.registers import (
    AVX512RegisterType,
    GeneralRegisterType,
    X86RegisterType,
)
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import MemoryReadEffect, MemoryWriteEffect
from xdsl.utils.exceptions import VerifyException


class MatmulIterator(StrEnum):
    NONE = "none"
    M = "m"
    N = "n"
    K = "k"


class AccumulatorAddOp(
    DSS_Operation[AVX512RegisterType, AVX512RegisterType, AVX512RegisterType]
):
    """A vector add whose destination reuses the accumulator source register."""

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1,), (), ((self.source2, self.destination),)
        )


@irdl_op_definition
class AccumulatorAddpdOp(AccumulatorAddOp):
    name = "xsmm.accumulator.vaddpd"

    def assembly_instruction_name(self) -> str:
        return "vaddpd"


@irdl_op_definition
class AccumulatorAddpsOp(AccumulatorAddOp):
    name = "xsmm.accumulator.vaddps"

    def assembly_instruction_name(self) -> str:
        return "vaddps"


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """A blocked matrix multiplication over complete M, N, and K extents.

    ``iterator`` describes the pointer-result contract of the operation for the
    currently supported non-transposed inputs:

    - ``n`` leaves A unchanged and advances B by ``n * ldb`` and C by
      ``n * ldc`` elements;
    - ``m`` advances A and C by ``m`` elements and leaves B unchanged;
    - ``k`` advances A by ``k * lda`` and B by ``k`` elements and leaves C
      unchanged;
    - ``none`` passes all three matrix pointers through unchanged.

    ``ins`` are additional read-only registers. ``outs`` are additional
    loop-carried registers and have pairwise matching ``out_results``. The
    current schedule uses an ``in`` for an optional AVX-512 mask.
    """

    name = "xsmm.matmul"

    a = operand_def(GeneralRegisterType)
    b = operand_def(GeneralRegisterType)
    c = operand_def(GeneralRegisterType)
    rbp = operand_def(GeneralRegisterType)
    rsp = operand_def(GeneralRegisterType)
    ins = var_operand_def(X86RegisterType)
    outs = var_operand_def(X86RegisterType)

    a_out = result_def(GeneralRegisterType)
    b_out = result_def(GeneralRegisterType)
    c_out = result_def(GeneralRegisterType)
    rbp_out = result_def(GeneralRegisterType)
    rsp_out = result_def(GeneralRegisterType)
    out_results = var_result_def(X86RegisterType)

    m = prop_def(IntegerAttr)
    n = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    lda = prop_def(IntegerAttr)
    ldb = prop_def(IntegerAttr)
    ldc = prop_def(IntegerAttr)
    datatype = prop_def(Float32Type | Float64Type)
    aligned_a = prop_def(BoolAttr)
    aligned_c = prop_def(BoolAttr)
    iterator = prop_def(StringAttr)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())

    def __init__(
        self,
        a: SSAValue,
        b: SSAValue,
        c: SSAValue,
        rbp: SSAValue,
        rsp: SSAValue,
        ins: Sequence[SSAValue] = (),
        outs: Sequence[SSAValue] = (),
        *,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        ldc: int,
        datatype: Float32Type | Float64Type,
        aligned_a: bool,
        aligned_c: bool,
        iterator: MatmulIterator,
    ):
        super().__init__(
            operands=(a, b, c, rbp, rsp, ins, outs),
            result_types=(
                a.type,
                b.type,
                c.type,
                rbp.type,
                rsp.type,
                tuple(out.type for out in outs),
            ),
            properties={
                "m": IntegerAttr(m, i64),
                "n": IntegerAttr(n, i64),
                "k": IntegerAttr(k, i64),
                "lda": IntegerAttr(lda, i64),
                "ldb": IntegerAttr(ldb, i64),
                "ldc": IntegerAttr(ldc, i64),
                "datatype": datatype,
                "aligned_a": BoolAttr.from_bool(aligned_a),
                "aligned_c": BoolAttr.from_bool(aligned_c),
                "iterator": StringAttr(iterator),
            },
        )

    def verify_(self) -> None:
        try:
            MatmulIterator(self.iterator.data)
        except ValueError as error:
            allowed = ", ".join(iterator.value for iterator in MatmulIterator)
            raise VerifyException(
                f"iterator must be one of {allowed}, got {self.iterator.data}"
            ) from error

        integer_properties = {
            "m": self.m.value.data,
            "n": self.n.value.data,
            "k": self.k.value.data,
            "lda": self.lda.value.data,
            "ldb": self.ldb.value.data,
            "ldc": self.ldc.value.data,
        }
        for name, value in integer_properties.items():
            if value <= 0:
                raise VerifyException(f"{name} must be positive, got {value}")

        vector_length = 512 // self.datatype.bitwidth
        if self.aligned_a.value.data and self.lda.value.data % vector_length:
            raise VerifyException(
                "aligned A requires lda to be a multiple of the vector length"
            )
        if self.aligned_c.value.data and self.ldc.value.data % vector_length:
            raise VerifyException(
                "aligned C requires ldc to be a multiple of the vector length"
            )

        inputs = (
            self.a,
            self.b,
            self.c,
            self.rbp,
            self.rsp,
            *self.outs,
        )
        outputs = (
            self.a_out,
            self.b_out,
            self.c_out,
            self.rbp_out,
            self.rsp_out,
            *self.out_results,
        )
        if tuple(value.type for value in inputs) != tuple(
            value.type for value in outputs
        ):
            raise VerifyException("operand and result types must match pairwise")


@irdl_op_definition
class MatmulRegOp(IRDLOperation):
    """A blocked, register-resident matrix multiplication body.

    The operation performs ``k`` K steps, advancing A by ``k * lda`` elements
    and B by ``k`` elements for the currently supported non-transposed inputs.

    The frame and stack pointers are passed through. C is register-resident and
    represented by ``outs``, so this operation does not carry a C pointer.

    ``ins`` are additional read-only registers. ``outs`` are additional
    loop-carried registers and have pairwise matching ``out_results``. The
    current schedule uses ``ins`` for an optional mask and ``outs`` for C
    accumulators.
    """

    name = "xsmm.matmul_reg"

    a = operand_def(GeneralRegisterType)
    b = operand_def(GeneralRegisterType)
    rbp = operand_def(GeneralRegisterType)
    rsp = operand_def(GeneralRegisterType)
    ins = var_operand_def(X86RegisterType)
    outs = var_operand_def(X86RegisterType)

    a_out = result_def(GeneralRegisterType)
    b_out = result_def(GeneralRegisterType)
    rbp_out = result_def(GeneralRegisterType)
    rsp_out = result_def(GeneralRegisterType)
    out_results = var_result_def(X86RegisterType)

    m = prop_def(IntegerAttr)
    n = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    lda = prop_def(IntegerAttr)
    ldb = prop_def(IntegerAttr)
    datatype = prop_def(Float32Type | Float64Type)
    aligned_a = prop_def(BoolAttr)

    irdl_options = (
        AttrSizedOperandSegments(as_property=True),
        AttrSizedResultSegments(as_property=True),
    )

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        a: SSAValue,
        b: SSAValue,
        rbp: SSAValue,
        rsp: SSAValue,
        ins: Sequence[SSAValue] = (),
        outs: Sequence[SSAValue] = (),
        *,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        datatype: Float32Type | Float64Type,
        aligned_a: bool,
    ):
        super().__init__(
            operands=(a, b, rbp, rsp, ins, outs),
            result_types=(
                a.type,
                b.type,
                rbp.type,
                rsp.type,
                tuple(out.type for out in outs),
            ),
            properties={
                "m": IntegerAttr(m, i64),
                "n": IntegerAttr(n, i64),
                "k": IntegerAttr(k, i64),
                "lda": IntegerAttr(lda, i64),
                "ldb": IntegerAttr(ldb, i64),
                "datatype": datatype,
                "aligned_a": BoolAttr.from_bool(aligned_a),
            },
        )

    def verify_(self) -> None:
        integer_properties = {
            "m": self.m.value.data,
            "n": self.n.value.data,
            "k": self.k.value.data,
            "lda": self.lda.value.data,
            "ldb": self.ldb.value.data,
        }
        for name, value in integer_properties.items():
            if value <= 0:
                raise VerifyException(f"{name} must be positive, got {value}")

        vector_length = 512 // self.datatype.bitwidth
        if self.aligned_a.value.data and self.lda.value.data % vector_length:
            raise VerifyException(
                "aligned A requires lda to be a multiple of the vector length"
            )

        inputs = (
            self.a,
            self.b,
            self.rbp,
            self.rsp,
            *self.outs,
        )
        outputs = (
            self.a_out,
            self.b_out,
            self.rbp_out,
            self.rsp_out,
            *self.out_results,
        )
        if tuple(value.type for value in inputs) != tuple(
            value.type for value in outputs
        ):
            raise VerifyException("operand and result types must match pairwise")


XSMM = Dialect(
    "xsmm",
    [AccumulatorAddpdOp, AccumulatorAddpsOp, MatmulOp, MatmulRegOp],
    [],
)
