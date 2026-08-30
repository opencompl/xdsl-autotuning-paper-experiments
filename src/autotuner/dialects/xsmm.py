from collections.abc import Sequence
from enum import StrEnum

from xdsl.dialects.builtin import (
    BoolAttr,
    Float32Type,
    Float64Type,
    IntegerAttr,
    StringAttr,
    i64,
)
from xdsl.dialects.x86.registers import (
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

    ``n_start`` is the logical lower bound of the N range. It does not affect
    pointer results, but lets N range splitting preserve absolute loop bounds.

    ``ins`` are additional read-only registers. ``outs`` are additional
    loop-carried registers and have pairwise matching ``out_results``. The
    current schedule uses an ``out`` for an optional AVX-512 mask.
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
    n_start = prop_def(IntegerAttr)
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
        n_start: int,
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
                "n_start": IntegerAttr(n_start, i64),
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

        n_start = self.n_start.value.data
        if n_start < 0:
            raise VerifyException(f"n_start must be non-negative, got {n_start}")

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

    The operation performs ``k`` K steps. ``iterator`` describes the
    pointer-result contract for the currently supported non-transposed inputs:

    - ``m`` advances A by ``m`` elements and leaves B unchanged;
    - ``n`` leaves A unchanged and advances B by ``n * ldb`` elements;
    - ``k`` advances A by ``k * lda`` and B by ``k`` elements;
    - ``none`` passes both matrix pointers through unchanged.

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
    iterator = prop_def(StringAttr)

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
        iterator: MatmulIterator,
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


def matmul_reg_pointer_offsets(
    op: MatmulRegOp, iterator: MatmulIterator | None = None
) -> tuple[int, int]:
    """Return the A and B pointer advances in bytes for an iterator."""
    iterator = MatmulIterator(op.iterator.data) if iterator is None else iterator
    element_size = op.datatype.bitwidth // 8
    match iterator:
        case MatmulIterator.NONE:
            return 0, 0
        case MatmulIterator.M:
            return op.m.value.data * element_size, 0
        case MatmulIterator.N:
            return 0, op.n.value.data * op.ldb.value.data * element_size
        case MatmulIterator.K:
            return (
                op.k.value.data * op.lda.value.data * element_size,
                op.k.value.data * element_size,
            )


XSMM = Dialect(
    "xsmm",
    [MatmulOp, MatmulRegOp],
    [],
)
