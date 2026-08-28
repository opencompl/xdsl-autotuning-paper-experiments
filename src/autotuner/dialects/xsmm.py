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
    AVX512MaskRegisterType,
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
    opt_operand_def,
    opt_result_def,
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


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """A blocked matrix multiplication over complete M, N, and K extents.

    ``iterator`` describes the pointer-result contract of the operation for the
    currently supported non-transposed inputs:

    - ``n`` leaves A unchanged and advances B by ``n * ldb`` and C by
      ``n * ldc`` elements;
    - ``m`` advances A and C by ``m`` elements and leaves B unchanged;
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
class MatmulKOp(IRDLOperation):
    """A blocked, register-resident matrix multiplication K body.

    The operation performs ``k_blocking`` K steps. For the currently supported
    non-transposed inputs, this advances A by ``k_blocking * lda`` elements and B
    by ``k_blocking`` elements. The corresponding output operands expose those
    advanced pointers; C and the frame and stack pointers are passed through.

    Transformations that tile this operation must preserve these pointer results.
    In particular, whether K is represented by one operation or by a loop of
    smaller operations must not affect ``b_out``.
    """

    name = "xsmm.matmul_k"

    a = operand_def(GeneralRegisterType)
    b = operand_def(GeneralRegisterType)
    c = operand_def(GeneralRegisterType)
    rbp = operand_def(GeneralRegisterType)
    rsp = operand_def(GeneralRegisterType)
    mask = opt_operand_def(AVX512MaskRegisterType)
    accumulators = var_operand_def(AVX512RegisterType)

    a_out = result_def(GeneralRegisterType)
    b_out = result_def(GeneralRegisterType)
    c_out = result_def(GeneralRegisterType)
    rbp_out = result_def(GeneralRegisterType)
    rsp_out = result_def(GeneralRegisterType)
    mask_out = opt_result_def(AVX512MaskRegisterType)
    accumulator_outs = var_result_def(AVX512RegisterType)

    m_blocking = prop_def(IntegerAttr)
    n_blocking = prop_def(IntegerAttr)
    k_blocking = prop_def(IntegerAttr)
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
        c: SSAValue,
        rbp: SSAValue,
        rsp: SSAValue,
        mask: SSAValue | None,
        accumulators: Sequence[SSAValue],
        *,
        m_blocking: int,
        n_blocking: int,
        k_blocking: int,
        lda: int,
        ldb: int,
        datatype: Float32Type | Float64Type,
        aligned_a: bool,
    ):
        super().__init__(
            operands=(a, b, c, rbp, rsp, mask, accumulators),
            result_types=(
                a.type,
                b.type,
                c.type,
                rbp.type,
                rsp.type,
                None if mask is None else mask.type,
                tuple(acc.type for acc in accumulators),
            ),
            properties={
                "m_blocking": IntegerAttr(m_blocking, i64),
                "n_blocking": IntegerAttr(n_blocking, i64),
                "k_blocking": IntegerAttr(k_blocking, i64),
                "lda": IntegerAttr(lda, i64),
                "ldb": IntegerAttr(ldb, i64),
                "datatype": datatype,
                "aligned_a": BoolAttr.from_bool(aligned_a),
            },
        )

    def verify_(self) -> None:
        integer_properties = {
            "m_blocking": self.m_blocking.value.data,
            "n_blocking": self.n_blocking.value.data,
            "k_blocking": self.k_blocking.value.data,
            "lda": self.lda.value.data,
            "ldb": self.ldb.value.data,
        }
        for name, value in integer_properties.items():
            if value <= 0:
                raise VerifyException(f"{name} must be positive, got {value}")

        vector_length = 512 // self.datatype.bitwidth
        m_blocking = self.m_blocking.value.data
        n_blocking = self.n_blocking.value.data
        m_vectors = (m_blocking + vector_length - 1) // vector_length
        expected_accumulators = m_vectors * n_blocking
        if len(self.accumulators) != expected_accumulators:
            raise VerifyException(
                "expected "
                f"{expected_accumulators} accumulators for {m_blocking}x{n_blocking} "
                f"blocking, got {len(self.accumulators)}"
            )

        if len(self.accumulator_outs) != len(self.accumulators):
            raise VerifyException(
                "expected the same number of accumulator operands and results"
            )

        needs_mask = m_blocking % vector_length != 0
        if bool(self.mask) != needs_mask:
            raise VerifyException(
                "expected a mask exactly when m_blocking is not a multiple of the "
                "vector length"
            )
        if bool(self.mask_out) != bool(self.mask):
            raise VerifyException("mask operand and result presence must match")

        if self.aligned_a.value.data and self.lda.value.data % vector_length:
            raise VerifyException(
                "aligned A requires lda to be a multiple of the vector length"
            )

        inputs = (
            self.a,
            self.b,
            self.c,
            self.rbp,
            self.rsp,
            *((self.mask,) if self.mask is not None else ()),
            *self.accumulators,
        )
        outputs = (
            self.a_out,
            self.b_out,
            self.c_out,
            self.rbp_out,
            self.rsp_out,
            *((self.mask_out,) if self.mask_out is not None else ()),
            *self.accumulator_outs,
        )
        if tuple(value.type for value in inputs) != tuple(
            value.type for value in outputs
        ):
            raise VerifyException("operand and result types must match pairwise")


XSMM = Dialect(
    "xsmm",
    [MatmulOp, MatmulKOp],
    [],
)
