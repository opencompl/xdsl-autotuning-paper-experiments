from collections.abc import Sequence

from xdsl.dialects.builtin import (
    BoolAttr,
    Float32Type,
    Float64Type,
    IntegerAttr,
    i64,
)
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
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
from xdsl.traits import MemoryReadEffect
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class MatmulKOp(IRDLOperation):
    """A blocked, register-resident matrix multiplication K body."""

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
    [MatmulKOp],
    [],
)
