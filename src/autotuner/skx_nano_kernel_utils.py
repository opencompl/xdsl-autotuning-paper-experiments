from dataclasses import dataclass

from xdsl import ir
from xdsl.dialects import x86
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulRegOp
from autotuner.instructions import MaskValue, PointerValue, VectorValue
from autotuner.nano_kernel import GemmDescriptor, TileSizes


def tile_sizes_from_op(op: MatmulRegOp) -> TileSizes:
    return TileSizes(
        op.m.value.data,
        op.n.value.data,
        op.k.value.data,
    )


def descriptor_from_op(op: MatmulRegOp) -> GemmDescriptor:
    return GemmDescriptor(
        m=op.m.value.data,
        n=op.n.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.m.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a.value.data),
        aligned_c=False,
    )


@dataclass(frozen=True)
class MatmulRegValues:
    a: PointerValue
    b: PointerValue
    rbp: PointerValue
    rsp: PointerValue
    mask: MaskValue | None
    accumulators: tuple[VectorValue, ...]

    @property
    def vals(self) -> tuple[ir.SSAValue, ...]:
        return (
            self.a,
            self.b,
            self.rbp,
            self.rsp,
            *self.accumulators,
        )


def values_from_op(op: MatmulRegOp) -> MatmulRegValues:
    vector_length = 512 // op.datatype.bitwidth
    m_vectors = (op.m.value.data + vector_length - 1) // vector_length
    expected_accumulators = m_vectors * op.n.value.data
    if len(op.outs) != expected_accumulators:
        raise PassFailedException(
            "SKX matmul_reg expected "
            f"{expected_accumulators} accumulator outs, got {len(op.outs)}"
        )

    needs_mask = op.m.value.data % vector_length != 0
    if len(op.ins) != int(needs_mask):
        raise PassFailedException(
            "SKX matmul_reg expects one mask in exactly when M has a partial vector"
        )

    mask = (
        None
        if not op.ins
        else ir.SSAValue.get(op.ins[0], type=x86.registers.AVX512MaskRegisterType)
    )
    return MatmulRegValues(
        ir.SSAValue.get(op.a, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.b, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.rbp, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.rsp, type=x86.registers.GeneralRegisterType),
        mask,
        tuple(
            ir.SSAValue.get(acc, type=x86.registers.AVX512RegisterType)
            for acc in op.outs
        ),
    )


def vector_register(
    index: int, *, disable_regalloc: bool
) -> x86.registers.AVX512RegisterType:
    if disable_regalloc:
        return x86.registers.AVX512RegisterType.unallocated()
    return x86.registers.AVX512RegisterType.from_index(index)
