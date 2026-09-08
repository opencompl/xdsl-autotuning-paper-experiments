from collections.abc import Mapping
from dataclasses import dataclass

from xdsl import ir
from xdsl.dialects import x86
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulRegOp
from autotuner.instructions import MaskValue, PointerValue, VectorValue
from autotuner.nano_kernel import FloatingPointType, GemmDescriptor, TileSizes

VECTOR_BANK_BITWIDTH: Mapping[type[x86.registers.X86VectorRegisterType], int] = {
    bank: bank.bitwidth()
    for bank in (
        x86.registers.SSERegisterType,
        x86.registers.AVX2RegisterType,
        x86.registers.AVX512RegisterType,
    )
}
"""Width in bits of each x86 vector register bank, narrowest first.

Doubles as the inventory of banks this lowering knows about, so iterating it
walks them in widening order.
"""


def bank_lanes(
    bank: type[x86.registers.X86VectorRegisterType], datatype: FloatingPointType
) -> int:
    """Return how many ``datatype`` elements one ``bank`` register holds."""
    try:
        bitwidth = VECTOR_BANK_BITWIDTH[bank]
    except KeyError:
        banks = ", ".join(known.name for known in VECTOR_BANK_BITWIDTH)
        raise PassFailedException(
            f"unknown x86 vector register bank {bank.name}; banks are {banks}"
        ) from None
    return bitwidth // datatype.bitwidth


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
    mask: MaskValue | None
    accumulators: tuple[VectorValue, ...]

    @property
    def vals(self) -> tuple[ir.SSAValue, ...]:
        return (
            self.a,
            self.b,
            *self.accumulators,
        )


def values_from_op(
    op: MatmulRegOp, bank: type[x86.registers.X86VectorRegisterType]
) -> MatmulRegValues:
    vector_lanes = bank_lanes(bank, op.datatype)
    m_vectors = (op.m.value.data + vector_lanes - 1) // vector_lanes
    expected_accumulators = m_vectors * op.n.value.data
    if len(op.outs) != expected_accumulators:
        raise PassFailedException(
            "SKX matmul_reg expected "
            f"{expected_accumulators} accumulator outs, got {len(op.outs)}"
        )

    needs_mask = op.m.value.data % vector_lanes != 0
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
        mask,
        tuple(
            ir.SSAValue.get(acc, type=x86.registers.X86VectorRegisterType)
            for acc in op.outs
        ),
    )


def vector_register(
    index: int,
    bank: type[x86.registers.X86VectorRegisterType],
    *,
    disable_regalloc: bool,
) -> x86.registers.X86VectorRegisterType:
    if disable_regalloc:
        return bank.unallocated()
    return bank.from_index(index)
