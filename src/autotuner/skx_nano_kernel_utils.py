from dataclasses import dataclass
from typing import cast

from xdsl import ir
from xdsl.dialects import builtin, x86
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import (
    MatmulIterator,
    MatmulRegOp,
    matmul_reg_pointer_offsets,
)
from autotuner.nano_kernel import GemmDescriptor, TileSizes

VectorValue = ir.SSAValue[x86.registers.AVX512RegisterType]
PointerValue = ir.SSAValue[x86.registers.GeneralRegisterType]
MaskValue = ir.SSAValue[x86.registers.AVX512MaskRegisterType]


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


def load_vector(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: PointerValue,
    offset: int,
    destination: x86.registers.AVX512RegisterType,
    *,
    aligned: bool,
    mask: MaskValue | None,
) -> VectorValue:
    if mask is None:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.DM_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.DM_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.DM_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.DM_VmovupdOp

        result = rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                destination=destination,
            ),
            insertion_point=insert_point,
        ).destination
    else:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.DMK_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.DMK_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.DMK_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.DMK_VmovupdOp

        result = rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                destination=destination,
                mask_reg=mask,
                z=True,
            ),
            insertion_point=insert_point,
        ).destination
    return cast(VectorValue, result)


def broadcast_scalar(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: PointerValue,
    offset: int,
    destination: x86.registers.AVX512RegisterType,
) -> VectorValue:
    match datatype:
        case builtin.Float32Type():
            broadcast_op_type = x86.ops.DM_VbroadcastssOp
        case builtin.Float64Type():
            broadcast_op_type = x86.ops.DM_VbroadcastsdOp

    result = rewriter.insert(
        broadcast_op_type(
            memory=pointer,
            memory_offset=offset,
            destination=destination,
        ),
        insertion_point=insert_point,
    ).destination
    return cast(VectorValue, result)


def zero_vector(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    register: x86.registers.AVX512RegisterType,
) -> VectorValue:
    register_value = rewriter.insert(
        x86.ops.GetAVXRegisterOp(register), insertion_point=insert_point
    ).result
    result = rewriter.insert(
        x86.ops.DSS_VpxordOp(
            register_value,
            register_value,
            destination=register,
        ),
        insertion_point=insert_point,
    ).destination
    return cast(VectorValue, result)


def multiply_add_memory(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    accumulator: VectorValue,
    a: VectorValue,
    b: PointerValue,
    b_offset: int,
) -> VectorValue:
    match datatype:
        case builtin.Float32Type():
            multiply_add_op_type = x86.ops.RSM_Vfmadd231psOp
        case builtin.Float64Type():
            multiply_add_op_type = x86.ops.RSM_Vfmadd231pdOp

    result = rewriter.insert(
        multiply_add_op_type(
            accumulator,
            a,
            b,
            b_offset,
            broadcast=True,
        ),
        insertion_point=insert_point,
    ).register_out
    return cast(VectorValue, result)


def multiply_add_registers(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    accumulator: VectorValue,
    a: VectorValue,
    b: VectorValue,
) -> VectorValue:
    match datatype:
        case builtin.Float32Type():
            multiply_add_op_type = x86.ops.RSS_Vfmadd231psOp
        case builtin.Float64Type():
            multiply_add_op_type = x86.ops.RSS_Vfmadd231pdOp

    result = rewriter.insert(
        multiply_add_op_type(accumulator, a, b),
        insertion_point=insert_point,
    ).register_out
    return cast(VectorValue, result)


def add_vectors(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    lhs: VectorValue,
    rhs: VectorValue,
    destination: x86.registers.AVX512RegisterType,
) -> VectorValue:
    match datatype:
        case builtin.Float32Type():
            add_op_type = x86.ops.DSS_VaddpsOp
        case builtin.Float64Type():
            add_op_type = x86.ops.DSS_VaddpdOp

    result = rewriter.insert(
        add_op_type(lhs, rhs, destination=destination),
        insertion_point=insert_point,
    ).destination
    return cast(VectorValue, result)


def advance_pointer(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    pointer: PointerValue,
    byte_offset: int,
) -> PointerValue:
    result = rewriter.insert(
        x86.ops.RI_AddOp(pointer, byte_offset),
        insertion_point=insert_point,
    ).register_out
    return cast(PointerValue, result)


def offset_pointer(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    pointer: PointerValue,
    byte_offset: int,
    *,
    emit_zero_sub: bool = False,
) -> PointerValue:
    """Apply a signed byte offset."""
    if byte_offset == 0 and not emit_zero_sub:
        return pointer
    op_type = x86.ops.RI_SubOp if byte_offset <= 0 else x86.ops.RI_AddOp
    result = rewriter.insert(
        op_type(pointer, abs(byte_offset)),
        insertion_point=insert_point,
    ).register_out
    return cast(PointerValue, result)


def apply_matmul_reg_pointer_contract(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    op: MatmulRegOp,
    values: MatmulRegValues,
) -> MatmulRegValues:
    """Normalize a nano-kernel's natural K traversal to the op iterator."""
    desired_a_offset, desired_b_offset = matmul_reg_pointer_offsets(op)
    k_a_offset, k_b_offset = matmul_reg_pointer_offsets(op, MatmulIterator.K)
    # Preserve the schedule's existing B-before-A normalization order.
    b = offset_pointer(rewriter, insert_point, values.b, desired_b_offset - k_b_offset)
    # Delay the independent A normalization until its first same-block use so
    # intervening C stores retain their established instruction order.
    a_insert_point = insert_point
    if (
        a_user := op.a_out.get_user_of_unique_use()
    ) is not None and a_user.parent_block() is op.parent_block():
        a_insert_point = InsertPoint.before(a_user)
    a = offset_pointer(
        rewriter,
        a_insert_point,
        values.a,
        desired_a_offset - k_a_offset,
        emit_zero_sub=op.iterator.data == MatmulIterator.M,
    )
    return MatmulRegValues(
        a,
        b,
        values.rbp,
        values.rsp,
        values.mask,
        values.accumulators,
    )
