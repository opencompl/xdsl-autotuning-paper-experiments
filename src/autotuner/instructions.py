"""Typed helpers for selecting and emitting x86 instructions."""

from typing import cast

from xdsl import ir
from xdsl.dialects import builtin, x86
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

VectorValue = ir.SSAValue[x86.registers.X86VectorRegisterType]
PointerValue = ir.SSAValue[x86.registers.GeneralRegisterType]
MaskValue = ir.SSAValue[x86.registers.AVX512MaskRegisterType]


def load_mask(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    gp_reg_tmp: x86.registers.GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
    active_lanes: int,
    datatype: builtin.Float64Type | builtin.Float32Type,
) -> MaskValue:
    """Materialize an AVX-512 mask enabling the ``active_lanes`` lowest lanes."""
    match datatype:
        case builtin.Float64Type():
            op_type = x86.ops.KS_KMovBOp
        case builtin.Float32Type():
            op_type = x86.ops.KS_KMovWOp

    mask = (1 << active_lanes) - 1
    mask_tmp_val = rewriter.insert(
        x86.ops.DI_MovOp(mask, destination=gp_reg_tmp), insertion_point=insert_point
    ).destination
    result = rewriter.insert(
        op_type(mask_tmp_val, destination=mask_reg), insertion_point=insert_point
    ).destination
    return cast(MaskValue, result)


def load_vector(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: PointerValue,
    offset: int,
    destination: x86.registers.X86VectorRegisterType,
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


def store_vector(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: PointerValue,
    offset: int,
    source: VectorValue,
    *,
    aligned: bool,
    mask: MaskValue | None,
) -> None:
    if mask is None:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.MS_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.MS_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.MS_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.MS_VmovupdOp

        rewriter.insert(
            move_op_type(memory=pointer, memory_offset=offset, source=source),
            insertion_point=insert_point,
        )
    else:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.MSK_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.MSK_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.MSK_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.MSK_VmovupdOp

        rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                source=source,
                mask_reg=mask,
            ),
            insertion_point=insert_point,
        )


def broadcast_scalar(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: PointerValue,
    offset: int,
    destination: x86.registers.X86VectorRegisterType,
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
    register: x86.registers.X86VectorRegisterType,
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
    destination: x86.registers.X86VectorRegisterType,
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
