from dataclasses import dataclass
from typing import cast

from xdsl import ir
from xdsl.dialects import builtin, x86
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.nano_kernel import GemmDescriptor, TargetInfo, TileSizes

VectorValue = ir.SSAValue[x86.registers.AVX512RegisterType]
PointerValue = ir.SSAValue[x86.registers.GeneralRegisterType]
MaskValue = ir.SSAValue[x86.registers.AVX512MaskRegisterType]


def supports_skx(descriptor: GemmDescriptor, target: TargetInfo) -> bool:
    return target.arch == "skx" and isinstance(
        descriptor.datatype, builtin.Float32Type | builtin.Float64Type
    )


def tile_sizes_from_op(op: MatmulKOp) -> TileSizes:
    return TileSizes(
        op.m_blocking.value.data,
        op.n_blocking.value.data,
        op.k_blocking.value.data,
    )


def descriptor_from_op(op: MatmulKOp) -> GemmDescriptor:
    return GemmDescriptor(
        m=op.m_blocking.value.data,
        n=op.n_blocking.value.data,
        k=op.k_blocking.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.m_blocking.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a.value.data),
        aligned_c=False,
    )


@dataclass(frozen=True)
class MatmulKValues:
    a: PointerValue
    b: PointerValue
    c: PointerValue
    rbp: PointerValue
    rsp: PointerValue
    mask: MaskValue | None
    accumulators: tuple[VectorValue, ...]

    @property
    def vals(self) -> tuple[ir.SSAValue, ...]:
        mask = () if self.mask is None else (self.mask,)
        return (
            self.a,
            self.b,
            self.c,
            self.rbp,
            self.rsp,
            *mask,
            *self.accumulators,
        )


def values_from_op(op: MatmulKOp) -> MatmulKValues:
    return MatmulKValues(
        ir.SSAValue.get(op.a, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.b, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.c, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.rbp, type=x86.registers.GeneralRegisterType),
        ir.SSAValue.get(op.rsp, type=x86.registers.GeneralRegisterType),
        (
            None
            if op.mask is None
            else ir.SSAValue.get(op.mask, type=x86.registers.AVX512MaskRegisterType)
        ),
        tuple(
            ir.SSAValue.get(acc, type=x86.registers.AVX512RegisterType)
            for acc in op.accumulators
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
