from typing import Literal
from xdsl.dialects import x86
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    GeneralRegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import Block, BlockArgument, SSAValue
from xdsl.rewriter import InsertPoint
from autotuner.libxsmm_gemm.generator_common import (
    GPRegMapping,
    LoopLabelTracker,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode, VectorRegT
from autotuner.libxsmm_gemm.libxsmm_main import GEMMPrefetchType


def libxsmm_x86_instruction_open_stream_gemm(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    skip_callee_save: bool,
    prefetch: GEMMPrefetchType,
) -> None:
    args = generated_code.func_op.body.block.args
    a, b, c = args

    # TODO: (Sasha) commented out to allow swapping a with b symbolically
    # generated_code.insert(x86.ops.DS_MovOp(a, destination=gp_reg_mapping.gp_reg_a))
    # generated_code.insert(x86.ops.DS_MovOp(b, destination=gp_reg_mapping.gp_reg_b))
    # generated_code.insert(x86.ops.DS_MovOp(c, destination=gp_reg_mapping.gp_reg_c))

    match prefetch:
        case GEMMPrefetchType.BL2 | GEMMPrefetchType.AL2:
            raise NotImplementedError


def libxsmm_x86_instruction_unified_vec_move_st(
    generated_code: GeneratedCode,
    i_vmove_instr: type[
        x86.ops.MS_VmovapdOp
        | x86.ops.MS_VmovupdOp
        | x86.ops.MS_VmovntpdOp
        | x86.ops.MS_VmovapsOp
        | x86.ops.MS_VmovupsOp
        | x86.ops.MS_VmovntpsOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    source_val: SSAValue[VectorRegT],
    use_masking: bool,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    is_store: Literal[True],
) -> None:
    assert i_vmove_instr is not None
    if generated_code.arch < Arch.LIBXSMM_X86_AVX:
        if use_masking:
            if issubclass(i_vmove_instr, x86.ops.DM_VmovapsOp | x86.ops.DM_VmovapsOp):
                ...
            #         libxsmm_generator_maskedstore_32bit_sse( generated_code, LIBXSMM_X86_GP_REG_RCX, 1, i_vec_reg_number_0, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_mask_reg_number );
            elif issubclass(i_vmove_instr, x86.ops.DM_VmovapdOp | x86.ops.DM_VmovapdOp):
                ...
            #         libxsmm_generator_maskedstore_64bit_sse( generated_code, i_vec_reg_number_0, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_mask_reg_number );
            else:
                assert False, f"Unsupported move op: {i_vmove_instr}"
        else:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                i_vmove_instr,
                base_val=base_val,
                displacement=displacement,
                source_val=source_val,
                mask_val=None,
                use_zero_masking=False,
                is_store=is_store,
            )

    else:
        vmove_instr = i_vmove_instr

        if generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX:
            raise NotImplementedError
        else:
            libxsmm_x86_instruction_vex_evex_mask_mov_st(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                source_val,
                use_masking,
                mask_val,
                is_store,
            )


def libxsmm_x86_instruction_unified_vec_move_ld(
    generated_code: GeneratedCode,
    i_vmove_instr: type[
        x86.ops.DM_VmovapdOp
        | x86.ops.DM_VmovapsOp
        | x86.ops.DM_VmovupsOp
        | x86.ops.DM_VmovupdOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    dest_reg: VectorRegT,
    use_masking: bool,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    is_store: Literal[False],
) -> SSAValue[VectorRegT]:
    assert i_vmove_instr is not None
    if generated_code.arch < Arch.LIBXSMM_X86_AVX:
        if use_masking:
            raise NotImplementedError
        else:
            return libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                i_vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                dest_reg,
                None,
                False,
                is_store,
            )

    else:
        vmove_instr = i_vmove_instr

        if generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX:
            raise NotImplementedError
        else:
            return libxsmm_x86_instruction_vex_evex_mask_mov_ld(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                dest_reg,
                use_masking,
                mask_val,
                is_store,
            )


def libxsmm_x86_instruction_jump_back_to_label(
    generated_code: GeneratedCode,
    jmp_instr: type[x86.ops.ConditionalJumpOperation],
    loop_label_tracker: LoopLabelTracker,
    curr_args: tuple[SSAValue, ...],
) -> tuple[BlockArgument, ...]:
    """
    In contrast to libxsmm, also inserts the comparison instruction.

    ``curr_args`` are the SSA values that are live across the loop, in the same order
    as the loop header block's arguments (i.e. the appropriate ``*Vals.vals``). Returns
    the arguments of the freshly created fallthrough block so the caller can rebuild its
    ``*Vals`` from them.
    """
    dest_block = loop_label_tracker.dest_blocks.pop()

    curr_block = generated_code.current_block
    curr_region = curr_block.parent
    assert curr_region is not None
    fallthrough_block = curr_block.next_block
    assert fallthrough_block is None

    fallthrough_block = Block(arg_types=dest_block.arg_types)
    curr_region.insert_block_after(fallthrough_block, curr_block)

    assert (cmp_op := curr_block.last_op) is not None
    assert len(cmp_op.results) == 1

    generated_code.insert(
        jmp_instr(cmp_op, curr_args, curr_args, dest_block, fallthrough_block)
    )

    # set insert point to fallthrough block
    generated_code.builder.insertion_point = InsertPoint.at_end(fallthrough_block)
    return fallthrough_block.args


def libxsmm_x86_instruction_register_jump_back_label(
    generated_code: GeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    generated_code.insert(x86.ops.LabelOp(f"l{loop_label_tracker.current_loop_number}"))


def libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp]
    | type[x86.ops.DSS_VpxordOp | x86.ops.DSS_VaddpdOp | x86.ops.DSS_VaddpsOp]
    | None,
    src0_reg: VectorRegT,
    src1_reg: VectorRegT,
    dst_reg: VectorRegT,
    src0_val: SSAValue[X86VectorRegisterType] | None,
    src1_val: SSAValue[X86VectorRegisterType] | None,
    dst_val: SSAValue[X86VectorRegisterType] | None,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    mask_cntl: int,
    sae_cntl: int,
    imm8: int | None,
) -> SSAValue[VectorRegT]:
    assert vec_instr is not None

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX
        and (mask_val is not None)
    ), (
        "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: Masking is only available for AVX512!"
    )

    # For zmm0 = zmm0 ^ zmm0 (= 0, essentially), the register is not yet in the context,
    # so we must just get the register.
    if src0_val is None:
        src0_val = generated_code.insert(x86.ops.GetAVXRegisterOp(src0_reg)).result
    if src1_val is None:
        if src1_reg == src0_reg:
            src1_val = src0_val
        else:
            src1_val = generated_code.insert(x86.ops.GetAVXRegisterOp(src1_reg)).result

    # build vXYZpd/ps/sd/ss instruction pure register use
    if generated_code.arch > Arch.LIBXSMM_X86_SSE42:
        # if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) and (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
        if imm8 is not None:
            raise NotImplementedError
        elif issubclass(vec_instr, x86.ops.DSS_Operation):
            res = generated_code.insert(
                vec_instr(src0_val, src1_val, destination=dst_reg)
            ).destination
        elif issubclass(vec_instr, x86.ops.RSS_Operation):
            assert dst_val is not None
            res = generated_code.insert(vec_instr(dst_val, src0_val, src1_val)).register_out
        else:
            assert False, f"Unsupported vec compute op: {vec_instr}"
    else:
        raise NotImplementedError

    return SSAValue.get(res, type=VectorRegT)


def libxsmm_x86_instruction_vec_compute_3reg(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp]
    | type[x86.ops.DSS_VpxordOp | x86.ops.DSS_VaddpdOp | x86.ops.DSS_VaddpsOp]
    | None,
    src0_reg: VectorRegT,
    src1_reg: VectorRegT,
    dst_reg: VectorRegT,
    src0_val: SSAValue[X86VectorRegisterType] | None,
    src1_val: SSAValue[X86VectorRegisterType] | None,
    dst_val: SSAValue[X86VectorRegisterType] | None,
) -> SSAValue[VectorRegT]:
    return libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(
        generated_code,
        vec_instr,
        src0_reg,
        src1_reg,
        dst_reg,
        src0_val,
        src1_val,
        dst_val,
        None,
        0,
        0,
        None,
    )


def libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    gp_reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    use_broadcast: int,
    src1_reg: VectorRegT,
    dst_reg: VectorRegT,
    src1_val: SSAValue[X86VectorRegisterType] | None,
    dst_val: SSAValue[X86VectorRegisterType] | None,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    mask_rnd_exp_cntl: int,
    imm8: int | None,
) -> SSAValue[VectorRegT]:
    assert vec_instr is not None

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX
        and (mask_val is not None)
    ), (
        "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: Masking is only available for AVX512!"
    )

    if generated_code.arch > Arch.LIBXSMM_X86_SSE42:
        if gp_reg_idx is None:
            if src1_val is None:
                src1_val = generated_code.insert(
                    x86.ops.GetAVXRegisterOp(src1_reg)
                ).result
            if dst_val is None:
                dst_val = generated_code.insert(
                    x86.ops.GetAVXRegisterOp(dst_reg)
                ).result

            match vec_instr:
                case x86.ops.RSS_Vfmadd231pdOp:
                    mem_instr = x86.ops.RSM_Vfmadd231pdOp
                case x86.ops.RSS_Vfmadd231psOp:
                    mem_instr = x86.ops.RSM_Vfmadd231psOp
                case _:
                    assert False, f"Unsupported vec compute mem op: {vec_instr}"

            res = generated_code.insert(
                mem_instr(
                    dst_val,
                    src1_val,
                    base_val,
                    displacement,
                    broadcast=bool(use_broadcast),
                )
            ).register_out
            return SSAValue.get(res, type=VectorRegT)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def libxsmm_x86_instruction_vec_compute_mem_2reg(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    gp_reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    use_broadcast: int,
    src1_reg: VectorRegT,
    dst_reg: VectorRegT,
    src1_val: SSAValue[X86VectorRegisterType] | None,
    dst_val: SSAValue[X86VectorRegisterType] | None,
) -> SSAValue[VectorRegT]:
    return libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(
        generated_code,
        vec_instr,
        base_val,
        gp_reg_idx,
        scale,
        displacement,
        use_broadcast,
        src1_reg,
        dst_reg,
        src1_val,
        dst_val,
        None,
        0,
        None,
    )


def libxsmm_x86_instruction_vex_evex_mask_mov_st(
    generated_code: GeneratedCode,
    vmove_instr: type[
        x86.ops.MS_VmovapdOp
        | x86.ops.MS_VmovupdOp
        | x86.ops.MS_VmovntpdOp
        | x86.ops.MS_VmovapsOp
        | x86.ops.MS_VmovupsOp
        | x86.ops.MS_VmovntpsOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    source_val: SSAValue[VectorRegT],
    use_masking: bool,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    is_store: Literal[True],
):
    if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        if use_masking:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                vmove_instr,
                base_val=base_val,
                displacement=displacement,
                source_val=source_val,
                mask_val=mask_val,
                use_zero_masking=not is_store,
                is_store=is_store,
            )
        else:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                vmove_instr,
                base_val=base_val,
                displacement=displacement,
                source_val=source_val,
                mask_val=None,
                use_zero_masking=not is_store,
                is_store=is_store,
            )
    elif generated_code.arch >= Arch.LIBXSMM_X86_AVX:
        if use_masking:
            libxsmm_x86_instruction_vec_mask_move_st(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                source_val,
                mask_val,
                is_store,
            )
        else:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                vmove_instr,
                base_val=base_val,
                displacement=displacement,
                source_val=source_val,
                mask_val=None,
                use_zero_masking=True,
                is_store=is_store,
            )
    else:
        assert False


def libxsmm_x86_instruction_vex_evex_mask_mov_ld(
    generated_code: GeneratedCode,
    vmove_instr: type[
        x86.ops.DM_VmovapdOp
        | x86.ops.DM_VmovapsOp
        | x86.ops.DM_VmovupsOp
        | x86.ops.DM_VmovupdOp
        | x86.ops.DM_VbroadcastsdOp
        | x86.ops.DM_VbroadcastssOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    dest_reg: VectorRegT,
    use_masking: bool,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    is_store: Literal[False],
) -> SSAValue[VectorRegT]:
    if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        if use_masking:
            return libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                dest_reg,
                mask_val,
                not is_store,
                is_store,
            )
        else:
            return libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                dest_reg,
                None,
                not is_store,
                is_store,
            )
    elif generated_code.arch >= Arch.LIBXSMM_X86_AVX:
        if use_masking:
            return libxsmm_x86_instruction_vec_mask_move_ld(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                dest_reg,
                mask_val,
                is_store,
            )
        else:
            return libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                dest_reg,
                None,
                True,
                is_store,
            )
    else:
        assert False


def libxsmm_x86_instruction_vec_mask_move_st(
    generated_code: GeneratedCode,
    vmove_instr: type[
        x86.ops.MS_VmovapdOp
        | x86.ops.MS_VmovupdOp
        | x86.ops.MS_VmovntpdOp
        | x86.ops.MS_VmovapsOp
        | x86.ops.MS_VmovupsOp
        | x86.ops.MS_VmovntpsOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    source_val: SSAValue[VectorRegT],
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    is_store: bool,
):
    raise NotImplementedError


def libxsmm_x86_instruction_vec_mask_move_ld(
    generated_code: GeneratedCode,
    vmove_instr: type[
        x86.ops.DM_VmovapdOp
        | x86.ops.DM_VmovapsOp
        | x86.ops.DM_VmovupsOp
        | x86.ops.DM_VmovupdOp
        | x86.ops.DM_VbroadcastsdOp
        | x86.ops.DM_VbroadcastssOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    dest_reg: VectorRegT,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    is_store: bool,
) -> SSAValue[VectorRegT]:
    raise NotImplementedError


def libxsmm_x86_instruction_vec_move_st(
    generated_code: GeneratedCode,
    vmove_instr: type[
        x86.ops.MS_VmovapdOp
        | x86.ops.MS_VmovupdOp
        | x86.ops.MS_VmovntpdOp
        | x86.ops.MS_VmovapsOp
        | x86.ops.MS_VmovupsOp
        | x86.ops.MS_VmovntpsOp
    ]
    | None,
    *,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    displacement: int,
    source_val: SSAValue[VectorRegT],
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    use_zero_masking: bool,
    is_store: Literal[True],
):
    """
    The is_store is True branches of `libxsmm_x86_instruction_vec_move`
    """
    assert vmove_instr is not None

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and mask_val is not None
    )

    # check zero masking
    assert not (use_zero_masking and (mask_val is not None) and is_store), (
        "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!"
    )

    if mask_val is not None:
        # Use the masking version of the operation
        assert isinstance(source_val.type, x86.registers.AVX512RegisterType), source_val
        assert issubclass(
            vmove_instr,
            x86.ops.MS_VmovapdOp
            | x86.ops.MS_VmovapsOp
            | x86.ops.MS_VmovupsOp
            | x86.ops.MS_VmovupdOp,
        )

        match vmove_instr:
            case x86.ops.MS_VmovapdOp:
                masked_vmove_instr = x86.ops.MSK_VmovapdOp
            case x86.ops.MS_VmovapsOp:
                masked_vmove_instr = x86.ops.MSK_VmovapsOp
            case x86.ops.MS_VmovupsOp:
                masked_vmove_instr = x86.ops.MSK_VmovupsOp
            case x86.ops.MS_VmovupdOp:
                masked_vmove_instr = x86.ops.MSK_VmovupdOp
            case _:
                assert False

        # build vmovpd/ps/sd/ss instruction, load use
        generated_code.insert(
            masked_vmove_instr(
                memory=base_val,
                memory_offset=displacement,
                source=source_val,
                mask_reg=mask_val,
            )
        )
    else:
        generated_code.insert(
            vmove_instr(memory=base_val, source=source_val, memory_offset=displacement)
        )


def libxsmm_x86_instruction_vec_move_ld(
    generated_code: GeneratedCode,
    instruction_set: int,
    vmove_instr: type[
        x86.ops.DM_VmovapdOp
        | x86.ops.DM_VmovapsOp
        | x86.ops.DM_VmovupsOp
        | x86.ops.DM_VmovupdOp
        | x86.ops.DM_VbroadcastsdOp
        | x86.ops.DM_VbroadcastssOp
    ]
    | None,
    base_val: SSAValue[x86.registers.GeneralRegisterType],
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    dest_reg: VectorRegT,
    mask_val: SSAValue[AVX512MaskRegisterType] | None,
    use_zero_masking: bool,
    is_store: Literal[False],
) -> SSAValue[VectorRegT]:
    """
    The is_store is False branches of `libxsmm_x86_instruction_vec_move`
    """
    assert vmove_instr is not None

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and mask_val is not None
    )

    # check zero masking
    assert not (use_zero_masking and (mask_val is not None) and is_store), (
        "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!"
    )

    if not use_zero_masking or mask_val is None:
        zero_flag = None
    else:
        zero_flag = True

    if mask_val is not None:
        # Use the masking version of the operation
        assert isinstance(dest_reg, x86.registers.AVX512RegisterType)
        assert issubclass(
            vmove_instr,
            x86.ops.DM_VmovapdOp
            | x86.ops.DM_VmovapsOp
            | x86.ops.DM_VmovupsOp
            | x86.ops.DM_VmovupdOp,
        )

        match vmove_instr:
            case x86.ops.DM_VmovapdOp:
                masked_vmove_instr = x86.ops.DMK_VmovapdOp
            case x86.ops.DM_VmovapsOp:
                masked_vmove_instr = x86.ops.DMK_VmovapsOp
            case x86.ops.DM_VmovupsOp:
                masked_vmove_instr = x86.ops.DMK_VmovupsOp
            case x86.ops.DM_VmovupdOp:
                masked_vmove_instr = x86.ops.DMK_VmovupdOp
            case _:
                assert False

        # build vmovpd/ps/sd/ss instruction, load use
        res = generated_code.insert(
            masked_vmove_instr(
                memory=base_val,
                memory_offset=displacement,
                destination=dest_reg,
                mask_reg=mask_val,
                z=zero_flag or False,
            )
        ).destination
    else:
        # build vmovpd/ps/sd/ss instruction, load use
        res = generated_code.insert(
            vmove_instr(memory=base_val, memory_offset=displacement, destination=dest_reg)
        ).destination

    return SSAValue.get(res, type=VectorRegT)


def libxsmm_x86_instruction_mask_move_ld(
    generated_code: GeneratedCode,
    mask_instr: type[x86.ops.KS_KMovBOp]
    | type[x86.ops.KS_KMovWOp]
    | type[x86.ops.KS_KMovWOp]
    | type[x86.ops.KS_KMovQOp],
    mask_tmp_val: SSAValue[GeneralRegisterType],
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> SSAValue[x86.registers.AVX512MaskRegisterType]:
    return generated_code.insert(
        mask_instr(
            mask_tmp_val,
            destination=mask_reg,
        )
    ).destination


def libxsmm_x86_instruction_mask_move_st(
    generated_code: GeneratedCode,
    mask_instr: type[x86.ops.DK_KMovBOp]
    | type[x86.ops.DK_KMovWOp]
    | type[x86.ops.DK_KMovWOp]
    | type[x86.ops.DK_KMovQOp],
    gp_reg: x86.registers.GeneralRegisterType,
    mask_val: SSAValue[x86.registers.AVX512MaskRegisterType],
):
    generated_code.insert(
        mask_instr(
            mask_val,
            destination=gp_reg,
        )
    )
