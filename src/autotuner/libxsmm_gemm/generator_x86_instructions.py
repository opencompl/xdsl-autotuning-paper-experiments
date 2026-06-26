from typing import Literal
from xdsl.dialects import x86
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    GeneralRegisterType,
)
from xdsl.ir import Block, SSAValue
from xdsl.rewriter import InsertPoint
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, LoopLabelTracker
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    use_masking: bool,
    mask_reg_number: int,
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
                generated_code.arch,
                i_vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                0,
                False,
                is_store,
            )

    else:
        vmove_instr = i_vmove_instr

        if generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX:
            raise NotImplementedError
        else:
            libxsmm_x86_instruction_vex_evex_mask_mov_st(
                generated_code,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                use_masking,
                mask_reg_number,
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    use_masking: bool,
    mask_reg_number: int,
    is_store: Literal[False],
) -> None:
    assert i_vmove_instr is not None
    if generated_code.arch < Arch.LIBXSMM_X86_AVX:
        if use_masking:
            raise NotImplementedError
            if issubclass(i_vmove_instr, x86.ops.DM_VmovapsOp | x86.ops.DM_VmovapsOp):
                #         libxsmm_generator_maskedload_32bit_sse( generated_code, LIBXSMM_X86_GP_REG_RCX, 1, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_vec_reg_number_0, i_mask_reg_number );
                ...
            elif issubclass(i_vmove_instr, x86.ops.DM_VmovapdOp | x86.ops.DM_VmovapdOp):
                #         libxsmm_generator_maskedload_64bit_sse( generated_code, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_vec_reg_number_0, i_mask_reg_number );
                ...
            else:
                assert False, f"Unsupported move op: {i_vmove_instr}"
        else:
            libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                i_vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                0,
                False,
                is_store,
            )

    else:
        vmove_instr = i_vmove_instr

        if generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX:
            raise NotImplementedError
        else:
            libxsmm_x86_instruction_vex_evex_mask_mov_ld(
                generated_code,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                use_masking,
                mask_reg_number,
                is_store,
            )


def libxsmm_x86_instruction_jump_back_to_label(
    generated_code: GeneratedCode,
    jmp_instr: type[x86.ops.ConditionalJumpOperation],
    loop_label_tracker: LoopLabelTracker,
):
    """
    In contrast to libxsmm, also inserts the comparison instruction
    """
    dest_block = loop_label_tracker.dest_blocks.pop()
    curr_vals = generated_code.current_val_by_reg

    curr_args = tuple(curr_vals[arg.type] for arg in dest_block.args)

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

    # set insert point to fallthrough block and update current values
    generated_code.builder.insertion_point = InsertPoint.at_end(fallthrough_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in fallthrough_block.args}


def libxsmm_x86_instruction_register_jump_back_label(
    generated_code: GeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    generated_code.insert(x86.ops.LabelOp(f"l{loop_label_tracker.current_loop_number}"))


def libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None,
    vector_name: Literal["x", "y", "z"],
    reg_number_src0: int,
    reg_number_src1: int,
    reg_number_dst: int,
    mask_reg_number: int,
    mask_cntl: int,
    sae_cntl: int,
    imm8: int | None,
):
    assert vec_instr is not None
    # if ( (libxsmm_x86_instruction_vec_is_hybrid( i_vec_instr )  == 0) and
    #     (libxsmm_x86_instruction_vec_is_regonly( i_vec_instr ) == 0)    ) {
    #     fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unexpected instruction number: 0x%08x\n", i_vec_instr);
    #     LIBXSMM_EXIT_ERROR(generated_code);
    #     return;
    # }

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and mask_reg_number
    ), (
        "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: Masking is only available for AVX512!"
    )

    match vector_name:
        case "x":
            source_type = x86.registers.SSERegisterType
        case "y":
            source_type = x86.registers.AVX2RegisterType
        case "z":
            source_type = x86.registers.AVX512RegisterType
    reg_src0 = source_type.from_index(reg_number_src0)
    reg_src1 = source_type.from_index(reg_number_src1)
    reg_dst = source_type.from_index(reg_number_dst)
    src0 = generated_code.get_val(reg_src0)
    src1 = generated_code.get_val(reg_src1)
    dst = generated_code.get_val(reg_dst)
    assert dst.type == reg_dst

    # build vXYZpd/ps/sd/ss instruction pure register use
    if generated_code.arch > Arch.LIBXSMM_X86_SSE42:
        # if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) and (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
        if imm8 is not None:
            raise NotImplementedError
        else:
            generated_code.insert(vec_instr(dst, src0, src1))
    else:
        raise NotImplementedError


def libxsmm_x86_instruction_vec_compute_3reg(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None,
    vector_name: Literal["x", "y", "z"],
    reg_number_src0: int,
    reg_number_src1: int,
    reg_number_dst: int,
) -> None:
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(
        generated_code,
        vec_instr,
        vector_name,
        reg_number_src0,
        reg_number_src1,
        reg_number_dst,
        0,
        0,
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    use_masking: bool,
    mask_reg_number: int,
    is_store: Literal[True],
):
    if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        if use_masking:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                mask_reg_number,
                not is_store,
                is_store,
            )
        else:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                0,
                not is_store,
                is_store,
            )
    elif generated_code.arch >= Arch.LIBXSMM_X86_AVX:
        if use_masking:
            libxsmm_x86_instruction_vec_mask_move_st(
                generated_code,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                mask_reg_number,
                is_store,
            )
        else:
            libxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                0,
                True,
                is_store,
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    use_masking: bool,
    mask_reg_number: int,
    is_store: Literal[False],
):
    if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        if use_masking:
            libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                mask_reg_number,
                not is_store,
                is_store,
            )
        else:
            libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                0,
                not is_store,
                is_store,
            )
    elif generated_code.arch >= Arch.LIBXSMM_X86_AVX:
        if use_masking:
            libxsmm_x86_instruction_vec_mask_move_ld(
                generated_code,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                mask_reg_number,
                is_store,
            )
        else:
            libxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                gp_reg_base,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                0,
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    vec_reg_mask_0: int,
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    vec_reg_mask_0: int,
    is_store: bool,
):
    raise NotImplementedError


def libxsmm_x86_instruction_vec_move_st(
    generated_code: GeneratedCode,
    instruction_set: int,
    vmove_instr: type[
        x86.ops.MS_VmovapdOp
        | x86.ops.MS_VmovupdOp
        | x86.ops.MS_VmovntpdOp
        | x86.ops.MS_VmovapsOp
        | x86.ops.MS_VmovupsOp
        | x86.ops.MS_VmovntpsOp
    ]
    | None,
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    mask_reg_number: int,
    use_zero_masking: bool,
    is_store: Literal[True],
):
    """
    The is_store is True branches of `libxsmm_x86_instruction_vec_move`
    """
    assert vmove_instr is not None

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and mask_reg_number
    )

    # check zero masking
    assert not (use_zero_masking and mask_reg_number and is_store), (
        "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!"
    )

    match vector_name:
        case "x":
            source_type = x86.registers.SSERegisterType
        case "y":
            source_type = x86.registers.AVX2RegisterType
        case "z":
            source_type = x86.registers.AVX512RegisterType
    source_reg = source_type.from_index(vec_reg_number_0)
    source = generated_code.current_val_by_reg[source_reg]
    base = generated_code.current_val_by_reg[gp_reg_base]

    if mask_reg_number:
        # Use the masking version of the operation
        assert isinstance(source_reg, x86.registers.AVX512RegisterType), source
        assert issubclass(
            vmove_instr,
            x86.ops.MS_VmovapdOp
            | x86.ops.MS_VmovapsOp
            | x86.ops.MS_VmovupsOp
            | x86.ops.MS_VmovupdOp,
        )

        mask_reg = AVX512MaskRegisterType.from_index(mask_reg_number)
        mask = generated_code.current_val_by_reg[mask_reg]
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
                memory=base,
                memory_offset=displacement,
                source=source,
                mask_reg=mask,
            )
        )
    else:
        generated_code.insert(
            vmove_instr(memory=base, source=source, memory_offset=displacement)
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
    gp_reg_base: x86.registers.GeneralRegisterType,
    reg_idx: x86.registers.GeneralRegisterType | None,
    scale: int,
    displacement: int,
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    mask_reg_number: int,
    use_zero_masking: bool,
    is_store: Literal[False],
):
    """
    The is_store is False branches of `libxsmm_x86_instruction_vec_move`
    """
    assert vmove_instr is not None

    # check that we are not masking 'y'
    assert not (
        generated_code.arch < Arch.LIBXSMM_X86_AVX512_VL128_SKX and mask_reg_number
    )

    # check zero masking
    assert not (use_zero_masking and mask_reg_number and is_store), (
        "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!"
    )

    if not use_zero_masking or not mask_reg_number:
        zero_flag = None
    else:
        zero_flag = True

    match vector_name:
        case "x":
            dest_type = x86.registers.SSERegisterType
        case "y":
            dest_type = x86.registers.AVX2RegisterType
        case "z":
            dest_type = x86.registers.AVX512RegisterType
    dest = dest_type.from_index(vec_reg_number_0)
    base = generated_code.current_val_by_reg[gp_reg_base]

    if mask_reg_number:
        # Use the masking version of the operation
        assert isinstance(dest, x86.registers.AVX512RegisterType)
        assert issubclass(
            vmove_instr,
            x86.ops.DM_VmovapdOp
            | x86.ops.DM_VmovapsOp
            | x86.ops.DM_VmovupsOp
            | x86.ops.DM_VmovupdOp,
        )

        mask_reg = AVX512MaskRegisterType.from_index(mask_reg_number)
        mask = generated_code.current_val_by_reg[mask_reg]
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
        generated_code.insert(
            masked_vmove_instr(
                memory=base,
                memory_offset=displacement,
                destination=dest,
                mask_reg=mask,
                z=zero_flag or False,
            )
        )
    else:
        # build vmovpd/ps/sd/ss instruction, load use
        generated_code.insert(
            vmove_instr(memory=base, memory_offset=displacement, destination=dest)
        )


def libxsmm_x86_instruction_mask_move_ld(
    generated_code: GeneratedCode,
    mask_instr: type[x86.ops.KS_KMovBOp]
    | type[x86.ops.KS_KMovWOp]
    | type[x86.ops.KS_KMovWOp]
    | type[x86.ops.KS_KMovQOp],
    mask_tmp_val: SSAValue[GeneralRegisterType],
    mask_reg_number: int,
):
    # char l_new_code[512];
    # int l_max_code_length = 511;
    # int l_code_length = 0;
    # char l_gp_reg_name[4];
    # char l_instr_name[16];
    # char l_prefix = '\0';

    # libxsmm_get_x86_gp_reg_name( gp_reg_number, l_gp_reg_name, 3 );
    # libxsmm_get_x86_instr_name( i_mask_instr, l_instr_name, 15 );

    generated_code.insert(
        mask_instr(
            mask_tmp_val,
            destination=x86.registers.AVX512MaskRegisterType.from_index(
                mask_reg_number
            ),
        )
    )


def libxsmm_x86_instruction_mask_move_st(
    generated_code: GeneratedCode,
    mask_instr: type[x86.ops.DK_KMovBOp]
    | type[x86.ops.DK_KMovWOp]
    | type[x86.ops.DK_KMovWOp]
    | type[x86.ops.DK_KMovQOp],
    gp_reg_number: x86.registers.GeneralRegisterType,
    mask_reg_number: int,
):
    generated_code.insert(
        mask_instr(
            generated_code.current_val_by_reg[
                x86.registers.AVX512MaskRegisterType.from_index(mask_reg_number)
            ],
            destination=gp_reg_number,
        )
    )
