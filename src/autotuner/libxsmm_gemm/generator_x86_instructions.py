from typing import Literal
from xdsl.dialects import x86
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

    generated_code.insert(x86.ops.DS_MovOp(a, destination=gp_reg_mapping.gp_reg_a))
    generated_code.insert(x86.ops.DS_MovOp(b, destination=gp_reg_mapping.gp_reg_b))
    generated_code.insert(x86.ops.DS_MovOp(c, destination=gp_reg_mapping.gp_reg_c))

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
            #         libxsmm_generator_maskedstore_32bit_sse( io_generated_code, LIBXSMM_X86_GP_REG_RCX, 1, i_vec_reg_number_0, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_mask_reg_number );
            elif issubclass(i_vmove_instr, x86.ops.DM_VmovapdOp | x86.ops.DM_VmovapdOp):
                ...
            #         libxsmm_generator_maskedstore_64bit_sse( io_generated_code, i_vec_reg_number_0, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_mask_reg_number );
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
                #         libxsmm_generator_maskedload_32bit_sse( io_generated_code, LIBXSMM_X86_GP_REG_RCX, 1, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_vec_reg_number_0, i_mask_reg_number );
                ...
            elif issubclass(i_vmove_instr, x86.ops.DM_VmovapdOp | x86.ops.DM_VmovapdOp):
                #         libxsmm_generator_maskedload_64bit_sse( io_generated_code, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_vec_reg_number_0, i_mask_reg_number );
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
    fallthrough_block = curr_block.next_block
    assert fallthrough_block is not None

    assert (cmp_op := curr_block.last_op) is not None
    assert len(cmp_op.results) == 1

    generated_code.insert(
        jmp_instr(cmp_op, curr_args, curr_args, dest_block, fallthrough_block)
    )

    # set insert point to fallthrough block and update current values
    generated_code.builder.insertion_point = InsertPoint.at_start(fallthrough_block)
    curr_vals.clear()
    curr_vals |= {arg.type: arg for arg in fallthrough_block.args}


def libxsmm_x86_instruction_register_jump_back_label(
    generated_code: GeneratedCode, loop_label_tracker: LoopLabelTracker
) -> None:
    generated_code.insert(x86.ops.LabelOp(f"{loop_label_tracker.current_loop_number}"))


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

    # TODO: handle masking
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

    # TODO: handle masking
    if not use_zero_masking or not mask_reg_number:
        _zero_flag = None
    else:
        _zero_flag = True

    match vector_name:
        case "x":
            dest_type = x86.registers.SSERegisterType
        case "y":
            dest_type = x86.registers.AVX2RegisterType
        case "z":
            dest_type = x86.registers.AVX512RegisterType
    dest = dest_type.from_index(vec_reg_number_0)
    base = generated_code.current_val_by_reg[gp_reg_base]

    # build vmovpd/ps/sd/ss instruction, load use
    generated_code.insert(
        vmove_instr(memory=base, memory_offset=displacement, destination=dest)
    )
