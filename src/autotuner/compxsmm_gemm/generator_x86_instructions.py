from typing import Literal
from xdsl.dialects import x86
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    GeneralRegisterType,
    X86VectorRegisterType,
)
from xdsl.ir import SSAValue
from autotuner.libxsmm_gemm.generator_common import GPRegMapping
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.compxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import GEMMPrefetchType


def compxsmm_x86_instruction_open_stream_gemm(
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


def compxsmm_x86_instruction_unified_vec_move_st(
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
            compxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                i_vmove_instr,
                base_val,
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
            compxsmm_x86_instruction_vex_evex_mask_mov_st(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                use_masking,
                mask_reg_number,
                is_store,
            )


def compxsmm_x86_instruction_unified_vec_move_ld(
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
            compxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                i_vmove_instr,
                base_val,
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
            compxsmm_x86_instruction_vex_evex_mask_mov_ld(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                use_masking,
                mask_reg_number,
                is_store,
            )


def compxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None,
    vector_name: Literal["x", "y", "z"],
    src0_val: SSAValue[X86VectorRegisterType],
    src1_val: SSAValue[X86VectorRegisterType],
    dst_val: SSAValue[X86VectorRegisterType],
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

    # build vXYZpd/ps/sd/ss instruction pure register use
    if generated_code.arch > Arch.LIBXSMM_X86_SSE42:
        # if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) and (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
        if imm8 is not None:
            raise NotImplementedError
        else:
            generated_code.insert(vec_instr(dst_val, src0_val, src1_val))
    else:
        raise NotImplementedError


def compxsmm_x86_instruction_vec_compute_3reg(
    generated_code: GeneratedCode,
    vec_instr: type[x86.ops.RSS_Vfmadd231pdOp | x86.ops.RSS_Vfmadd231psOp] | None,
    vector_name: Literal["x", "y", "z"],
    src0_val: SSAValue[X86VectorRegisterType],
    src1_val: SSAValue[X86VectorRegisterType],
    dst_val: SSAValue[X86VectorRegisterType],
) -> None:
    compxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(
        generated_code,
        vec_instr,
        vector_name,
        src0_val,
        src1_val,
        dst_val,
        0,
        0,
        0,
        None,
    )


def compxsmm_x86_instruction_vex_evex_mask_mov_st(
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
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    use_masking: bool,
    mask_reg_number: int,
    is_store: Literal[True],
):
    if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        if use_masking:
            compxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
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
            compxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
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
            compxsmm_x86_instruction_vec_mask_move_st(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                mask_reg_number,
                is_store,
            )
        else:
            compxsmm_x86_instruction_vec_move_st(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
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


def compxsmm_x86_instruction_vex_evex_mask_mov_ld(
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
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    use_masking: bool,
    mask_reg_number: int,
    is_store: Literal[False],
):
    if generated_code.arch >= Arch.LIBXSMM_X86_AVX512_VL128_SKX:
        if use_masking:
            compxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
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
            compxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
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
            compxsmm_x86_instruction_vec_mask_move_ld(
                generated_code,
                vmove_instr,
                base_val,
                reg_idx,
                scale,
                displacement,
                vector_name,
                vec_reg_number_0,
                mask_reg_number,
                is_store,
            )
        else:
            compxsmm_x86_instruction_vec_move_ld(
                generated_code,
                generated_code.arch,
                vmove_instr,
                base_val,
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


def compxsmm_x86_instruction_vec_mask_move_st(
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
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    vec_reg_mask_0: int,
    is_store: bool,
):
    raise NotImplementedError


def compxsmm_x86_instruction_vec_mask_move_ld(
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
    vector_name: Literal["x", "y", "z"],
    vec_reg_number_0: int,
    vec_reg_mask_0: int,
    is_store: bool,
):
    raise NotImplementedError


def compxsmm_x86_instruction_vec_move_st(
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
    base_val: SSAValue[x86.registers.GeneralRegisterType],
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
                memory=base_val,
                memory_offset=displacement,
                source=source,
                mask_reg=mask,
            )
        )
    else:
        generated_code.insert(
            vmove_instr(memory=base_val, source=source, memory_offset=displacement)
        )


def compxsmm_x86_instruction_vec_move_ld(
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
                memory=base_val,
                memory_offset=displacement,
                destination=dest,
                mask_reg=mask,
                z=zero_flag or False,
            )
        )
    else:
        # build vmovpd/ps/sd/ss instruction, load use
        generated_code.insert(
            vmove_instr(memory=base_val, memory_offset=displacement, destination=dest)
        )


def compxsmm_x86_instruction_mask_move_ld(
    generated_code: GeneratedCode,
    mask_instr: type[x86.ops.KS_KMovBOp]
    | type[x86.ops.KS_KMovWOp]
    | type[x86.ops.KS_KMovWOp]
    | type[x86.ops.KS_KMovQOp],
    mask_tmp_val: SSAValue[GeneralRegisterType],
    mask_reg: x86.registers.AVX512MaskRegisterType,
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
            destination=mask_reg,
        )
    )


def compxsmm_x86_instruction_mask_move_st(
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
