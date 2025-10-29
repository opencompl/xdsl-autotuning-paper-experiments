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
): ...


# LIBXSMM_API_INTERN
# void libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( libxsmm_generated_code* io_generated_code,
#                                                              const unsigned int      i_vec_instr,
#                                                              const char              i_vector_name,
#                                                              const unsigned int      i_reg_number_src0,
#                                                              const unsigned int      i_reg_number_src1,
#                                                              const unsigned int      i_reg_number_dst,
#                                                              const unsigned int      i_mask_reg_number,
#                                                              const unsigned int      i_mask_cntl,
#                                                              const unsigned char     i_sae_cntl,
#                                                              const unsigned int      i_imm8 ) {
#   if ( (libxsmm_x86_instruction_vec_is_hybrid( i_vec_instr )  == 0) and
#        (libxsmm_x86_instruction_vec_is_regonly( i_vec_instr ) == 0)    ) {
#     fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unexpected instruction number: 0x%08x\n", i_vec_instr);
#     LIBXSMM_EXIT_ERROR(io_generated_code);
#     return;
#   }

#   /* check that we are not masking 'y' */
#   if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) and (i_mask_reg_number != 0) ) {
#     fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: Masking is only available for AVX512!\n");
#     LIBXSMM_EXIT_ERROR(io_generated_code);
#     return;
#   }

#   /* select the code generator REX/VEX/EVEX */
#   if ( io_generated_code->code_type > 1 ) {
#     unsigned int l_encoder; /* 2=EVEX, 1=VEX, 0=REX */
#     unsigned int l_encoder_arch = 2;
#     unsigned int l_encoder_instr = ((i_vec_instr >> 30) & 0x03);
#     unsigned int l_reg_number_src0 = 0;
#     unsigned int l_reg_number_src1 = 0;
#     unsigned int l_reg_number_dst = 0;

#     /* determine encoder */
#     if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
#       l_encoder_arch = 0;
#     }
#     else if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
#       l_encoder_arch = 1;
#     }
#     if ( (l_encoder_arch == 2) and ((l_encoder_instr == 3) || (l_encoder_instr == 0)) ) {
#       l_encoder = 2;
#     } else if ( (l_encoder_arch >= 1) and ((l_encoder_instr == 1) || (l_encoder_instr == 0)) ) {
#       l_encoder = 1;
#     } else {
#       l_encoder = 0;
#     }

#     /* check that we have an UNDEF for 2 src operands */
#     if ( ((i_vec_instr >> 28) & 3) == 2 ) {
#       if ( ((l_encoder != 0) and (i_reg_number_src1 != LIBXSMM_X86_VEC_REG_UNDEF)) || ((l_encoder == 0) and ((i_reg_number_src1 != i_reg_number_dst) and (i_reg_number_src1 != LIBXSMM_X86_VEC_REG_UNDEF))) ) {
#         fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a 2 src operand instruction (0x%08x), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
#         LIBXSMM_EXIT_ERROR(io_generated_code);
#         return;
#       }
#       l_reg_number_src1 = 0;
#     } else if ( ((i_vec_instr >> 28) & 3) == 1 ) {
#       if ( i_reg_number_src0 != LIBXSMM_X86_VEC_REG_UNDEF ) {
#         fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a 1 src operand instruction (0x%08x), i_reg_number_src0 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
#         LIBXSMM_EXIT_ERROR(io_generated_code);
#         return;
#       }
#       l_reg_number_src0 = 0;
#     } else {
#       l_reg_number_src1 = i_reg_number_src1;
#     }

#     /* check if we need to flip operands */
#     if ( ((i_vec_instr >> 24) & 0x08 ) == 0x08 ) {
#       l_reg_number_dst = i_reg_number_src0;
#       l_reg_number_src0 = i_reg_number_dst;
#     } else {
#       l_reg_number_dst = i_reg_number_dst;
#       l_reg_number_src0 = i_reg_number_src0;
#     }

#     /* check if we have op-code extension in modrm/reg */
#     if ( ((i_vec_instr >> 24) & 0x04 ) == 0x04 ) {
#       if ( ((i_vec_instr >> 28) & 0x3) == 0x2 ) {
#         l_reg_number_src1 = i_reg_number_dst;
#         l_reg_number_dst = ((i_vec_instr >> 20) & 0x07);
#       } else if ( ((i_vec_instr >> 28) & 0x3) == 0x1 )  {
#         l_reg_number_src0 = i_reg_number_dst;
#         l_reg_number_dst = ((i_vec_instr >> 20) & 0x07);
#       } else {
#         fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a op-code modrm/reg extended instruction (0x%08x), i_reg_number_src1 or i_reg_number_src0 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
#         LIBXSMM_EXIT_ERROR(io_generated_code);
#         return;
#       }
#     }

#     /* encode main instruction */
#     if ( l_encoder == 2 ) {
#       libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

#       /* set simd name */
#       switch(i_vector_name) {
#         case 'x':
#           l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
#           break;
#         case 'y':
#           l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
#           break;
#         case 'z':
#           l_simd_name = LIBXSMM_X86_SIMD_NAME_ZMM;
#           break;
#         default:
#           fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unsupported vlen: %c\n", i_vector_name);
#           break;
#       }

#       libxsmm_x86_instruction_evex_compute_3reg( io_generated_code, i_vec_instr, l_simd_name,
#             l_reg_number_src0, l_reg_number_src1, l_reg_number_dst, i_mask_reg_number, i_mask_cntl, i_sae_cntl );
#     } else if ( l_encoder == 1 ) {
#       libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

#       /* set simd name */
#       switch(i_vector_name) {
#         case 'x':
#           l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
#           break;
#         case 'y':
#           l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
#           break;
#         default:
#           fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unsupported vlen: %c\n", i_vector_name);
#           break;
#       }

#       libxsmm_x86_instruction_vex_compute_3reg( io_generated_code, i_vec_instr, l_simd_name,
#             l_reg_number_src0, l_reg_number_src1, l_reg_number_dst );
#     } else {
#       libxsmm_x86_instruction_rex_compute_2reg( io_generated_code, i_vec_instr,
#             l_reg_number_src0, l_reg_number_dst );
#     }

#     /* add imm if needed */
#     if ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) {
#       if ( i_imm8 != LIBXSMM_X86_IMM_UNDEF ) {
#         unsigned char* code = (unsigned char *) io_generated_code->generated_code;
#         code[io_generated_code->code_size++] = (unsigned char)i_imm8;
#       } else {
#         fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: imm8 required by instr, but LIBXSMM_X86_IMM_UNDEF was provided!\n");
#         LIBXSMM_EXIT_ERROR(io_generated_code);
#         return;
#       }
#     }
#   } else {
#     char l_new_code[512];
#     int l_max_code_length = 511;
#     int l_code_length = 0;
#     char l_instr_name[16];
#     unsigned int l_imm8 = (unsigned int)i_imm8;
#     libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

#     /* build vXYZpd/ps/sd/ss instruction pure register use*/
#     if ( io_generated_code->arch > LIBXSMM_X86_SSE42 ) {
#       if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) and (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
#         if ( io_generated_code->code_type == 0 ) {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
#         } else {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
#         }
#       } else {
#         if ( io_generated_code->code_type == 0 ) {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
#         } else {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
#         }
#       }
#     } else {
#       if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) and (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
#         if ( io_generated_code->code_type == 0 ) {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
#         } else {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
#         }
#       } else {
#         if ( io_generated_code->code_type == 0 ) {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
#         } else {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
#         }
#       }
#     }
#     libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#   }
# }


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
