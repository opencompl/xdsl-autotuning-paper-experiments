from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def libxsmm_generator_gemm_avx512_kloop_kernel(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    k_blocking: int,
) -> None:
    k = 0
    _k_pack_factor = 1
    m_vector = (
        m_blocking // micro_kernel_config.vector_length
        if (m_blocking % micro_kernel_config.vector_length == 0)
        else (m_blocking // micro_kernel_config.vector_length) + 1
    )

    is_Abf8_Bf16_gemm = (
        Datatype.BF8 == gemm_desc.datatype.a and Datatype.F16 == gemm_desc.datatype.b
    )
    is_Ai8_Bf16_gemm = (
        Datatype.I8 == gemm_desc.datatype.a and Datatype.F16 == gemm_desc.datatype.b
    )
    is_Ai8_Bbf16_gemm = (
        Datatype.I8 == gemm_desc.datatype.a and Datatype.BF16 == gemm_desc.datatype.b
    )
    is_Ai4_Bf16_gemm = (
        GEMMFlag.INTERPRETE_A_AS_INT4_VNNI2 in gemm_desc.flags
        and Datatype.I8 == gemm_desc.datatype.a
        and Datatype.F16 == gemm_desc.datatype.b
        and (
            Datatype.F16 == gemm_desc.datatype.c or Datatype.F32 == gemm_desc.datatype.c
        )
    )
    is_Ai4_Bi8_gemm = gemm_desc.is_Ai4_Bi8_gemm()
    is_Ai2_Bi8_gemm = gemm_desc.is_Ai2_Bi8_gemm()
    is_Ai1_Bi8_gemm = gemm_desc.is_Ai1_Bi8_gemm()
    is_Af16_Bf16_gemm = (
        Datatype.F16 == gemm_desc.datatype.a and Datatype.F16 == gemm_desc.datatype.b
    )
    _is_Ai8_Bbf16_gemm_bf16fma = (
        is_Ai8_Bbf16_gemm and micro_kernel_config.vmul_instruction == "VDPBF16PS"
    )

    is_i8_uu_ss_gemm = gemm_desc.datatype.ab == "I8" and (
        (
            GEMMFlag.A_UNSIGNED not in gemm_desc.flags
            and GEMMFlag.B_UNSIGNED not in gemm_desc.flags
        )
        or (
            GEMMFlag.A_UNSIGNED in gemm_desc.flags
            and GEMMFlag.B_UNSIGNED in gemm_desc.flags
        )
    )

    is_not_cpx_bf16 = (
        generated_code.arch != "AVX512_CPX"
        and gemm_desc.datatype.ab == "BF16"
        and GEMMFlag.VNNI_A in gemm_desc.flags
    )

    if GEMMFlag.VNNI_A in gemm_desc.flags:
        raise NotImplementedError

    if (
        m_vector == 1
        and GEMMFlag.DECOMPRESS_A_VIA_BITMASK not in gemm_desc.flags
        and not (not is_Abf8_Bf16_gemm)
        and (not is_Ai8_Bf16_gemm)
        and (not is_Ai8_Bbf16_gemm)
        and (not is_Ai4_Bf16_gemm)
        and (not is_Ai4_Bi8_gemm)
        and (not is_Af16_Bf16_gemm)
        and (not is_i8_uu_ss_gemm)
        and (not is_Ai2_Bi8_gemm)
        and (not is_Ai1_Bi8_gemm)
        and (Datatype.BF8 != gemm_desc.datatype.ab)
        and (not is_not_cpx_bf16)
    ):
        raise NotImplementedError
    else:
        # void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
        #                                 const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int);
        if (
            is_Ai8_Bbf16_gemm
            or is_Ai4_Bf16_gemm
            or GEMMFlag.DECOMPRESS_A_VIA_BITMASK in gemm_desc.flags
            or is_Ai4_Bi8_gemm
            or is_Ai2_Bi8_gemm
            or is_Ai1_Bi8_gemm
        ):
            raise NotImplementedError
        elif (
            Arch.LIBXSMM_X86_AVX512_VL256_SKX
            <= generated_code.arch
            < Arch.LIBXSMM_X86_AVX512_SKX
        ):
            raise NotImplementedError
        elif (
            generated_code.arch != Arch.LIBXSMM_X86_AVX512_CPX
            and Datatype.BF16 == gemm_desc.datatype.ab
        ):
            raise NotImplementedError
        elif is_i8_uu_ss_gemm:
            raise NotImplementedError
        else:
            generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst

        for k in range(k_blocking):
            generator_microkernel(
                generated_code,
                gp_reg_mapping,
                micro_kernel_config,
                gemm_desc,
                m_blocking,
                n_blocking,
            )


def libxsmm_generator_gemm_avx512_microkernel_nofsdbcst(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
) -> None:
    return


# LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_nofsdbcst( libxsmm_generated_code*            io_generated_code,
#                                                                              const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
#                                                                              const libxsmm_micro_kernel_config* i_micro_kernel_config,
#                                                                              const libxsmm_gemm_descriptor*     i_xgemm_desc,
#                                                                              const unsigned int                 i_m_blocking,
#                                                                              const unsigned int                 i_n_blocking )
# {
#   /* deriving register blocking from kernel config */
#   unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
#   /* register blocking counter in n */
#   unsigned int l_n = 0;
#   /* register blocking counter in m */
#   unsigned int l_m = 0;
#   unsigned int l_k = 0;
#   /* start register of accumulator */
#   unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
#   unsigned int l_vreg_ab_offset = 0;
#   /* temp variable for b-offset to handle no-trans/trans B */
#   int l_b_offset = 0;
#   /* k packing factor for VNNI */
#   unsigned int l_k_pack_factor = 1;
#   unsigned int l_is_Abf8_Bf16_gemm = ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
#                                      ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                       (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
#                                       ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
#   unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai2_Bi8_gemm = libxsmm_x86_is_Ai2_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai1_Bi8_gemm = libxsmm_x86_is_Ai1_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_k_iters = (l_is_Ai4_Bf16_gemm > 0 || l_is_Ai4_Bi8_gemm > 0) ? 2 : 1;
#   unsigned int l_is_Af16_Bf16_gemm =((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = ((l_is_Ai8_Bbf16_gemm > 0) && (i_micro_kernel_config->vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS)) ? 1 : 0;
#   unsigned int l_use_f16_replacement_fma = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR && LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_use_f32_compute_with_f16_inp = (((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Af16_Bf16_gemm > 0) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_COMP_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_mask_load_i1 = 1;

#   char vname_cvt = i_micro_kernel_config->vector_name;

#   if (l_is_Ai8_Bf16_gemm > 0) {
#     if (l_is_Ai4_Bf16_gemm > 0) {
#       l_vreg_ab_offset = 3;
#       if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
#         l_vreg_ab_offset = 2 + l_m_blocking;
#       }
#     } else {
#       l_vreg_ab_offset = 1;
#       if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
#         l_vreg_ab_offset = l_m_blocking;
#       }
#     }
#     if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
#       l_vreg_ab_offset += l_m_blocking;
#     }
#   }

#   if (l_is_Ai8_Bbf16_gemm > 0) {
#     l_vreg_ab_offset = 2;
#     if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) > 0) {
#       l_vreg_ab_offset = 1 + l_m_blocking;
#     }
#   }

#   if (l_is_Ai4_Bi8_gemm > 0) {
#     l_vreg_ab_offset = 3;
#     if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0 || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_MxK_ZPT) > 0) {
#       l_vreg_ab_offset += l_m_blocking;
#     }
#   }

#   if (l_is_Ai2_Bi8_gemm > 0) {
#     l_vreg_ab_offset = 3;
#   }

#   if (l_is_Ai1_Bi8_gemm > 0) {
#     l_vreg_ab_offset = 2;
#   }

# #if !defined(NDEBUG)
#   if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
#     return;
#   }

#   if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX && io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
#       if (l_is_Ai8_Bbf16_gemm > 0) {
#         if ( ((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
#           return;
#         }
#       } else if (l_is_Ai4_Bf16_gemm > 0)  {
#         int l_m_scf_vregs = ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF) == 0) ?  3 + l_m_blocking : 2 + 2*l_m_blocking;
#         if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT) > 0) {
#           l_m_scf_vregs += l_m_blocking;
#         }
#         if ( ((l_m_blocking*i_n_blocking) + l_m_blocking + 1 + l_m_scf_vregs) > 32 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
#           return;
#         }
#       } else {
#         if ( ((l_m_blocking*i_n_blocking) + i_n_blocking + 1) > 32 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
#           return;
#         }
#       }
#       if ( (l_m_blocking < 1) || (l_m_blocking > 8) ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
#         return;
#       }
#   } else {
#       if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
#         return;
#       }
#       if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
#         return;
#       }
#   }
# #endif

#   /* for VNNI we are stepping through to pack ks */
#   if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
#     l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
#   }

#   /* load column vectors of A upfront */
#   for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
#     const char *const l_env_a_k_pf_dist = getenv("LIBXSMM_GEMM_K_A_PF_DIST");
#     unsigned l_a_k_pf_dist = (l_env_a_k_pf_dist == 0) ? 0 : atoi(l_env_a_k_pf_dist);
#     char l_a_vname = (l_is_Ai8_Bf16_gemm == 0 && l_is_Abf8_Bf16_gemm == 0) ? i_micro_kernel_config->vector_name : ((l_use_f32_compute_with_f16_inp) ? (i_micro_kernel_config->vector_name == 'z' ? 'x' : 'x') : ((l_use_f16_replacement_fma > 0) ? (i_micro_kernel_config->vector_name == 'z' ? 'x' : 'x') : (i_micro_kernel_config->vector_name == 'z' ? 'y' : 'x')));
#     unsigned int l_a_vmove_instruction = ((l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && (l_m != (l_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->a_vmove_instruction;

#     if (l_is_Af16_Bf16_gemm > 0) {
#       if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
#         l_a_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
#       }
#     }

#     if (l_use_f16_replacement_fma > 0) {
#       vname_cvt = (l_a_vname == 'y') ? 'z' : ((l_a_vname == 'x') ? ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z') : i_micro_kernel_config->vector_name);
#     }

#     if (l_is_Ai8_Bbf16_gemm > 0) {
#       libxsmm_generator_gemm_avx512_microkernel_loadNinterleave_A_pair_k_i8_to_bf16( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
#           1+l_m+l_vreg_ab_offset, l_vreg_ab_offset, 0, l_m_blocking, l_m );
#     } else {
#       if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
#         unsigned int l_current_mask_reg = (3+l_m)%8;
#         char decompress_vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ? (i_micro_kernel_config->vector_name == 'z' ? 'y' : 'x') : i_micro_kernel_config->vector_name;
#         /* Load bit mask for current expand operation */
#         libxsmm_x86_instruction_mask_move_mem( io_generated_code,
#             (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_X86_INSTR_KMOVD_LD : LIBXSMM_X86_INSTR_KMOVW_LD,
#             i_gp_reg_mapping->gp_reg_bitmap_a,
#             LIBXSMM_X86_GP_REG_UNDEF, 0,
#             (i_micro_kernel_config->vector_length/8) * l_m * l_k_pack_factor,
#             l_current_mask_reg );

#         libxsmm_x86_instruction_prefetch(io_generated_code,
#             LIBXSMM_X86_INSTR_PREFETCHT0,
#             i_gp_reg_mapping->gp_reg_bitmap_a,
#             LIBXSMM_X86_GP_REG_UNDEF, 0,
#             (i_micro_kernel_config->vector_length/8) * l_m * l_k_pack_factor +  16 * 64);

#         /* Expand operation */
#         libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
#                                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) || LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_X86_INSTR_VPEXPANDW : LIBXSMM_X86_INSTR_VPEXPANDD,
#                                                      decompress_vname,
#                                                      i_gp_reg_mapping->gp_reg_a,
#                                                      i_gp_reg_mapping->gp_reg_decompressed_elts,
#                                                      i_micro_kernel_config->datatype_size_in,
#                                                      0,
#                                                      0,
#                                                      LIBXSMM_X86_VEC_REG_UNDEF,
#                                                      1+l_m+l_vreg_ab_offset,
#                                                      l_current_mask_reg,
#                                                      1,
#                                                      0);
#         if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) {
#           /* Convert bf16 to f32 */
#           libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, 1+l_m+l_vreg_ab_offset, 1+l_m+l_vreg_ab_offset);
#         }

#         libxsmm_x86_instruction_prefetch(io_generated_code,
#             LIBXSMM_X86_INSTR_PREFETCHT0,
#             i_gp_reg_mapping->gp_reg_a,
#             i_gp_reg_mapping->gp_reg_decompressed_elts, i_micro_kernel_config->datatype_size_in,
#             16 * 64);

#         /* Move zmm to reg */
#         libxsmm_x86_instruction_mask_move( io_generated_code,
#           (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) ? LIBXSMM_X86_INSTR_KMOVD_GPR_ST : LIBXSMM_X86_INSTR_KMOVW_GPR_ST,
#           i_gp_reg_mapping->gp_reg_popcnt,
#           l_current_mask_reg );
#         /* Popcount */
#         libxsmm_x86_instruction_alu_reg( io_generated_code,
#             LIBXSMM_X86_INSTR_POPCNT,
#             i_gp_reg_mapping->gp_reg_popcnt,
#             i_gp_reg_mapping->gp_reg_popcnt);
#         /* Adjust count of decompressed elements */
#         libxsmm_x86_instruction_alu_reg( io_generated_code,
#             LIBXSMM_X86_INSTR_ADDQ,
#             i_gp_reg_mapping->gp_reg_popcnt,
#             i_gp_reg_mapping->gp_reg_decompressed_elts);
#       } else {
#         if (l_is_Ai1_Bi8_gemm > 0) {
#           libxsmm_x86_instruction_mask_move_mem( io_generated_code, LIBXSMM_X86_INSTR_KMOVQ_LD, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_UNDEF, 0, ((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor)/8, l_mask_load_i1+l_m%7);
#         } else {
#           libxsmm_x86_instruction_vec_move( io_generated_code,
#               i_micro_kernel_config->instruction_set,
#               l_a_vmove_instruction,
#               i_gp_reg_mapping->gp_reg_a,
#               LIBXSMM_X86_GP_REG_UNDEF, 0,
#               (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
#               l_a_vname,
#               (l_is_Ai2_Bi8_gemm > 0) ? l_m+l_vreg_ab_offset : 1+l_m+l_vreg_ab_offset, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
#         }

#         if ((((i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor) % 64 == 0) && (l_a_k_pf_dist > 0)) {
#           libxsmm_x86_instruction_prefetch( io_generated_code,
#               LIBXSMM_X86_INSTR_PREFETCHT0,
#               i_gp_reg_mapping->gp_reg_a,
#               LIBXSMM_X86_GP_REG_UNDEF, 0,
#               (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor + l_a_k_pf_dist * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in);
#         }
#       }
#     }

#     if (l_is_Ai4_Bi8_gemm > 0) {
#       /* Process vreg A in case of i4i8 */
#       libxsmm_generator_gemm_avx512_microkernel_process_vreg_A_for_i4i8 (io_generated_code, i_micro_kernel_config, i_xgemm_desc,
#           l_m, l_m_blocking, 1+l_m+l_vreg_ab_offset);
#     } else if (l_is_Ai2_Bi8_gemm > 0) {
#       /* Process vreg A in case of i2i8 */
#       libxsmm_generator_gemm_avx512_microkernel_process_vreg_A_for_i2i8 (io_generated_code, i_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m+l_vreg_ab_offset, 1+l_m+l_vreg_ab_offset);
#       break;
#     } else if (l_is_Ai1_Bi8_gemm > 0) {
#       libxsmm_generator_gemm_avx512_microkernel_process_vreg_A_for_i1i8 (io_generated_code, i_micro_kernel_config, l_mask_load_i1+l_m%7,  1+l_m+l_vreg_ab_offset, 0, 1);
#     } else {
#       /* Process vreg A in case of f16/i8 */
#       libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( io_generated_code, i_micro_kernel_config, i_xgemm_desc,
#           vname_cvt, l_is_Ai8_Bf16_gemm, l_is_Abf8_Bf16_gemm, l_is_Af16_Bf16_gemm, l_use_f16_replacement_fma, l_use_f32_compute_with_f16_inp, l_m, l_m_blocking, 1+l_m+l_vreg_ab_offset);
#     }

#     /* prefetch a different A matrix provided by the prefetch pointers */
#     if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) /*|| (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2CL1)*/ ) {
#       libxsmm_x86_instruction_prefetch( io_generated_code,
#           LIBXSMM_X86_INSTR_PREFETCHT1,
#           i_gp_reg_mapping->gp_reg_a_prefetch,
#           LIBXSMM_X86_GP_REG_UNDEF, 0,
#           (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor);
#     }
#   }

#   for ( l_k = 0; l_k < l_k_iters; l_k++ ) {
#     for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
#       char l_b_vname = i_micro_kernel_config->vector_name;
#       unsigned int l_b_vmove_instruction = i_micro_kernel_config->b_vmove_instruction;

#       if (l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) {
#         if ( l_use_f32_compute_with_f16_inp > 0 || l_use_f16_replacement_fma > 0 ) {
#           l_b_vname = (i_micro_kernel_config->vector_name == 'z') ? 'y' : 'x';
#         }
#       }

#       if ((l_is_Ai8_Bbf16_gemm > 0) && (l_is_Ai8_Bbf16_gemm_bf16fma == 0)) {
#         l_b_vmove_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
#       }

#       /* handle trans B */
#       if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
#         unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2) : l_k_pack_factor;
#         l_b_offset = l_n * i_micro_kernel_config->datatype_size_in2 * l_k_pack_advance + (i_micro_kernel_config->datatype_size_in2 * l_k * i_xgemm_desc->ldb) ;
#       } else {
#         unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2) : l_k_pack_factor;
#         l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in2 + l_k * l_k_pack_advance * i_micro_kernel_config->datatype_size_in2;
#       }

#       l_b_vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0)) ? (l_b_vname == 'z' ? 'y' : 'x') : l_b_vname;
#       libxsmm_x86_instruction_vec_move( io_generated_code,
#           i_micro_kernel_config->instruction_set,
#           l_b_vmove_instruction,
#           i_gp_reg_mapping->gp_reg_b,
#           LIBXSMM_X86_GP_REG_UNDEF, 0,
#           l_b_offset,
#           l_b_vname,
#           l_vreg_ab_offset, 0, 1, 0 );

#       if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0 &&  LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype)) {
#         /* Convert bf16 to f32 */
#         libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, l_vreg_ab_offset, l_vreg_ab_offset);
#       }

#       if ((l_is_Af16_Bf16_gemm > 0 || l_is_Ai8_Bf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0) && (l_use_f16_replacement_fma > 0 || l_use_f32_compute_with_f16_inp > 0)) {
#         libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vreg_ab_offset, l_vreg_ab_offset );
#       }

#       if ( l_is_Ai8_Bbf16_gemm > 0 && l_is_Ai8_Bbf16_gemm_bf16fma == 0 ) {
#         libxsmm_x86_instruction_vec_compute_2reg_imm8(io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_micro_kernel_config->vector_name, l_vreg_ab_offset, l_vreg_ab_offset, 16);
#       }

#       if (l_n == i_n_blocking - 1) {
#         /* handle trans B */
#         if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
#           l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in2 * l_k_iters;
#         } else {
#           unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2) : l_k_pack_factor;
#           l_b_offset = i_micro_kernel_config->datatype_size_in2 * l_k_pack_advance * l_k_iters;
#         }

#         if (l_k == (l_k_iters-1)) {
#           libxsmm_x86_instruction_alu_imm( io_generated_code,
#               i_micro_kernel_config->alu_add_instruction,
#               i_gp_reg_mapping->gp_reg_b,
#               l_b_offset );
#         }
#       }

#       for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
#         /* post increment early */
#         if ( (l_m == 0) && (l_n == i_n_blocking-1) && (l_k == 0)) {
#           unsigned int l_k_pack_advance = (l_is_Ai8_Bbf16_gemm > 0) ? ((l_is_Ai8_Bbf16_gemm_bf16fma == 0) ? 1 : 2): l_k_pack_factor;
#           unsigned int l_a_adjust = (l_is_Ai2_Bi8_gemm > 0) ? 4 : ((l_is_Ai1_Bi8_gemm > 0) ? 8 : 1);

#           if (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) {
#             libxsmm_x86_instruction_alu_imm( io_generated_code,
#                 i_micro_kernel_config->alu_add_instruction,
#                 i_gp_reg_mapping->gp_reg_bitmap_a,
#                 (long long)(i_xgemm_desc->lda/8) * l_k_pack_advance);
#           } else {
#             libxsmm_x86_instruction_alu_imm( io_generated_code,
#                 i_micro_kernel_config->alu_add_instruction,
#                 i_gp_reg_mapping->gp_reg_a,
#                 (long long)i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in * l_k_pack_advance/l_a_adjust);
#           }

#           /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
#           if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) /*|| (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2CL1)*/ ) {
#             libxsmm_x86_instruction_alu_imm( io_generated_code,
#                 i_micro_kernel_config->alu_add_instruction,
#                 i_gp_reg_mapping->gp_reg_a_prefetch,
#                 (long long)i_xgemm_desc->lda * i_micro_kernel_config->datatype_size_in * l_k_pack_factor/l_a_adjust);
#           }
#         }
#         /* issue fma */
#         if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
#           if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0) {
#             libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
#                 i_micro_kernel_config->vmul_instruction,
#                 i_micro_kernel_config->vector_name,
#                 l_vreg_ab_offset,
#                 1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
#                 l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
#           } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
#             libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
#                 i_micro_kernel_config->vmul_instruction,
#                 i_micro_kernel_config->vector_name,
#                 1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
#                 l_vreg_ab_offset,
#                 l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
#           } else {
#             /* should not happen */
#           }
#         } else if (l_use_f16_replacement_fma > 0) {
#           libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, vname_cvt, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
#           libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
#               i_micro_kernel_config->vmul_instruction,
#               vname_cvt,
#               1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
#               l_vreg_ab_offset,
#               l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
#           libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, vname_cvt, l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0,
#                                                                   (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
#         } else {
#           libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
#               i_micro_kernel_config->vmul_instruction,
#               i_micro_kernel_config->vector_name,
#               1+l_m+l_vreg_ab_offset + l_k * l_m_blocking,
#               l_vreg_ab_offset,
#               l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
#         }
#       }
#     }
#   }
# }
