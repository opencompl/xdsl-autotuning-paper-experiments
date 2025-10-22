from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor


def libxsmm_generator_gemm_avx512_kloop_kernel(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    micro_kernel_config: MicroKernelConfig,
    gemm_desc: GEMMDescriptor,
    m_blocking: int,
    n_blocking: int,
    k_blocking: int,
) -> None:
    raise NotImplementedError


# void libxsmm_generator_gemm_avx512_kloop_kernel( libxsmm_generated_code*            io_generated_code,
#                                                  const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
#                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
#                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
#                                                  const unsigned int                 i_m_blocking,
#                                                  const unsigned int                 i_n_blocking,
#                                                  const unsigned int                 i_k_blocking )
# {
#   unsigned int l_k = 0;
#   unsigned int l_k_pack_factor = 1;
#   unsigned int l_m_vector = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
#   unsigned int l_is_Abf8_Bf16_gemm = ((LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai8_Bf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai8_Bbf16_gemm = ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai4_Bf16_gemm = (((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2) > 0) &&
#                                      ((LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                       (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype )) &&
#                                       ((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ))))) ? 1 : 0;
#   unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai2_Bi8_gemm = libxsmm_x86_is_Ai2_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai1_Bi8_gemm = libxsmm_x86_is_Ai1_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Af16_Bf16_gemm =((LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_A_PREC( i_xgemm_desc->datatype )) &&
#                                      (LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_B_PREC( i_xgemm_desc->datatype ))) ? 1 : 0;
#   unsigned int l_is_Ai8_Bbf16_gemm_bf16fma = ((l_is_Ai8_Bbf16_gemm > 0) && (i_micro_kernel_config->vmul_instruction == LIBXSMM_X86_INSTR_VDPBF16PS)) ? 1 : 0;

#   unsigned int l_is_i8_uu_ss_gemm = (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
#                                     ( ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) ||
#                                       ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) >  0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) >  0) ) );
#   unsigned int l_is_not_cpx_bf16 = ((io_generated_code->arch != LIBXSMM_X86_AVX512_CPX) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0));

#   if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
#     l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) );
#   }

#   if (  ( l_m_vector == 1 ) &&
#         ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) == 0) &&
#         (l_is_Abf8_Bf16_gemm == 0) && (l_is_Ai8_Bf16_gemm == 0) && (l_is_Ai8_Bbf16_gemm == 0) && (l_is_Ai4_Bf16_gemm == 0) &&
#         (l_is_Ai4_Bi8_gemm == 0) && (l_is_Af16_Bf16_gemm == 0) && (l_is_i8_uu_ss_gemm == 0) && (l_is_Ai2_Bi8_gemm == 0) && (l_is_Ai1_Bi8_gemm == 0) &&
#         (LIBXSMM_DATATYPE_BF8 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) &&
#         (l_is_not_cpx_bf16 == 0) ) {
#     libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
#                                                        i_xgemm_desc, i_n_blocking, i_k_blocking );
#   } else {
#     void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
#                                     const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int);

#     if (l_is_Ai8_Bbf16_gemm > 0 || l_is_Ai4_Bf16_gemm > 0 || ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0) || l_is_Ai4_Bi8_gemm > 0 || l_is_Ai2_Bi8_gemm > 0 || l_is_Ai1_Bi8_gemm > 0) {
#       if (l_is_Ai8_Bbf16_gemm_bf16fma > 0 || l_is_Ai4_Bf16_gemm > 0) {
#         l_k_pack_factor = 2;
#       } else if (l_is_Ai4_Bi8_gemm > 0) {
#         l_k_pack_factor *= 2;
#       } else if (l_is_Ai2_Bi8_gemm > 0 || l_is_Ai1_Bi8_gemm > 0) {
#         l_k_pack_factor = 4;
#       } else {
#         l_k_pack_factor = 1;
#       }
#       l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
#     } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
#       if ( (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
#         l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_bf16_emu_nofsdbcst;
#       } else if ( l_is_i8_uu_ss_gemm != 0 ) {
#         l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_i8_ss_uu_emu_nofsdbcst;
#       } else {
#         l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_m8_nofsdbcst;
#       }
#     } else if ( (io_generated_code->arch != LIBXSMM_X86_AVX512_CPX) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) ) {
#       l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_bf16_emu_nofsdbcst;
#     } else if ( l_is_i8_uu_ss_gemm != 0 ) {
#       l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_i8_ss_uu_emu_nofsdbcst;
#     } else {
#       l_generator_microkernel = libxsmm_generator_gemm_avx512_microkernel_nofsdbcst;
#     }

#     for ( l_k = 0; l_k < i_k_blocking; l_k += l_k_pack_factor) {
#       l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
#                               i_xgemm_desc, i_m_blocking, i_n_blocking);
#     }
#   }
# }
