from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86.ops import print_assembly
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Block, Region

from autotuner.libxsmm_gemm.generator_common import libxsmm_mmfunction_signature
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor


def libxsmm_generator_gemm_directasm(
    file_out: Path, routine_name: str, desc: GEMMDescriptor, arch: Arch
):
    module_op = ModuleOp(Region(Block()))

    # Instead of function signature, generate `FuncOp`.
    func_op = libxsmm_mmfunction_signature(module_op, routine_name)

    # Generate the actual kernel code for current description depending on the
    # architecture.
    libxsmm_generator_gemm_kernel(func_op, arch, desc)

    # Append code to source file
    with open(file_out, "rw") as f:
        print_assembly(module_op, f)


def libxsmm_generator_gemm_kernel(func_op: FuncOp, arch: Arch, desc: GEMMDescriptor):
    raise NotImplementedError


#   /* apply the alignment override */
#   libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
#   unsigned int l_vector_length = 1;
#   int l_aarch64_bfdot = libxsmm_cpuid_arm_use_bfdot();
#   int l_aarch64_i8dot = libxsmm_cpuid_arm_use_i8dot();
#   unsigned int l_saved_arch = io_generated_code->arch;
#   unsigned int l_is_Ai4_Bi8_gemm = libxsmm_x86_is_Ai4_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai2_Bi8_gemm = libxsmm_x86_is_Ai2_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Ai1_Bi8_gemm = libxsmm_x86_is_Ai1_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Amxfp4_Bbf16_gemm = libxsmm_x86_is_Amxfp4_Bbf16_gemm(i_xgemm_desc);
#   unsigned int l_is_Amxfp4_Bfp32_gemm = libxsmm_x86_is_Amxfp4_Bfp32_gemm(i_xgemm_desc);
#   unsigned int l_is_Amxfp4_Bi8_gemm = libxsmm_x86_is_Amxfp4_Bi8_gemm(i_xgemm_desc);
#   unsigned int l_is_Abf8_Bbf16_gemm = libxsmm_x86_is_Abf8_Bbf16_gemm(i_xgemm_desc);
#   unsigned int l_is_Abf8_Bf16_gemm = libxsmm_x86_is_Abf8_Bf16_gemm(i_xgemm_desc);
#   unsigned int l_is_Ahf8_Bbf16_gemm = libxsmm_x86_is_Ahf8_Bbf16_gemm(i_xgemm_desc);
#   unsigned int l_is_var_ld = ( (l_xgemm_desc_mod.lda == 0) && (l_xgemm_desc_mod.ldb == 0) && (l_xgemm_desc_mod.ldc == 0) );

#   /* Support this precision only in avx2 for now  */
#   if ( l_is_Amxfp4_Bfp32_gemm > 0 ) {
#     if (io_generated_code->arch >= LIBXSMM_X86_AVX2 && io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) {
#       io_generated_code->arch = LIBXSMM_X86_AVX2;
#     }
#   }

#   if ( l_is_Amxfp4_Bi8_gemm > 0 ) {
#     if (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   }

#   if ( l_is_Amxfp4_Bbf16_gemm > 0 ) {
#     if (io_generated_code->arch >= LIBXSMM_X86_AVX2 && io_generated_code->arch < LIBXSMM_X86_AVX512_SPR) {
#       if ((io_generated_code->arch >= LIBXSMM_X86_AVX2_SRF) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX)) {
#         io_generated_code->arch = LIBXSMM_X86_AVX2_SRF;
#       } else {
#         io_generated_code->arch = LIBXSMM_X86_AVX2;
#       }
#     }
#   }

#   /* Check if it is a supported spmm with bitmap */
#   if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK) > 0) {
#     if ((io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT )) {
#       if ( !(
#            ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) &&
#              ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32))     )  ||
#             ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) &&
#               (((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8 || LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#               ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8 || LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)    ) )) ||
#             ((io_generated_code->arch >= LIBXSMM_X86_AVX512_GNR) &&
#               (((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#               ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)    ) )) ||
#             ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) &&
#               (((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#               ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#               (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)    ) ))   )) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#       if ( !((l_xgemm_desc_mod.m == 16 || l_xgemm_desc_mod.m == 32) && (l_xgemm_desc_mod.n <= 32) && (l_xgemm_desc_mod.lda == l_xgemm_desc_mod.m) && (l_xgemm_desc_mod.k % 32 == 0)) ) {
#         if ( !(l_xgemm_desc_mod.m == 16 || l_xgemm_desc_mod.m == 32)) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
#           return;
#         }
#         if ( l_xgemm_desc_mod.n > 32) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
#           return;
#         }
#         if ( l_xgemm_desc_mod.k % 32 != 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_K_BLOCK );
#           return;
#         }
#         if ( l_xgemm_desc_mod.lda != l_xgemm_desc_mod.m ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
#           return;
#         }
#       }
#     } else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
#       return;
#     }
#   }

#   /* check for generally supported precisions */
#   if ( !(
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)     )  ||
# #if 0
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_U8)   &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)     )  ||
# #endif
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)     )  ||
# #if 0
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_U8)   &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
# #endif
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8)   && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)    ) ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)    ) ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8) && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16)    ) ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)     )  ||
#          ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8)     )
#         ) ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#     return;
#   }

#   /* Currently, RVV supports F32 without transpose only */
#   if ((io_generated_code->arch >= LIBXSMM_RV64_MVL128)) {
#     if (!(((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32)
#         ) || ((LIBXSMM_GEMM_GETENUM_A_PREC(    l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)  && (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)  &&
#           (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)  && (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F64)
#         ) ) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }

#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#     }
#     else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#     }
#   }

#   if ((io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT )) {
#     if (LIBXSMM_DATATYPE_UNSUPPORTED != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) {
#       /* Supported JITed combos: */
#       /* i2i8 && m=32 && arch == SRF (B both signed and unsigned) */
#       /* i1i8 && m=32 && arch == SRF */
#       /* i2i8 && m=32 && arch >= SPR (B both signed and unsigned) */
#       /* i2i8 && m=64 && arch >= AVX_512 && arch < SPR && B unsigned */
#       /* i1i8 && m=64 && arch >= SPR && B unsigned */

#       /* Check for supported i2i8 and i1i8 combinations */
#       if (l_is_Ai4_Bi8_gemm > 0 || l_is_Ai2_Bi8_gemm > 0 || l_is_Ai1_Bi8_gemm > 0) {
#         int l_is_supported = 0;
#         int l_b_unsigned = (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0;

#         if (l_is_Ai4_Bi8_gemm > 0 && io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) {
#           l_is_supported = 1;
#         }

#         /* i2i8 && m=32 && arch == SRF (B both signed and unsigned) */
#         if (l_is_Ai2_Bi8_gemm > 0 && l_xgemm_desc_mod.m == 32 && io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) {
#           l_is_supported = 1;
#         }
#         /* i1i8 && m=32 && arch == SRF */
#         else if (l_is_Ai1_Bi8_gemm > 0 && l_xgemm_desc_mod.m == 32 && io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) {
#           l_is_supported = 1;
#         }
#         /* i2i8 && m=32 && arch >= SPR (B both signed and unsigned) */
#         else if (l_is_Ai2_Bi8_gemm > 0 && l_xgemm_desc_mod.m == 32 && io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
#           l_is_supported = 1;
#         }
#         /* i2i8 && m=64 && arch >= AVX_512 && arch < SPR && B unsigned */
#         else if (l_is_Ai2_Bi8_gemm > 0 && l_xgemm_desc_mod.m == 64 &&
#                  io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR && l_b_unsigned) {
#           l_is_supported = 1;
#           io_generated_code->arch = LIBXSMM_X86_AVX512_CLX;
#         }
#         /* i1i8 && m=64 && arch >= SPR && B unsigned */
#         else if (l_is_Ai1_Bi8_gemm > 0 && l_xgemm_desc_mod.m == 64 &&
#                  io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR && l_b_unsigned) {
#           l_is_supported = 1;
#           io_generated_code->arch = LIBXSMM_X86_AVX512_CLX;
#         }

#         if (!l_is_supported) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#           return;
#         }
#       }
#     } else {
#       if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
#            (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8 || LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) &&
#            (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#            ((LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) || (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32))) {
#         /* We are good...  */
#       } else if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) &&
#            (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8 && l_is_Amxfp4_Bbf16_gemm == 0) &&
#            (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) && (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) &&
#            ((LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) || (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32))) {
#         if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#           return;
#         } else if (l_xgemm_desc_mod.k % 2 != 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#           return;
#         } else {
#           /* We are good...  */
#         }
#       } else if ((l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR)) {
#         if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#           return;
#         } else if (l_xgemm_desc_mod.k % 2 != 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#           return;
#         } else {
#           /* We are good...  */
#         }
#       } else if ((l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) && (io_generated_code->arch >= LIBXSMM_X86_AVX2)) {
#         if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#           return;
#         } else if (l_xgemm_desc_mod.k % 32 != 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#           return;
#         } else {
#           /* We are good...  */
#         }
#       } else if ((l_is_Amxfp4_Bi8_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) && (io_generated_code->arch >= LIBXSMM_X86_AVX2_SRF)) {
#         if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0 ) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
#           return;
#         } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#           return;
#         } else if (l_xgemm_desc_mod.k % 32 != 0) {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#           return;
#         } else {
#           /* We are good...  */
#         }
#       } else {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#     }
#   }

#   /* We allow b vnniT for x86 and bf16 whenever possible */
#   if ( ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) && ( io_generated_code->arch >= LIBXSMM_X86_GENERIC ) &&  ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
#       if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ) {
#         /* we are fine, use avx512 path */
#         if (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) {
#           io_generated_code->arch = LIBXSMM_X86_AVX512_SKX;
#         }
#       } else {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#       }
#     } else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#       return;
#     }
#   }

#   /* overwrite VNNI Flag when K == 1 */
#   if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) &&
#        (l_xgemm_desc_mod.k == 1) &&
#        ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
#     l_xgemm_desc_mod.flags = l_xgemm_desc_mod.flags & (~LIBXSMM_GEMM_FLAG_VNNI_A);
#   }

#   if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype )) ||
#        (LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype )) ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#     return;
#   }

#   /* determining vector length depending on architecture and precision */
#   if ( io_generated_code->arch <= LIBXSMM_TARGET_ARCH_GENERIC ) {
#     /* nothing to do */
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     l_vector_length = 2;
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     l_vector_length = 4;
#     if (l_xgemm_desc_mod.k % 4 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_I16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     l_vector_length = 4;
#     if (l_xgemm_desc_mod.k % 2 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     /* some checks as we cannot mask everything */
#     if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       l_vector_length = 8;
#     } else {
#       l_vector_length = 4;
#     }
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) && LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )  ) {
#     l_vector_length = 8;
#     if (l_xgemm_desc_mod.k % 4 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) && ( (LIBXSMM_DATATYPE_I16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ) ) {
#     l_vector_length = 8;
#     if (l_xgemm_desc_mod.k % 2 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) && ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ) ) {
#     /* some checks as we cannot mask everything */
#     if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       l_vector_length = 16;
#     } else {
#       l_vector_length = 8;
#     }
#   } else if ( ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) ) ||
#               ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2_SRF ) && ( l_is_Amxfp4_Bi8_gemm > 0) ) ) {
#     l_vector_length = 8;
#     if (l_xgemm_desc_mod.k % 32 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_SKX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_SKX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 16;
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) &&
#               ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     l_vector_length = 8;
#     /* some checks as we cannot mask everything */
#     if (l_xgemm_desc_mod.k % 2 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) &&
#               ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     l_vector_length = 16;
#     /* some checks as we cannot mask everything */
#     if (l_xgemm_desc_mod.k % 2 != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) &&
#               ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ||
#                 ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ||
#                 ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) )    ) ) {
#     l_vector_length = 8;
#     if ( (l_xgemm_desc_mod.k % 4 != 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) &&
#               ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ||
#                 ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC(  l_xgemm_desc_mod.datatype ) )  ||
#                 ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC(  l_xgemm_desc_mod.datatype ) ) ) ) {
#     l_vector_length = 16;
#     /* some checks as we cannot mask everything */
#     if ( (l_xgemm_desc_mod.k % 4 != 0) ) {
#       if ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (l_xgemm_desc_mod.k % 2 == 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0)) {
#         /* We are good... */
#       } else {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#     }
#     if ((l_is_Ai4_Bi8_gemm > 0) && (l_xgemm_desc_mod.k % 8 != 0)) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#     if ((l_is_Ai2_Bi8_gemm > 0) && (l_xgemm_desc_mod.k % 4 != 0 || l_xgemm_desc_mod.m % 32 != 0)) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#     if ((l_is_Ai1_Bi8_gemm > 0) && (l_xgemm_desc_mod.k % 4 != 0 || l_xgemm_desc_mod.m % 16 != 0)) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) &&
#               ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     /* some checks as we cannot mask everything */
#     if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       l_vector_length = 16;
#       /*l_xgemm_desc_mod.k = l_xgemm_desc_mod.k;*/
#       /*l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb;*/
#     } else {
#       l_vector_length = 8;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) &&
#               ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     /* some checks as we cannot mask everything */
#     if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       l_vector_length = 32;
#       /*l_xgemm_desc_mod.k = l_xgemm_desc_mod.k;*/
#       /*l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb;*/
#     } else {
#       l_vector_length = 16;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_GNR ) &&
#               ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) && (l_xgemm_desc_mod.k % 2 == 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0)) ) {
#     l_vector_length = 16;
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8) &&
#               (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0 && l_is_Amxfp4_Bbf16_gemm == 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX ) {
#         l_vector_length = 16;
#       } else {
#         l_vector_length = 8;
#       }
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) &&
#               (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8 || LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) &&
#               (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0 && io_generated_code->arch < LIBXSMM_X86_AVX512_GNR) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       if (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT) {
#         l_vector_length = 32;
#       } else {
#         l_vector_length = 16;
#       }
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) &&
#               (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_HF8 || LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) &&
#               (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if (((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) && (l_xgemm_desc_mod.k % 2 == 0)) {
#       l_vector_length = 16;
#     } else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) &&
#               (LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       if (LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_COMP_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_IMPLICIT) {
#         l_vector_length = 32;
#       } else {
#         l_vector_length = 16;
#       }
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) &&
#               (LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       l_vector_length = 16;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) &&
#               (LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       l_vector_length = 8;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SPR ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) &&
#               (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8 || LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) &&
#               (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       l_vector_length = 16;
#     }
#   } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
#               ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_SKX ) &&
#               (LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_I8 || LIBXSMM_GEMM_GETENUM_A_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_BF8) &&
#               (LIBXSMM_GEMM_GETENUM_B_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16) &&
#               (LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F16 || LIBXSMM_GEMM_GETENUM_C_PREC( l_xgemm_desc_mod.datatype ) == LIBXSMM_DATATYPE_F32) ) {
#     if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     } else {
#       l_vector_length = 8;
#     }
#   } else if ((l_is_Ai4_Bi8_gemm > 0 || l_is_Ai2_Bi8_gemm > 0  || l_is_Ai1_Bi8_gemm > 0 || l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Abf8_Bf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0 || l_is_Amxfp4_Bfp32_gemm > 0 || l_is_Amxfp4_Bi8_gemm > 0) && ( io_generated_code->arch >= LIBXSMM_AARCH64_V81 ) &&  ( io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT )) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#     return;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 2;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V82 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V82 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 2;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 2;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE128 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV2 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE128 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV2 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 2;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE512 || io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 16;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE512 || io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M4 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 16;
#   } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M4 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch >= LIBXSMM_AARCH64_V81    )  &&
#               ( io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT ) &&
#               (    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) && l_aarch64_bfdot != 0 )
#                 || ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) && l_aarch64_i8dot != 0 ) ) ) {
#     /* TODO (BFDOT): add flags and check on MMLA-formated A/B */
#     /* TODO (BFDOT): add support for at least m % 2 == 0, k % 4 == 0 when running BF16 */
#     /* TODO (BFDOT): adjust checks for future SVE kernels */
#     if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#       if ((l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0)) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#     } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#       if (l_xgemm_desc_mod.k % 4 != 0)  {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#     }
#     /* ASIMD + BFDOT */
#     if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
# #if 0
#       l_vector_length = 4;
# #else
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#      return;
# #endif
#     }
#     /* SVE256 + BFDOT */
#     else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 )  ) {
#       l_vector_length = 8;
#     }
#     /* SVE128 + BFDOT */
#     else if ( io_generated_code->arch == LIBXSMM_AARCH64_SVE128 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV2 ) {
#       l_vector_length = 4;
#     }
#     else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
# #if 0
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#       return;
#     }
# #endif
#   } else if ( ( io_generated_code->arch >= LIBXSMM_AARCH64_V81    )  &&
#               ( io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT ) &&
#               (    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) && l_aarch64_bfdot == 0 )
#                 || ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) && l_aarch64_i8dot == 0) ) ) {
#     /* TODO (MMLA): add flags and check on MMLA-formated A/B */
#     /* TODO (MMLA): add support for at least m % 2 == 0, k % 4 == 0 when running BF16 */
#     /* TODO (MMLA): adjust checks for future SVE kernels */
#     if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#       if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ){
#         /* All good */
#       } else {
#         if (l_xgemm_desc_mod.k % 4 != 0) {
#          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#          return;
#         }
#       }
#     } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#       if (l_xgemm_desc_mod.k % 8 != 0)  {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#       if ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 && (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0 ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#         return;
#       }
#     }
#     /* ASIMD + MMLA */
#     /* TODO: These are not properly implemented yet */
#     if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
# #if 0
#       l_vector_length = 4;
# #else
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#     return;
# #endif
#     }
#     /* SVE256 + MMLA */
#     else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 )  ) {
#       l_vector_length = 8;
#     }
#     /* SVE128 + MMLA */
#     else if ( io_generated_code->arch == LIBXSMM_AARCH64_SVE128 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV2) {
#       l_vector_length = 4;
#     } else {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#       return;
#     }
# #if 0
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#       return;
#     }
# #endif
#   } else if ( ( io_generated_code->arch == LIBXSMM_RV64_MVL128 || io_generated_code->arch == LIBXSMM_RV64_MVL128_LMUL ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else if ( ( io_generated_code->arch == LIBXSMM_RV64_MVL128 || io_generated_code->arch == LIBXSMM_RV64_MVL128_LMUL ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 2;
#   } else if ( ( io_generated_code->arch == LIBXSMM_RV64_MVL256 || io_generated_code->arch == LIBXSMM_RV64_MVL256_LMUL ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 8;
#   } else if ( ( io_generated_code->arch == LIBXSMM_RV64_MVL256 || io_generated_code->arch == LIBXSMM_RV64_MVL256_LMUL ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) {
#     l_vector_length = 4;
#   } else {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
#     return;
#   }

#   /* check LDA */
#   if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) == LIBXSMM_GEMM_FLAG_TRANS_A ) {
#     if ( (l_xgemm_desc_mod.lda < l_xgemm_desc_mod.k) && (l_is_var_ld == 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA_TRANS );
#       return;
#     }
#     if ((LIBXSMM_DATATYPE_F32 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) &&
#         (LIBXSMM_DATATYPE_F64 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) &&
#         (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ) {
#         if (( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) &&  io_generated_code->arch >= LIBXSMM_X86_AVX512_DMR) {
#           /* We are good */
#         } else {
#           LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
#           return;
#         }
#     } else {
#       /* BF16 A transpose is supported forflat A */
#       if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#         return;
#       }
#     }
#   } else {
#     if ( (l_xgemm_desc_mod.lda < l_xgemm_desc_mod.m) && (l_is_var_ld == 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
#       return;
#     }
#   }

#   /* check LDB */
#   if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
#     if ( (l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.n) && (l_is_var_ld == 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB_TRANS );
#       return;
#     }
#   } else {
#     if ( (l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.k) && (l_is_var_ld == 0) ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
#       return;
#     }
#   }

#   /* check LDC */
#   if ( (l_xgemm_desc_mod.ldc < l_xgemm_desc_mod.m) && (l_is_var_ld == 0) ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
#     return;
#   }

#   /* check for trans A cases which are not supported in the generator */
#   if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0 ) {
#     if ( (LIBXSMM_DATATYPE_F32 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) &&
#          (LIBXSMM_DATATYPE_F64 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) &&
#          (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ) {
#       if (( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) && io_generated_code->arch >= LIBXSMM_X86_AVX512_DMR) {
#         /* We are good */
#       } else {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
#         return;
#       }
#     } else {
#       /* BF16 A transpose is supported forflat A */
#       if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#         return;
#       }
#     }
#   }

#   /* check for trans B cases which are not supported in the generator */
#   if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
#     if ( (LIBXSMM_DATATYPE_I16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ||
#          (LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ))    ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
#       return;
#     } else {
#       /* we are fine, we have transpose support */
#     }
#     if ( ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#       if ( (l_aarch64_bfdot == 0 ) && ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 )) {
#         /* we are fine, we do support mmla kernels with B in vnni4t */
#       } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) {
#         LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
#         return;
#       }
#     }
#   }

#   if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
#     if ( (l_aarch64_bfdot == 0 ) && ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 )) {
#       /* we are fine, we do support mmla kernels with B in vnni4t */
#     } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#       return;
#     }
#   }

#   /* check for VNNI flag being set in case of low precision GEMM */
#   /* TODO (MMLA): adjust for aarch64 using i8-MMLA instructions
#   if ( ( LIBXSMM_DATATYPE_I16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ||
#        ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#       return;
#     }
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
#       return;
#     }
#   }
#   if ( ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) {
#     if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
#       return;
#     }
#   }
#   */

#   /* right now we only support eltwise fusion on SPR and BF16 */
#   /* TODO: EVANGELOS -- AMMEND */
# #if 0
#   if ( ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SPR) || (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) ) &&
#        ( l_xgemm_desc_mod.meltw_operation != LIBXSMM_MELTW_OPERATION_NONE ) ) {
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
#     return;
#   }
# #endif

#   /* check if alignment is not possible */
#   if ( 0 != (l_xgemm_desc_mod.lda % l_vector_length) ) {
#     l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_A;
#   }
#   if ( 0 != (l_xgemm_desc_mod.ldc % l_vector_length) ) {
#     l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_C;
#   }

#   if ( io_generated_code->arch <= LIBXSMM_TARGET_ARCH_GENERIC ) {
#     libxsmm_generator_gemm_noarch_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else if ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
#     /* call actual kernel generation with revised parameters */
#     /* TODO: check for VNNI format */
#     if ( ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) && ( io_generated_code->arch < LIBXSMM_X86_ALLFEAT ) ) &&
#          ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ||
#            ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) && io_generated_code->arch >= LIBXSMM_X86_AVX512_GNR) ||
#            ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype ) ) ) &&
#          (((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) || ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 && io_generated_code->arch >= LIBXSMM_X86_AVX512_DMR)) ) {
#       libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
#     } else if ( ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) && ( io_generated_code->arch < LIBXSMM_X86_ALLFEAT ) ) &&
#                 ( (l_is_Abf8_Bbf16_gemm > 0 || l_is_Ahf8_Bbf16_gemm > 0 || l_is_Amxfp4_Bbf16_gemm > 0) || ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) || ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )) || ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( l_xgemm_desc_mod.datatype )))) {
#       libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
#     } else if (( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_GNR ) && ( io_generated_code->arch < LIBXSMM_X86_ALLFEAT ) ) && ( l_is_Abf8_Bf16_gemm > 0 ) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
#       libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
#     } else {
#       libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
#     }
#   } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_V81) || (io_generated_code->arch == LIBXSMM_AARCH64_V82) ) {
#     libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else if ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
#     libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_SVE128) || (io_generated_code->arch == LIBXSMM_AARCH64_NEOV2) ) {
#     libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_SVE256) || (io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) ) {
#     libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_SVE512) || (io_generated_code->arch == LIBXSMM_AARCH64_A64FX) ) {
#     libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else if ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M4 ) {
#     if( LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC(l_xgemm_desc_mod.datatype) &&
#         ((i_xgemm_desc->flags == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI ) ||
#          (i_xgemm_desc->flags == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI + LIBXSMM_GEMM_FLAG_TRANS_B )) ){
#       libxsmm_generator_gemm_aarch64_kernel_sme_het_blocking( io_generated_code, &l_xgemm_desc_mod );
#     } else {
#       libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
#     }
#   } else if ( io_generated_code->arch >= LIBXSMM_RV64_MVL128 ) {
#     libxsmm_generator_gemm_rv64_kernel( io_generated_code, &l_xgemm_desc_mod );
#   } else {
#     printf("Arch %d\n", io_generated_code->arch);
#     LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
#     return;
#   }

#   /* restore arch */
#   io_generated_code->arch = l_saved_arch;
