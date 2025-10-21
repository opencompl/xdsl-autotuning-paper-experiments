from xdsl.dialects import x86
from autotuner.libxsmm_gemm.generator_common import GPRegMapping
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

    builder = generated_code.builder

    builder.insert(x86.ops.DS_MovOp(a, destination=gp_reg_mapping.gp_reg_a))
    builder.insert(x86.ops.DS_MovOp(b, destination=gp_reg_mapping.gp_reg_b))
    builder.insert(x86.ops.DS_MovOp(c, destination=gp_reg_mapping.gp_reg_c))

    match prefetch:
        case GEMMPrefetchType.BL2 | GEMMPrefetchType.AL2:
            raise NotImplementedError


def libxsmm_x86_instruction_close_stream_gemm(
    generated_code: GeneratedCode,
    gp_reg_mapping: GPRegMapping,
    skip_callee_save: bool,
    prefetch: GEMMPrefetchType,
) -> None:
    raise NotImplementedError


# LIBXSMM_API_INTERN
# void libxsmm_x86_instruction_close_stream_gemm( libxsmm_generated_code*       io_generated_code,
#                                                 const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
#                                                 const unsigned int            skip_callee_save,
#                                                 unsigned int                  i_prefetch) {
#   /* TODO: add checks in debug mode */
#   if ( io_generated_code->code_type > 1 ) {
#     /* TODO: this is a very simple System V ABI 64 interface */
#     unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
#     unsigned int l_code_size = io_generated_code->code_size;
#     unsigned int l_max_size = io_generated_code->buffer_size;

#     if (l_max_size < (l_code_size + 10)) {
#       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
#       return;
#     }

#     if ( skip_callee_save == 0 ) {
#       /* pop callee save registers */
# #if defined(_WIN32) || defined(__CYGWIN__)
#       /* pop rsi */
#       l_code_buffer[l_code_size++] = 0x5e;
#       /* pop rdi */
#       l_code_buffer[l_code_size++] = 0x5f;
# #endif
#        /* pop r15 */
#       l_code_buffer[l_code_size++] = 0x41;
#       l_code_buffer[l_code_size++] = 0x5f;
#       /* pop r14 */
#       l_code_buffer[l_code_size++] = 0x41;
#       l_code_buffer[l_code_size++] = 0x5e;
#       /* pop r13 */
#       l_code_buffer[l_code_size++] = 0x41;
#       l_code_buffer[l_code_size++] = 0x5d;
#       /* pop r12 */
#       l_code_buffer[l_code_size++] = 0x41;
#       l_code_buffer[l_code_size++] = 0x5c;
#       /* pop rbx */
#       l_code_buffer[l_code_size++] = 0x5b;

#       /* on windows we also have to restore xmm6-xmm15 */
# #if defined(_WIN32) || defined(__CYGWIN__)
#       {
#         unsigned int l_i;
#         unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
#                                                                                      : LIBXSMM_X86_INSTR_VMOVUPS_LD;
#         /* update code length */
#         io_generated_code->code_size = l_code_size;
#         /* save 10 xmm onto the stack */
#         for (l_i = 0; l_i < 10; ++l_i) {
#           libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code, l_simd_load_instr, 'x', LIBXSMM_X86_GP_REG_RSP,
#             LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
#         }
#         /* increase rsp by 160 (10x16) */
#         libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 160);
#         /* update code length */
#         l_code_size = io_generated_code->code_size;
#       }
# #endif
#     }

#     /* retq */
#     /* TODO: I do not know if this is the correct placement in the generation process */
#     l_code_buffer[l_code_size++] = 0xc3;

#     /* update code length */
#     io_generated_code->code_size = l_code_size;
#   } else if ( io_generated_code->code_type == 1 ) {
#     /* TODO: this is currently System V AMD64 RTL(C) ABI only */
#     char l_new_code[512];
#     int l_max_code_length = 511;
#     int l_code_length = 0;

#     if ( skip_callee_save == 0 ) {
# #if defined(_WIN32) || defined(__CYGWIN__)
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rsi\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rdi\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
# #endif
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r15\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r14\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r13\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r12\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#       l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rbx\n" );
#       libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

#       /* on windows we also have to restore xmm6-xmm15 */
# #if defined(_WIN32) || defined(__CYGWIN__)
#       {
#         unsigned int l_i;
#         unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
#                                                                                      : LIBXSMM_X86_INSTR_VMOVUPS_LD;
#         char l_gp_reg_base_name[4];
#         char l_instr_name[16];

#         libxsmm_get_x86_gp_reg_name(LIBXSMM_X86_GP_REG_RSP, l_gp_reg_base_name, 3);
#         libxsmm_get_x86_instr_name(l_simd_load_instr, l_instr_name, 15);

#         /* save 10 xmm onto the stack */
#         for (l_i = 0; l_i < 10; ++l_i) {
#           l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length,
#             "                       %s %i(%%%s), %%%cmm%u\\n\\t\"\n", l_instr_name, 144 - (l_i * 16), l_gp_reg_base_name,
#             'x', 6 + l_i);
#           libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
#         }
#         /* increase rsp by 160 (10x16) */
#         libxsmm_get_x86_instr_name(LIBXSMM_X86_INSTR_ADDQ, l_instr_name, 15);
#         l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%i, %%%s\n", l_instr_name,
#           160, l_gp_reg_base_name);
#         libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
#       }
# #endif
#     }

#     /* TODO: I do not know if this is the correct placement in the generation process */
#     l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
#     libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#   } else {
#     char l_new_code[1024];
#     int l_max_code_length = 1023;
#     int l_code_length = 0;
#     char l_gp_reg_a[4];
#     char l_gp_reg_b[4];
#     char l_gp_reg_c[4];
#     char l_gp_reg_pre_a[4];
#     char l_gp_reg_pre_b[4];
#     char l_gp_reg_mloop[4];
#     char l_gp_reg_nloop[4];
#     char l_gp_reg_kloop[4];

#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_a, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_b, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_c, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_pre_a, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_pre_b, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_mloop, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_nloop, 3 );
#     libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_kloop, 3 );

#     if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
#       if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
#         l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
#       } else {
#         l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
#       }
#     } else {
#       if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
#         l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
#       } else {
#         l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
#       }
#     }
#     libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#   }
# }
