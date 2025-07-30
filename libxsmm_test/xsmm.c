void libxsmm_matmul(const double* A, const double* B, double* C) {
#ifdef __AVX2__
#ifdef __AVX512F__
#pragma message ("LIBXSMM KERNEL COMPILATION WARNING: compiling AVX2 code on AVX512 or newer architecture: " __FILE__)
#endif
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "pushq %%rbp\n\t"
                       "movq %%rsp, %%rbp\n\t"
                       "subq $192, %%rsp\n\t"
                       "movq $18446744073709551552, %%r10\n\t"
                       "andq %%r10, %%rsp\n\t"
                       "subq $64, %%rsp\n\t"
                       "subq $64, %%rsp\n\t"
                       "movq $0, %%r11\n\t"
                       "33:\n\t"
                       "addq $2, %%r11\n\t"
                       "movq $0, %%r10\n\t"
                       "34:\n\t"
                       "addq $4, %%r10\n\t"
                       "vxorpd %%ymm14, %%ymm14, %%ymm14\n\t"
                       "vxorpd %%ymm15, %%ymm15, %%ymm15\n\t"
                       "vmovupd 0(%%rdi), %%ymm2\n\t"
                       "addq $32, %%rdi\n\t"
                       "vbroadcastsd 0(%%rsi), %%ymm0\n\t"
                       "vfmadd231pd %%ymm2, %%ymm0, %%ymm14\n\t"
                       "vbroadcastsd 32(%%rsi), %%ymm1\n\t"
                       "addq $8, %%rsi\n\t"
                       "vfmadd231pd %%ymm2, %%ymm1, %%ymm15\n\t"
                       "vmovupd 0(%%rdi), %%ymm2\n\t"
                       "addq $32, %%rdi\n\t"
                       "vbroadcastsd 0(%%rsi), %%ymm0\n\t"
                       "vfmadd231pd %%ymm2, %%ymm0, %%ymm14\n\t"
                       "vbroadcastsd 32(%%rsi), %%ymm1\n\t"
                       "addq $8, %%rsi\n\t"
                       "vfmadd231pd %%ymm2, %%ymm1, %%ymm15\n\t"
                       "vmovupd 0(%%rdi), %%ymm2\n\t"
                       "addq $32, %%rdi\n\t"
                       "vbroadcastsd 0(%%rsi), %%ymm0\n\t"
                       "vfmadd231pd %%ymm2, %%ymm0, %%ymm14\n\t"
                       "vbroadcastsd 32(%%rsi), %%ymm1\n\t"
                       "addq $8, %%rsi\n\t"
                       "vfmadd231pd %%ymm2, %%ymm1, %%ymm15\n\t"
                       "vmovupd 0(%%rdi), %%ymm2\n\t"
                       "addq $32, %%rdi\n\t"
                       "vbroadcastsd 0(%%rsi), %%ymm0\n\t"
                       "vfmadd231pd %%ymm2, %%ymm0, %%ymm14\n\t"
                       "vbroadcastsd 32(%%rsi), %%ymm1\n\t"
                       "addq $8, %%rsi\n\t"
                       "vfmadd231pd %%ymm2, %%ymm1, %%ymm15\n\t"
                       "subq $32, %%rsi\n\t"
                       "vmovupd %%ymm14, 0(%%rdx)\n\t"
                       "vmovupd %%ymm15, 32(%%rdx)\n\t"
                       "addq $32, %%rdx\n\t"
                       "subq $96, %%rdi\n\t"
                       "cmpq $4, %%r10\n\t"
                       "jl 34b\n\t"
                       "addq $32, %%rdx\n\t"
                       "addq $64, %%rsi\n\t"
                       "subq $32, %%rdi\n\t"
                       "cmpq $4, %%r11\n\t"
                       "jl 33b\n\t"
                       "movq %%rbp, %%rsp\n\t"
                       "popq %%rbp\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "rdi","rsi","rdx","r10","r11","r12","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif

#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp atomic
#endif
#endif
}

