// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128 128 128 128 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128 128 128 128 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 128 : i64, n = 38 : i64, k = 128 : i64, lda = 128 : i64, ldb = 128 : i64, ldc = 128 : i64, datatype = f32, aligned_a = true, aligned_c = true, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      push rbx
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovaps zmm23, [rdx]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm21, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm20, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovaps zmm5, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovaps zmm6, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovaps zmm7, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovaps zmm8, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovaps zmm9, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovaps zmm10, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovaps zmm11, [rdx+1728]
// CHECK-REGALLOC-NEXT:      vmovaps zmm12, [rdx+2048]
// CHECK-REGALLOC-NEXT:      vmovaps zmm13, [rdx+2112]
// CHECK-REGALLOC-NEXT:      vmovaps zmm14, [rdx+2176]
// CHECK-REGALLOC-NEXT:      vmovaps zmm15, [rdx+2240]
// CHECK-REGALLOC-NEXT:      vmovaps zmm16, [rdx+2560]
// CHECK-REGALLOC-NEXT:      vmovaps zmm17, [rdx+2624]
// CHECK-REGALLOC-NEXT:      vmovaps zmm18, [rdx+2688]
// CHECK-REGALLOC-NEXT:      vmovaps zmm19, [rdx+2752]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm25, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm27, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vmovaps zmm27, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm28, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm25, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vmovaps zmm25, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm28, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vmovaps zmm28, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm27, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      vmovaps [rdx], zmm23
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+64], zmm22
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+128], zmm21
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+192], zmm20
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+512], zmm0
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+576], zmm1
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+640], zmm2
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+704], zmm3
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1024], zmm4
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1088], zmm5
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1152], zmm6
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1216], zmm7
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1536], zmm8
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1600], zmm9
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1664], zmm10
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1728], zmm11
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2048], zmm12
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2112], zmm13
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2176], zmm14
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2240], zmm15
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2560], zmm16
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2624], zmm17
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2688], zmm18
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2752], zmm19
// CHECK-REGALLOC-NEXT:      sub rdi, 65280
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      sub rdi, 512
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      add rdx, 2560
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovaps zmm19, [rdx]
// CHECK-REGALLOC-NEXT:      vmovaps zmm18, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm17, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm16, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovaps zmm15, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovaps zmm14, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovaps zmm13, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovaps zmm12, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovaps zmm11, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovaps zmm10, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovaps zmm9, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovaps zmm8, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovaps zmm7, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovaps zmm6, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovaps zmm5, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdx+1728]
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdx+2048]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdx+2112]
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdx+2176]
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdx+2240]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovaps zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm23, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovaps zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm21, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vmovaps zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm20, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      vmovaps [rdx], zmm19
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+64], zmm18
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+128], zmm17
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+192], zmm16
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+512], zmm15
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+576], zmm14
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+640], zmm13
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+704], zmm12
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1024], zmm11
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1088], zmm10
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1152], zmm9
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1216], zmm8
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1536], zmm7
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1600], zmm6
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1664], zmm5
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1728], zmm4
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2048], zmm3
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2112], zmm2
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2176], zmm1
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2240], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 65280
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      sub rdi, 512
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      add rdx, 2048
// CHECK-REGALLOC-NEXT:      cmp rax, 20
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      ret
