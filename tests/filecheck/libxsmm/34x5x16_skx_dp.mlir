// RUN: libxsmm-gemm  dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm  dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 32
// CHECK-NEXT:      vmovupd zmm12, [rdx]
// CHECK-NEXT:      vmovupd zmm13, [rdx+64]
// CHECK-NEXT:      vmovupd zmm14, [rdx+128]
// CHECK-NEXT:      vmovupd zmm15, [rdx+192]
// CHECK-NEXT:      vmovupd zmm16, [rdx+272]
// CHECK-NEXT:      vmovupd zmm17, [rdx+336]
// CHECK-NEXT:      vmovupd zmm18, [rdx+400]
// CHECK-NEXT:      vmovupd zmm19, [rdx+464]
// CHECK-NEXT:      vmovupd zmm20, [rdx+544]
// CHECK-NEXT:      vmovupd zmm21, [rdx+608]
// CHECK-NEXT:      vmovupd zmm22, [rdx+672]
// CHECK-NEXT:      vmovupd zmm23, [rdx+736]
// CHECK-NEXT:      vmovupd zmm24, [rdx+816]
// CHECK-NEXT:      vmovupd zmm25, [rdx+880]
// CHECK-NEXT:      vmovupd zmm26, [rdx+944]
// CHECK-NEXT:      vmovupd zmm27, [rdx+1008]
// CHECK-NEXT:      vmovupd zmm28, [rdx+1088]
// CHECK-NEXT:      vmovupd zmm29, [rdx+1152]
// CHECK-NEXT:      vmovupd zmm30, [rdx+1216]
// CHECK-NEXT:      vmovupd zmm31, [rdx+1280]
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovupd [rdx], zmm12
// CHECK-NEXT:      vmovupd [rdx+64], zmm13
// CHECK-NEXT:      vmovupd [rdx+128], zmm14
// CHECK-NEXT:      vmovupd [rdx+192], zmm15
// CHECK-NEXT:      vmovupd [rdx+272], zmm16
// CHECK-NEXT:      vmovupd [rdx+336], zmm17
// CHECK-NEXT:      vmovupd [rdx+400], zmm18
// CHECK-NEXT:      vmovupd [rdx+464], zmm19
// CHECK-NEXT:      vmovupd [rdx+544], zmm20
// CHECK-NEXT:      vmovupd [rdx+608], zmm21
// CHECK-NEXT:      vmovupd [rdx+672], zmm22
// CHECK-NEXT:      vmovupd [rdx+736], zmm23
// CHECK-NEXT:      vmovupd [rdx+816], zmm24
// CHECK-NEXT:      vmovupd [rdx+880], zmm25
// CHECK-NEXT:      vmovupd [rdx+944], zmm26
// CHECK-NEXT:      vmovupd [rdx+1008], zmm27
// CHECK-NEXT:      vmovupd [rdx+1088], zmm28
// CHECK-NEXT:      vmovupd [rdx+1152], zmm29
// CHECK-NEXT:      vmovupd [rdx+1216], zmm30
// CHECK-NEXT:      vmovupd [rdx+1280], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 4096
// CHECK-NEXT:      cmp r10, 32
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      mov r15, 3
// CHECK-NEXT:      kmovb k1, r15d
// CHECK-NEXT:      mov r10, 32
// CHECK-NEXT:  [[SCF_M2_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 2
// CHECK-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+272]
// CHECK-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+544]
// CHECK-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+816]
// CHECK-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+1088]
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 272
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-NEXT:      vmovupd [rdx+272] {k1}, zmm28
// CHECK-NEXT:      vmovupd [rdx+544] {k1}, zmm29
// CHECK-NEXT:      vmovupd [rdx+816] {k1}, zmm30
// CHECK-NEXT:      vmovupd [rdx+1088] {k1}, zmm31
// CHECK-NEXT:      add rdx, 16
// CHECK-NEXT:      sub rdi, 4336
// CHECK-NEXT:      cmp r10, 34
// CHECK-NEXT:      jl [[SCF_M2_BODY]]
// CHECK-NEXT:      add rdx, 1088
// CHECK-NEXT:      add rsi, 640
// CHECK-NEXT:      sub rdi, 272
// CHECK-NEXT:      cmp r11, 5
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret

// CHECK-IR-LIBXSMM:       builtin.module {
// CHECK-IR-LIBXSMM-NEXT:    x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-IR-LIBXSMM-NEXT:      %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-IR-LIBXSMM-NEXT:      %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-IR-LIBXSMM-NEXT:      %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb0(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb0(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l33"
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %26 = x86.ri.add %25, 32 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %27 = x86.dm.vmovupd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %28 = x86.dm.vmovupd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %29 = x86.dm.vmovupd [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.dm.vmovupd [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vmovupd [%21 + 272] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dm.vmovupd [%21 + 336] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dm.vmovupd [%21 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dm.vmovupd [%21 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.dm.vmovupd [%21 + 544] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dm.vmovupd [%21 + 608] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.dm.vmovupd [%21 + 672] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dm.vmovupd [%21 + 736] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.dm.vmovupd [%21 + 816] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dm.vmovupd [%21 + 880] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.dm.vmovupd [%21 + 944] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dm.vmovupd [%21 + 1008] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.dm.vmovupd [%21 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.dm.vmovupd [%21 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.dm.vmovupd [%21 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dm.vmovupd [%21 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.dm.vmovupd [%19] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.dm.vmovupd [%19 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.dm.vmovupd [%19 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.dm.vmovupd [%19 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.dm.vbroadcastsd [%20] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %52 = x86.rss.vfmadd231pd %27, %47, %51 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %53 = x86.rss.vfmadd231pd %28, %48, %51 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %54 = x86.rss.vfmadd231pd %29, %49, %51 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %55 = x86.rss.vfmadd231pd %30, %50, %51 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %56 = x86.dm.vbroadcastsd [%20 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %57 = x86.rss.vfmadd231pd %31, %47, %56 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %58 = x86.rss.vfmadd231pd %32, %48, %56 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %59 = x86.rss.vfmadd231pd %33, %49, %56 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %60 = x86.rss.vfmadd231pd %34, %50, %56 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %61 = x86.dm.vbroadcastsd [%20 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %62 = x86.rss.vfmadd231pd %35, %47, %61 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %63 = x86.rss.vfmadd231pd %36, %48, %61 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %64 = x86.rss.vfmadd231pd %37, %49, %61 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %65 = x86.rss.vfmadd231pd %38, %50, %61 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %66 = x86.dm.vbroadcastsd [%20 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %67 = x86.rss.vfmadd231pd %39, %47, %66 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %68 = x86.rss.vfmadd231pd %40, %48, %66 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %69 = x86.rss.vfmadd231pd %41, %49, %66 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %70 = x86.rss.vfmadd231pd %42, %50, %66 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %71 = x86.dm.vbroadcastsd [%20 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %72 = x86.ri.add %20, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %73 = x86.ri.add %19, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %74 = x86.rss.vfmadd231pd %43, %47, %71 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %75 = x86.rss.vfmadd231pd %44, %48, %71 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %76 = x86.rss.vfmadd231pd %45, %49, %71 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %77 = x86.rss.vfmadd231pd %46, %50, %71 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %78 = x86.dm.vmovupd [%73] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %79 = x86.dm.vmovupd [%73 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %80 = x86.dm.vmovupd [%73 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %81 = x86.dm.vmovupd [%73 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %82 = x86.dm.vbroadcastsd [%72] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %83 = x86.rss.vfmadd231pd %52, %78, %82 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %84 = x86.rss.vfmadd231pd %53, %79, %82 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %85 = x86.rss.vfmadd231pd %54, %80, %82 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %86 = x86.rss.vfmadd231pd %55, %81, %82 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %87 = x86.dm.vbroadcastsd [%72 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %88 = x86.rss.vfmadd231pd %57, %78, %87 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %89 = x86.rss.vfmadd231pd %58, %79, %87 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %90 = x86.rss.vfmadd231pd %59, %80, %87 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %91 = x86.rss.vfmadd231pd %60, %81, %87 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %92 = x86.dm.vbroadcastsd [%72 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %93 = x86.rss.vfmadd231pd %62, %78, %92 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %94 = x86.rss.vfmadd231pd %63, %79, %92 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %95 = x86.rss.vfmadd231pd %64, %80, %92 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %96 = x86.rss.vfmadd231pd %65, %81, %92 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %97 = x86.dm.vbroadcastsd [%72 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %98 = x86.rss.vfmadd231pd %67, %78, %97 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %99 = x86.rss.vfmadd231pd %68, %79, %97 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %100 = x86.rss.vfmadd231pd %69, %80, %97 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %101 = x86.rss.vfmadd231pd %70, %81, %97 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %102 = x86.dm.vbroadcastsd [%72 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %103 = x86.ri.add %72, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %104 = x86.ri.add %73, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %105 = x86.rss.vfmadd231pd %74, %78, %102 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %106 = x86.rss.vfmadd231pd %75, %79, %102 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %107 = x86.rss.vfmadd231pd %76, %80, %102 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %108 = x86.rss.vfmadd231pd %77, %81, %102 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %109 = x86.dm.vmovupd [%104] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %110 = x86.dm.vmovupd [%104 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %111 = x86.dm.vmovupd [%104 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %112 = x86.dm.vmovupd [%104 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %113 = x86.dm.vbroadcastsd [%103] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %114 = x86.rss.vfmadd231pd %83, %109, %113 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %115 = x86.rss.vfmadd231pd %84, %110, %113 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.rss.vfmadd231pd %85, %111, %113 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.rss.vfmadd231pd %86, %112, %113 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.dm.vbroadcastsd [%103 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.rss.vfmadd231pd %88, %109, %118 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.rss.vfmadd231pd %89, %110, %118 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.rss.vfmadd231pd %90, %111, %118 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.rss.vfmadd231pd %91, %112, %118 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.dm.vbroadcastsd [%103 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.rss.vfmadd231pd %93, %109, %123 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %125 = x86.rss.vfmadd231pd %94, %110, %123 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %126 = x86.rss.vfmadd231pd %95, %111, %123 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %127 = x86.rss.vfmadd231pd %96, %112, %123 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %128 = x86.dm.vbroadcastsd [%103 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %129 = x86.rss.vfmadd231pd %98, %109, %128 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %130 = x86.rss.vfmadd231pd %99, %110, %128 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %131 = x86.rss.vfmadd231pd %100, %111, %128 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %132 = x86.rss.vfmadd231pd %101, %112, %128 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %133 = x86.dm.vbroadcastsd [%103 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %134 = x86.ri.add %103, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %135 = x86.ri.add %104, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %136 = x86.rss.vfmadd231pd %105, %109, %133 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %137 = x86.rss.vfmadd231pd %106, %110, %133 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %138 = x86.rss.vfmadd231pd %107, %111, %133 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %139 = x86.rss.vfmadd231pd %108, %112, %133 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %140 = x86.dm.vmovupd [%135] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %141 = x86.dm.vmovupd [%135 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %142 = x86.dm.vmovupd [%135 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %143 = x86.dm.vmovupd [%135 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %144 = x86.dm.vbroadcastsd [%134] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %145 = x86.rss.vfmadd231pd %114, %140, %144 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %146 = x86.rss.vfmadd231pd %115, %141, %144 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %147 = x86.rss.vfmadd231pd %116, %142, %144 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %148 = x86.rss.vfmadd231pd %117, %143, %144 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %149 = x86.dm.vbroadcastsd [%134 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %150 = x86.rss.vfmadd231pd %119, %140, %149 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %151 = x86.rss.vfmadd231pd %120, %141, %149 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.rss.vfmadd231pd %121, %142, %149 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.rss.vfmadd231pd %122, %143, %149 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.dm.vbroadcastsd [%134 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.rss.vfmadd231pd %124, %140, %154 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.rss.vfmadd231pd %125, %141, %154 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.rss.vfmadd231pd %126, %142, %154 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.rss.vfmadd231pd %127, %143, %154 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.dm.vbroadcastsd [%134 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.rss.vfmadd231pd %129, %140, %159 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %161 = x86.rss.vfmadd231pd %130, %141, %159 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %162 = x86.rss.vfmadd231pd %131, %142, %159 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %163 = x86.rss.vfmadd231pd %132, %143, %159 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %164 = x86.dm.vbroadcastsd [%134 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %165 = x86.ri.add %134, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %166 = x86.ri.add %135, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %167 = x86.rss.vfmadd231pd %136, %140, %164 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %168 = x86.rss.vfmadd231pd %137, %141, %164 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %169 = x86.rss.vfmadd231pd %138, %142, %164 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %170 = x86.rss.vfmadd231pd %139, %143, %164 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %171 = x86.dm.vmovupd [%166] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %172 = x86.dm.vmovupd [%166 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %173 = x86.dm.vmovupd [%166 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %174 = x86.dm.vmovupd [%166 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %175 = x86.dm.vbroadcastsd [%165] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %176 = x86.rss.vfmadd231pd %145, %171, %175 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %177 = x86.rss.vfmadd231pd %146, %172, %175 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %178 = x86.rss.vfmadd231pd %147, %173, %175 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %179 = x86.rss.vfmadd231pd %148, %174, %175 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %180 = x86.dm.vbroadcastsd [%165 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %181 = x86.rss.vfmadd231pd %150, %171, %180 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %182 = x86.rss.vfmadd231pd %151, %172, %180 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %183 = x86.rss.vfmadd231pd %152, %173, %180 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %184 = x86.rss.vfmadd231pd %153, %174, %180 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %185 = x86.dm.vbroadcastsd [%165 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %186 = x86.rss.vfmadd231pd %155, %171, %185 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %187 = x86.rss.vfmadd231pd %156, %172, %185 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %188 = x86.rss.vfmadd231pd %157, %173, %185 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.rss.vfmadd231pd %158, %174, %185 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.dm.vbroadcastsd [%165 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.rss.vfmadd231pd %160, %171, %190 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.rss.vfmadd231pd %161, %172, %190 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %193 = x86.rss.vfmadd231pd %162, %173, %190 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %194 = x86.rss.vfmadd231pd %163, %174, %190 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %195 = x86.dm.vbroadcastsd [%165 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %196 = x86.ri.add %165, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %197 = x86.ri.add %166, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %198 = x86.rss.vfmadd231pd %167, %171, %195 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %199 = x86.rss.vfmadd231pd %168, %172, %195 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %200 = x86.rss.vfmadd231pd %169, %173, %195 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %201 = x86.rss.vfmadd231pd %170, %174, %195 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %202 = x86.dm.vmovupd [%197] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %203 = x86.dm.vmovupd [%197 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %204 = x86.dm.vmovupd [%197 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %205 = x86.dm.vmovupd [%197 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %206 = x86.dm.vbroadcastsd [%196] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %207 = x86.rss.vfmadd231pd %176, %202, %206 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %208 = x86.rss.vfmadd231pd %177, %203, %206 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %209 = x86.rss.vfmadd231pd %178, %204, %206 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %210 = x86.rss.vfmadd231pd %179, %205, %206 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %211 = x86.dm.vbroadcastsd [%196 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %212 = x86.rss.vfmadd231pd %181, %202, %211 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %213 = x86.rss.vfmadd231pd %182, %203, %211 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %214 = x86.rss.vfmadd231pd %183, %204, %211 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %215 = x86.rss.vfmadd231pd %184, %205, %211 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %216 = x86.dm.vbroadcastsd [%196 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %217 = x86.rss.vfmadd231pd %186, %202, %216 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %218 = x86.rss.vfmadd231pd %187, %203, %216 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %219 = x86.rss.vfmadd231pd %188, %204, %216 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %220 = x86.rss.vfmadd231pd %189, %205, %216 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %221 = x86.dm.vbroadcastsd [%196 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %222 = x86.rss.vfmadd231pd %191, %202, %221 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %223 = x86.rss.vfmadd231pd %192, %203, %221 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %224 = x86.rss.vfmadd231pd %193, %204, %221 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %225 = x86.rss.vfmadd231pd %194, %205, %221 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %226 = x86.dm.vbroadcastsd [%196 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %227 = x86.ri.add %196, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %228 = x86.ri.add %197, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %229 = x86.rss.vfmadd231pd %198, %202, %226 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %230 = x86.rss.vfmadd231pd %199, %203, %226 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %231 = x86.rss.vfmadd231pd %200, %204, %226 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %232 = x86.rss.vfmadd231pd %201, %205, %226 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %233 = x86.dm.vmovupd [%228] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %234 = x86.dm.vmovupd [%228 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %235 = x86.dm.vmovupd [%228 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %236 = x86.dm.vmovupd [%228 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %237 = x86.dm.vbroadcastsd [%227] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %238 = x86.rss.vfmadd231pd %207, %233, %237 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %239 = x86.rss.vfmadd231pd %208, %234, %237 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %240 = x86.rss.vfmadd231pd %209, %235, %237 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %241 = x86.rss.vfmadd231pd %210, %236, %237 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %242 = x86.dm.vbroadcastsd [%227 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %243 = x86.rss.vfmadd231pd %212, %233, %242 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %244 = x86.rss.vfmadd231pd %213, %234, %242 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %245 = x86.rss.vfmadd231pd %214, %235, %242 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %246 = x86.rss.vfmadd231pd %215, %236, %242 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %247 = x86.dm.vbroadcastsd [%227 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %248 = x86.rss.vfmadd231pd %217, %233, %247 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %249 = x86.rss.vfmadd231pd %218, %234, %247 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %250 = x86.rss.vfmadd231pd %219, %235, %247 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %251 = x86.rss.vfmadd231pd %220, %236, %247 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %252 = x86.dm.vbroadcastsd [%227 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %253 = x86.rss.vfmadd231pd %222, %233, %252 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %254 = x86.rss.vfmadd231pd %223, %234, %252 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %255 = x86.rss.vfmadd231pd %224, %235, %252 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %256 = x86.rss.vfmadd231pd %225, %236, %252 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %257 = x86.dm.vbroadcastsd [%227 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %258 = x86.ri.add %227, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %259 = x86.ri.add %228, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %260 = x86.rss.vfmadd231pd %229, %233, %257 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %261 = x86.rss.vfmadd231pd %230, %234, %257 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %262 = x86.rss.vfmadd231pd %231, %235, %257 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %263 = x86.rss.vfmadd231pd %232, %236, %257 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %264 = x86.dm.vmovupd [%259] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %265 = x86.dm.vmovupd [%259 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %266 = x86.dm.vmovupd [%259 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %267 = x86.dm.vmovupd [%259 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %268 = x86.dm.vbroadcastsd [%258] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %269 = x86.rss.vfmadd231pd %238, %264, %268 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %270 = x86.rss.vfmadd231pd %239, %265, %268 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %271 = x86.rss.vfmadd231pd %240, %266, %268 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %272 = x86.rss.vfmadd231pd %241, %267, %268 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %273 = x86.dm.vbroadcastsd [%258 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %274 = x86.rss.vfmadd231pd %243, %264, %273 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %275 = x86.rss.vfmadd231pd %244, %265, %273 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %276 = x86.rss.vfmadd231pd %245, %266, %273 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %277 = x86.rss.vfmadd231pd %246, %267, %273 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %278 = x86.dm.vbroadcastsd [%258 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %279 = x86.rss.vfmadd231pd %248, %264, %278 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %280 = x86.rss.vfmadd231pd %249, %265, %278 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %281 = x86.rss.vfmadd231pd %250, %266, %278 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %282 = x86.rss.vfmadd231pd %251, %267, %278 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %283 = x86.dm.vbroadcastsd [%258 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %284 = x86.rss.vfmadd231pd %253, %264, %283 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %285 = x86.rss.vfmadd231pd %254, %265, %283 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %286 = x86.rss.vfmadd231pd %255, %266, %283 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %287 = x86.rss.vfmadd231pd %256, %267, %283 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %288 = x86.dm.vbroadcastsd [%258 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %289 = x86.ri.add %258, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %290 = x86.ri.add %259, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %291 = x86.rss.vfmadd231pd %260, %264, %288 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %292 = x86.rss.vfmadd231pd %261, %265, %288 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %293 = x86.rss.vfmadd231pd %262, %266, %288 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %294 = x86.rss.vfmadd231pd %263, %267, %288 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %295 = x86.dm.vmovupd [%290] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %296 = x86.dm.vmovupd [%290 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %297 = x86.dm.vmovupd [%290 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %298 = x86.dm.vmovupd [%290 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %299 = x86.dm.vbroadcastsd [%289] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %300 = x86.rss.vfmadd231pd %269, %295, %299 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %301 = x86.rss.vfmadd231pd %270, %296, %299 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %302 = x86.rss.vfmadd231pd %271, %297, %299 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %303 = x86.rss.vfmadd231pd %272, %298, %299 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %304 = x86.dm.vbroadcastsd [%289 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %305 = x86.rss.vfmadd231pd %274, %295, %304 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %306 = x86.rss.vfmadd231pd %275, %296, %304 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %307 = x86.rss.vfmadd231pd %276, %297, %304 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %308 = x86.rss.vfmadd231pd %277, %298, %304 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %309 = x86.dm.vbroadcastsd [%289 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %310 = x86.rss.vfmadd231pd %279, %295, %309 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %311 = x86.rss.vfmadd231pd %280, %296, %309 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %312 = x86.rss.vfmadd231pd %281, %297, %309 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %313 = x86.rss.vfmadd231pd %282, %298, %309 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.dm.vbroadcastsd [%289 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.rss.vfmadd231pd %284, %295, %314 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.rss.vfmadd231pd %285, %296, %314 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.rss.vfmadd231pd %286, %297, %314 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %318 = x86.rss.vfmadd231pd %287, %298, %314 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %319 = x86.dm.vbroadcastsd [%289 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %320 = x86.ri.add %289, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %321 = x86.ri.add %290, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %322 = x86.rss.vfmadd231pd %291, %295, %319 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %323 = x86.rss.vfmadd231pd %292, %296, %319 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %324 = x86.rss.vfmadd231pd %293, %297, %319 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %325 = x86.rss.vfmadd231pd %294, %298, %319 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %326 = x86.dm.vmovupd [%321] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %327 = x86.dm.vmovupd [%321 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %328 = x86.dm.vmovupd [%321 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %329 = x86.dm.vmovupd [%321 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %330 = x86.dm.vbroadcastsd [%320] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %331 = x86.rss.vfmadd231pd %300, %326, %330 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %332 = x86.rss.vfmadd231pd %301, %327, %330 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %333 = x86.rss.vfmadd231pd %302, %328, %330 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %334 = x86.rss.vfmadd231pd %303, %329, %330 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %335 = x86.dm.vbroadcastsd [%320 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %336 = x86.rss.vfmadd231pd %305, %326, %335 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %337 = x86.rss.vfmadd231pd %306, %327, %335 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %338 = x86.rss.vfmadd231pd %307, %328, %335 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %339 = x86.rss.vfmadd231pd %308, %329, %335 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %340 = x86.dm.vbroadcastsd [%320 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %341 = x86.rss.vfmadd231pd %310, %326, %340 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %342 = x86.rss.vfmadd231pd %311, %327, %340 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %343 = x86.rss.vfmadd231pd %312, %328, %340 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %344 = x86.rss.vfmadd231pd %313, %329, %340 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %345 = x86.dm.vbroadcastsd [%320 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %346 = x86.rss.vfmadd231pd %315, %326, %345 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %347 = x86.rss.vfmadd231pd %316, %327, %345 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %348 = x86.rss.vfmadd231pd %317, %328, %345 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %349 = x86.rss.vfmadd231pd %318, %329, %345 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %350 = x86.dm.vbroadcastsd [%320 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %351 = x86.ri.add %320, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %352 = x86.ri.add %321, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %353 = x86.rss.vfmadd231pd %322, %326, %350 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %354 = x86.rss.vfmadd231pd %323, %327, %350 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %355 = x86.rss.vfmadd231pd %324, %328, %350 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %356 = x86.rss.vfmadd231pd %325, %329, %350 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %357 = x86.dm.vmovupd [%352] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %358 = x86.dm.vmovupd [%352 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %359 = x86.dm.vmovupd [%352 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %360 = x86.dm.vmovupd [%352 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %361 = x86.dm.vbroadcastsd [%351] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %362 = x86.rss.vfmadd231pd %331, %357, %361 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %363 = x86.rss.vfmadd231pd %332, %358, %361 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %364 = x86.rss.vfmadd231pd %333, %359, %361 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %365 = x86.rss.vfmadd231pd %334, %360, %361 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %366 = x86.dm.vbroadcastsd [%351 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.rss.vfmadd231pd %336, %357, %366 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %368 = x86.rss.vfmadd231pd %337, %358, %366 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %369 = x86.rss.vfmadd231pd %338, %359, %366 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %370 = x86.rss.vfmadd231pd %339, %360, %366 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %371 = x86.dm.vbroadcastsd [%351 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %372 = x86.rss.vfmadd231pd %341, %357, %371 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %373 = x86.rss.vfmadd231pd %342, %358, %371 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %374 = x86.rss.vfmadd231pd %343, %359, %371 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %375 = x86.rss.vfmadd231pd %344, %360, %371 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %376 = x86.dm.vbroadcastsd [%351 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %377 = x86.rss.vfmadd231pd %346, %357, %376 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %378 = x86.rss.vfmadd231pd %347, %358, %376 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %379 = x86.rss.vfmadd231pd %348, %359, %376 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %380 = x86.rss.vfmadd231pd %349, %360, %376 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %381 = x86.dm.vbroadcastsd [%351 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %382 = x86.ri.add %351, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %383 = x86.ri.add %352, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %384 = x86.rss.vfmadd231pd %353, %357, %381 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %385 = x86.rss.vfmadd231pd %354, %358, %381 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %386 = x86.rss.vfmadd231pd %355, %359, %381 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %387 = x86.rss.vfmadd231pd %356, %360, %381 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %388 = x86.dm.vmovupd [%383] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %389 = x86.dm.vmovupd [%383 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %390 = x86.dm.vmovupd [%383 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %391 = x86.dm.vmovupd [%383 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %392 = x86.dm.vbroadcastsd [%382] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %393 = x86.rss.vfmadd231pd %362, %388, %392 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %394 = x86.rss.vfmadd231pd %363, %389, %392 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %395 = x86.rss.vfmadd231pd %364, %390, %392 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %396 = x86.rss.vfmadd231pd %365, %391, %392 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %397 = x86.dm.vbroadcastsd [%382 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %398 = x86.rss.vfmadd231pd %367, %388, %397 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %399 = x86.rss.vfmadd231pd %368, %389, %397 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %400 = x86.rss.vfmadd231pd %369, %390, %397 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %401 = x86.rss.vfmadd231pd %370, %391, %397 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %402 = x86.dm.vbroadcastsd [%382 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %403 = x86.rss.vfmadd231pd %372, %388, %402 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %404 = x86.rss.vfmadd231pd %373, %389, %402 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %405 = x86.rss.vfmadd231pd %374, %390, %402 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %406 = x86.rss.vfmadd231pd %375, %391, %402 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %407 = x86.dm.vbroadcastsd [%382 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %408 = x86.rss.vfmadd231pd %377, %388, %407 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %409 = x86.rss.vfmadd231pd %378, %389, %407 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %410 = x86.rss.vfmadd231pd %379, %390, %407 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %411 = x86.rss.vfmadd231pd %380, %391, %407 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %412 = x86.dm.vbroadcastsd [%382 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %413 = x86.ri.add %382, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %414 = x86.ri.add %383, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %415 = x86.rss.vfmadd231pd %384, %388, %412 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %416 = x86.rss.vfmadd231pd %385, %389, %412 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %417 = x86.rss.vfmadd231pd %386, %390, %412 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %418 = x86.rss.vfmadd231pd %387, %391, %412 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %419 = x86.dm.vmovupd [%414] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %420 = x86.dm.vmovupd [%414 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %421 = x86.dm.vmovupd [%414 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %422 = x86.dm.vmovupd [%414 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %423 = x86.dm.vbroadcastsd [%413] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %424 = x86.rss.vfmadd231pd %393, %419, %423 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %425 = x86.rss.vfmadd231pd %394, %420, %423 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %426 = x86.rss.vfmadd231pd %395, %421, %423 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %427 = x86.rss.vfmadd231pd %396, %422, %423 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %428 = x86.dm.vbroadcastsd [%413 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %429 = x86.rss.vfmadd231pd %398, %419, %428 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %430 = x86.rss.vfmadd231pd %399, %420, %428 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %431 = x86.rss.vfmadd231pd %400, %421, %428 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %432 = x86.rss.vfmadd231pd %401, %422, %428 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %433 = x86.dm.vbroadcastsd [%413 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %434 = x86.rss.vfmadd231pd %403, %419, %433 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %435 = x86.rss.vfmadd231pd %404, %420, %433 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %436 = x86.rss.vfmadd231pd %405, %421, %433 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %437 = x86.rss.vfmadd231pd %406, %422, %433 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %438 = x86.dm.vbroadcastsd [%413 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %439 = x86.rss.vfmadd231pd %408, %419, %438 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %440 = x86.rss.vfmadd231pd %409, %420, %438 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %441 = x86.rss.vfmadd231pd %410, %421, %438 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %442 = x86.rss.vfmadd231pd %411, %422, %438 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %443 = x86.dm.vbroadcastsd [%413 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %444 = x86.ri.add %413, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %445 = x86.ri.add %414, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %446 = x86.rss.vfmadd231pd %415, %419, %443 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %447 = x86.rss.vfmadd231pd %416, %420, %443 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %448 = x86.rss.vfmadd231pd %417, %421, %443 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %449 = x86.rss.vfmadd231pd %418, %422, %443 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %450 = x86.dm.vmovupd [%445] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %451 = x86.dm.vmovupd [%445 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %452 = x86.dm.vmovupd [%445 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %453 = x86.dm.vmovupd [%445 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %454 = x86.dm.vbroadcastsd [%444] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %455 = x86.rss.vfmadd231pd %424, %450, %454 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %456 = x86.rss.vfmadd231pd %425, %451, %454 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %457 = x86.rss.vfmadd231pd %426, %452, %454 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %458 = x86.rss.vfmadd231pd %427, %453, %454 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %459 = x86.dm.vbroadcastsd [%444 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %460 = x86.rss.vfmadd231pd %429, %450, %459 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %461 = x86.rss.vfmadd231pd %430, %451, %459 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %462 = x86.rss.vfmadd231pd %431, %452, %459 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %463 = x86.rss.vfmadd231pd %432, %453, %459 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %464 = x86.dm.vbroadcastsd [%444 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %465 = x86.rss.vfmadd231pd %434, %450, %464 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %466 = x86.rss.vfmadd231pd %435, %451, %464 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %467 = x86.rss.vfmadd231pd %436, %452, %464 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %468 = x86.rss.vfmadd231pd %437, %453, %464 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %469 = x86.dm.vbroadcastsd [%444 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %470 = x86.rss.vfmadd231pd %439, %450, %469 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %471 = x86.rss.vfmadd231pd %440, %451, %469 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %472 = x86.rss.vfmadd231pd %441, %452, %469 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %473 = x86.rss.vfmadd231pd %442, %453, %469 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %474 = x86.dm.vbroadcastsd [%444 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %475 = x86.ri.add %444, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %476 = x86.ri.add %445, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %477 = x86.rss.vfmadd231pd %446, %450, %474 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %478 = x86.rss.vfmadd231pd %447, %451, %474 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %479 = x86.rss.vfmadd231pd %448, %452, %474 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %480 = x86.rss.vfmadd231pd %449, %453, %474 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %481 = x86.dm.vmovupd [%476] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %482 = x86.dm.vmovupd [%476 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %483 = x86.dm.vmovupd [%476 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %484 = x86.dm.vmovupd [%476 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %485 = x86.dm.vbroadcastsd [%475] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %486 = x86.rss.vfmadd231pd %455, %481, %485 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %487 = x86.rss.vfmadd231pd %456, %482, %485 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %488 = x86.rss.vfmadd231pd %457, %483, %485 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %489 = x86.rss.vfmadd231pd %458, %484, %485 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %490 = x86.dm.vbroadcastsd [%475 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %491 = x86.rss.vfmadd231pd %460, %481, %490 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %492 = x86.rss.vfmadd231pd %461, %482, %490 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %493 = x86.rss.vfmadd231pd %462, %483, %490 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %494 = x86.rss.vfmadd231pd %463, %484, %490 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %495 = x86.dm.vbroadcastsd [%475 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %496 = x86.rss.vfmadd231pd %465, %481, %495 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %497 = x86.rss.vfmadd231pd %466, %482, %495 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %498 = x86.rss.vfmadd231pd %467, %483, %495 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %499 = x86.rss.vfmadd231pd %468, %484, %495 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %500 = x86.dm.vbroadcastsd [%475 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %501 = x86.rss.vfmadd231pd %470, %481, %500 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %502 = x86.rss.vfmadd231pd %471, %482, %500 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %503 = x86.rss.vfmadd231pd %472, %483, %500 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %504 = x86.rss.vfmadd231pd %473, %484, %500 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %505 = x86.dm.vbroadcastsd [%475 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %506 = x86.ri.add %475, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %507 = x86.ri.add %476, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %508 = x86.rss.vfmadd231pd %477, %481, %505 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %509 = x86.rss.vfmadd231pd %478, %482, %505 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %510 = x86.rss.vfmadd231pd %479, %483, %505 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %511 = x86.rss.vfmadd231pd %480, %484, %505 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %512 = x86.dm.vmovupd [%507] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %513 = x86.dm.vmovupd [%507 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %514 = x86.dm.vmovupd [%507 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %515 = x86.dm.vmovupd [%507 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %516 = x86.dm.vbroadcastsd [%506] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %517 = x86.rss.vfmadd231pd %486, %512, %516 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %518 = x86.rss.vfmadd231pd %487, %513, %516 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %519 = x86.rss.vfmadd231pd %488, %514, %516 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %520 = x86.rss.vfmadd231pd %489, %515, %516 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %521 = x86.dm.vbroadcastsd [%506 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %522 = x86.rss.vfmadd231pd %491, %512, %521 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %523 = x86.rss.vfmadd231pd %492, %513, %521 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %524 = x86.rss.vfmadd231pd %493, %514, %521 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %525 = x86.rss.vfmadd231pd %494, %515, %521 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %526 = x86.dm.vbroadcastsd [%506 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %527 = x86.rss.vfmadd231pd %496, %512, %526 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %528 = x86.rss.vfmadd231pd %497, %513, %526 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %529 = x86.rss.vfmadd231pd %498, %514, %526 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %530 = x86.rss.vfmadd231pd %499, %515, %526 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %531 = x86.dm.vbroadcastsd [%506 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %532 = x86.rss.vfmadd231pd %501, %512, %531 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %533 = x86.rss.vfmadd231pd %502, %513, %531 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %534 = x86.rss.vfmadd231pd %503, %514, %531 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %535 = x86.rss.vfmadd231pd %504, %515, %531 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %536 = x86.dm.vbroadcastsd [%506 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %537 = x86.ri.add %506, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %538 = x86.ri.add %507, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %539 = x86.rss.vfmadd231pd %508, %512, %536 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %540 = x86.rss.vfmadd231pd %509, %513, %536 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %541 = x86.rss.vfmadd231pd %510, %514, %536 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %542 = x86.rss.vfmadd231pd %511, %515, %536 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21], %517 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 64], %518 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 128], %519 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 192], %520 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 272], %522 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 336], %523 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 400], %524 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 464], %525 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 544], %527 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 608], %528 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 672], %529 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 736], %530 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 816], %532 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 880], %533 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 944], %534 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 1008], %535 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 1088], %539 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 1152], %540 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 1216], %541 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%21 + 1280], %542 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %543 = x86.ri.add %21, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %544 = x86.ri.sub %538, 4096 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %545 = x86.si.cmp %26, 32 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %545 : !x86.rflags<rflags>, ^bb1(%544 : !x86.reg64<rdi>, %537 : !x86.reg64<rsi>, %543 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>), ^bb2(%544 : !x86.reg64<rdi>, %537 : !x86.reg64<rsi>, %543 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%546: !x86.reg64<rdi>, %547: !x86.reg64<rsi>, %548: !x86.reg64<rdx>, %549: !x86.reg64<rbp>, %550: !x86.reg64<rsp>, %551: !x86.reg64<r11>, %552: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %553 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %554 = x86.ks.kmovb %553 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %555 = x86.di.mov 32 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb3(%546 : !x86.reg64<rdi>, %547 : !x86.reg64<rsi>, %548 : !x86.reg64<rdx>, %549 : !x86.reg64<rbp>, %550 : !x86.reg64<rsp>, %551 : !x86.reg64<r11>, %555 : !x86.reg64<r10>, %554 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%556: !x86.reg64<rdi>, %557: !x86.reg64<rsi>, %558: !x86.reg64<rdx>, %559: !x86.reg64<rbp>, %560: !x86.reg64<rsp>, %561: !x86.reg64<r11>, %562: !x86.reg64<r10>, %563: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l35"
// CHECK-IR-LIBXSMM-NEXT:      %564 = x86.ri.add %562, 2 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %565 = x86.dmk.vmovupd[%558], %563 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %566 = x86.dmk.vmovupd[%558 + 272], %563 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %567 = x86.dmk.vmovupd[%558 + 544], %563 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %568 = x86.dmk.vmovupd[%558 + 816], %563 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %569 = x86.dmk.vmovupd[%558 + 1088], %563 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %570 = x86.dmk.vmovupd[%556], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %571 = x86.dm.vbroadcastsd [%557] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %572 = x86.rss.vfmadd231pd %565, %570, %571 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %573 = x86.dm.vbroadcastsd [%557 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %574 = x86.rss.vfmadd231pd %566, %570, %573 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %575 = x86.dm.vbroadcastsd [%557 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %576 = x86.rss.vfmadd231pd %567, %570, %575 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %577 = x86.dm.vbroadcastsd [%557 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %578 = x86.rss.vfmadd231pd %568, %570, %577 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %579 = x86.dm.vbroadcastsd [%557 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %580 = x86.ri.add %557, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %581 = x86.ri.add %556, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %582 = x86.rss.vfmadd231pd %569, %570, %579 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %583 = x86.dmk.vmovupd[%581], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %584 = x86.dm.vbroadcastsd [%580] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %585 = x86.rss.vfmadd231pd %572, %583, %584 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %586 = x86.dm.vbroadcastsd [%580 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %587 = x86.rss.vfmadd231pd %574, %583, %586 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %588 = x86.dm.vbroadcastsd [%580 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %589 = x86.rss.vfmadd231pd %576, %583, %588 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %590 = x86.dm.vbroadcastsd [%580 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %591 = x86.rss.vfmadd231pd %578, %583, %590 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %592 = x86.dm.vbroadcastsd [%580 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %593 = x86.ri.add %580, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %594 = x86.ri.add %581, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %595 = x86.rss.vfmadd231pd %582, %583, %592 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %596 = x86.dmk.vmovupd[%594], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %597 = x86.dm.vbroadcastsd [%593] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %598 = x86.rss.vfmadd231pd %585, %596, %597 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %599 = x86.dm.vbroadcastsd [%593 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %600 = x86.rss.vfmadd231pd %587, %596, %599 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %601 = x86.dm.vbroadcastsd [%593 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %602 = x86.rss.vfmadd231pd %589, %596, %601 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %603 = x86.dm.vbroadcastsd [%593 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %604 = x86.rss.vfmadd231pd %591, %596, %603 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %605 = x86.dm.vbroadcastsd [%593 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %606 = x86.ri.add %593, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %607 = x86.ri.add %594, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %608 = x86.rss.vfmadd231pd %595, %596, %605 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %609 = x86.dmk.vmovupd[%607], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %610 = x86.dm.vbroadcastsd [%606] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %611 = x86.rss.vfmadd231pd %598, %609, %610 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %612 = x86.dm.vbroadcastsd [%606 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %613 = x86.rss.vfmadd231pd %600, %609, %612 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %614 = x86.dm.vbroadcastsd [%606 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %615 = x86.rss.vfmadd231pd %602, %609, %614 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %616 = x86.dm.vbroadcastsd [%606 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %617 = x86.rss.vfmadd231pd %604, %609, %616 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %618 = x86.dm.vbroadcastsd [%606 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %619 = x86.ri.add %606, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %620 = x86.ri.add %607, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %621 = x86.rss.vfmadd231pd %608, %609, %618 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %622 = x86.dmk.vmovupd[%620], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %623 = x86.dm.vbroadcastsd [%619] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %624 = x86.rss.vfmadd231pd %611, %622, %623 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %625 = x86.dm.vbroadcastsd [%619 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %626 = x86.rss.vfmadd231pd %613, %622, %625 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %627 = x86.dm.vbroadcastsd [%619 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %628 = x86.rss.vfmadd231pd %615, %622, %627 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %629 = x86.dm.vbroadcastsd [%619 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %630 = x86.rss.vfmadd231pd %617, %622, %629 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %631 = x86.dm.vbroadcastsd [%619 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %632 = x86.ri.add %619, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %633 = x86.ri.add %620, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %634 = x86.rss.vfmadd231pd %621, %622, %631 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %635 = x86.dmk.vmovupd[%633], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %636 = x86.dm.vbroadcastsd [%632] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %637 = x86.rss.vfmadd231pd %624, %635, %636 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %638 = x86.dm.vbroadcastsd [%632 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %639 = x86.rss.vfmadd231pd %626, %635, %638 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %640 = x86.dm.vbroadcastsd [%632 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %641 = x86.rss.vfmadd231pd %628, %635, %640 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %642 = x86.dm.vbroadcastsd [%632 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %643 = x86.rss.vfmadd231pd %630, %635, %642 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %644 = x86.dm.vbroadcastsd [%632 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %645 = x86.ri.add %632, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %646 = x86.ri.add %633, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %647 = x86.rss.vfmadd231pd %634, %635, %644 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %648 = x86.dmk.vmovupd[%646], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %649 = x86.dm.vbroadcastsd [%645] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %650 = x86.rss.vfmadd231pd %637, %648, %649 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %651 = x86.dm.vbroadcastsd [%645 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %652 = x86.rss.vfmadd231pd %639, %648, %651 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %653 = x86.dm.vbroadcastsd [%645 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %654 = x86.rss.vfmadd231pd %641, %648, %653 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %655 = x86.dm.vbroadcastsd [%645 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %656 = x86.rss.vfmadd231pd %643, %648, %655 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %657 = x86.dm.vbroadcastsd [%645 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %658 = x86.ri.add %645, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %659 = x86.ri.add %646, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %660 = x86.rss.vfmadd231pd %647, %648, %657 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %661 = x86.dmk.vmovupd[%659], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %662 = x86.dm.vbroadcastsd [%658] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %663 = x86.rss.vfmadd231pd %650, %661, %662 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %664 = x86.dm.vbroadcastsd [%658 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %665 = x86.rss.vfmadd231pd %652, %661, %664 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %666 = x86.dm.vbroadcastsd [%658 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %667 = x86.rss.vfmadd231pd %654, %661, %666 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %668 = x86.dm.vbroadcastsd [%658 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %669 = x86.rss.vfmadd231pd %656, %661, %668 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %670 = x86.dm.vbroadcastsd [%658 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %671 = x86.ri.add %658, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %672 = x86.ri.add %659, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %673 = x86.rss.vfmadd231pd %660, %661, %670 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %674 = x86.dmk.vmovupd[%672], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %675 = x86.dm.vbroadcastsd [%671] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %676 = x86.rss.vfmadd231pd %663, %674, %675 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %677 = x86.dm.vbroadcastsd [%671 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %678 = x86.rss.vfmadd231pd %665, %674, %677 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %679 = x86.dm.vbroadcastsd [%671 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %680 = x86.rss.vfmadd231pd %667, %674, %679 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %681 = x86.dm.vbroadcastsd [%671 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %682 = x86.rss.vfmadd231pd %669, %674, %681 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %683 = x86.dm.vbroadcastsd [%671 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %684 = x86.ri.add %671, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %685 = x86.ri.add %672, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %686 = x86.rss.vfmadd231pd %673, %674, %683 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %687 = x86.dmk.vmovupd[%685], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %688 = x86.dm.vbroadcastsd [%684] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %689 = x86.rss.vfmadd231pd %676, %687, %688 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %690 = x86.dm.vbroadcastsd [%684 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %691 = x86.rss.vfmadd231pd %678, %687, %690 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %692 = x86.dm.vbroadcastsd [%684 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %693 = x86.rss.vfmadd231pd %680, %687, %692 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %694 = x86.dm.vbroadcastsd [%684 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %695 = x86.rss.vfmadd231pd %682, %687, %694 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %696 = x86.dm.vbroadcastsd [%684 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %697 = x86.ri.add %684, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %698 = x86.ri.add %685, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %699 = x86.rss.vfmadd231pd %686, %687, %696 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %700 = x86.dmk.vmovupd[%698], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %701 = x86.dm.vbroadcastsd [%697] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %702 = x86.rss.vfmadd231pd %689, %700, %701 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %703 = x86.dm.vbroadcastsd [%697 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %704 = x86.rss.vfmadd231pd %691, %700, %703 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %705 = x86.dm.vbroadcastsd [%697 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %706 = x86.rss.vfmadd231pd %693, %700, %705 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %707 = x86.dm.vbroadcastsd [%697 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %708 = x86.rss.vfmadd231pd %695, %700, %707 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %709 = x86.dm.vbroadcastsd [%697 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %710 = x86.ri.add %697, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %711 = x86.ri.add %698, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %712 = x86.rss.vfmadd231pd %699, %700, %709 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %713 = x86.dmk.vmovupd[%711], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %714 = x86.dm.vbroadcastsd [%710] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %715 = x86.rss.vfmadd231pd %702, %713, %714 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %716 = x86.dm.vbroadcastsd [%710 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %717 = x86.rss.vfmadd231pd %704, %713, %716 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %718 = x86.dm.vbroadcastsd [%710 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %719 = x86.rss.vfmadd231pd %706, %713, %718 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %720 = x86.dm.vbroadcastsd [%710 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %721 = x86.rss.vfmadd231pd %708, %713, %720 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %722 = x86.dm.vbroadcastsd [%710 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %723 = x86.ri.add %710, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %724 = x86.ri.add %711, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %725 = x86.rss.vfmadd231pd %712, %713, %722 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %726 = x86.dmk.vmovupd[%724], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %727 = x86.dm.vbroadcastsd [%723] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %728 = x86.rss.vfmadd231pd %715, %726, %727 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %729 = x86.dm.vbroadcastsd [%723 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %730 = x86.rss.vfmadd231pd %717, %726, %729 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %731 = x86.dm.vbroadcastsd [%723 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %732 = x86.rss.vfmadd231pd %719, %726, %731 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %733 = x86.dm.vbroadcastsd [%723 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %734 = x86.rss.vfmadd231pd %721, %726, %733 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %735 = x86.dm.vbroadcastsd [%723 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %736 = x86.ri.add %723, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %737 = x86.ri.add %724, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %738 = x86.rss.vfmadd231pd %725, %726, %735 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %739 = x86.dmk.vmovupd[%737], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %740 = x86.dm.vbroadcastsd [%736] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %741 = x86.rss.vfmadd231pd %728, %739, %740 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %742 = x86.dm.vbroadcastsd [%736 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %743 = x86.rss.vfmadd231pd %730, %739, %742 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %744 = x86.dm.vbroadcastsd [%736 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %745 = x86.rss.vfmadd231pd %732, %739, %744 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %746 = x86.dm.vbroadcastsd [%736 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %747 = x86.rss.vfmadd231pd %734, %739, %746 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %748 = x86.dm.vbroadcastsd [%736 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %749 = x86.ri.add %736, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %750 = x86.ri.add %737, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %751 = x86.rss.vfmadd231pd %738, %739, %748 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %752 = x86.dmk.vmovupd[%750], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %753 = x86.dm.vbroadcastsd [%749] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %754 = x86.rss.vfmadd231pd %741, %752, %753 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %755 = x86.dm.vbroadcastsd [%749 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %756 = x86.rss.vfmadd231pd %743, %752, %755 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %757 = x86.dm.vbroadcastsd [%749 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %758 = x86.rss.vfmadd231pd %745, %752, %757 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %759 = x86.dm.vbroadcastsd [%749 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %760 = x86.rss.vfmadd231pd %747, %752, %759 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %761 = x86.dm.vbroadcastsd [%749 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %762 = x86.ri.add %749, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %763 = x86.ri.add %750, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %764 = x86.rss.vfmadd231pd %751, %752, %761 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %765 = x86.dmk.vmovupd[%763], %563 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %766 = x86.dm.vbroadcastsd [%762] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %767 = x86.rss.vfmadd231pd %754, %765, %766 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %768 = x86.dm.vbroadcastsd [%762 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %769 = x86.rss.vfmadd231pd %756, %765, %768 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %770 = x86.dm.vbroadcastsd [%762 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %771 = x86.rss.vfmadd231pd %758, %765, %770 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %772 = x86.dm.vbroadcastsd [%762 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %773 = x86.rss.vfmadd231pd %760, %765, %772 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %774 = x86.dm.vbroadcastsd [%762 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %775 = x86.ri.add %762, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %776 = x86.ri.add %763, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %777 = x86.rss.vfmadd231pd %764, %765, %774 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%558], %767, %563 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%558 + 272], %769, %563 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%558 + 544], %771, %563 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%558 + 816], %773, %563 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%558 + 1088], %777, %563 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %778 = x86.ri.add %558, 16 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %779 = x86.ri.sub %776, 4336 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %780 = x86.si.cmp %564, 34 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %780 : !x86.rflags<rflags>, ^bb3(%779 : !x86.reg64<rdi>, %775 : !x86.reg64<rsi>, %778 : !x86.reg64<rdx>, %559 : !x86.reg64<rbp>, %560 : !x86.reg64<rsp>, %561 : !x86.reg64<r11>, %564 : !x86.reg64<r10>, %563 : !x86.avx512maskreg<k1>), ^bb4(%779 : !x86.reg64<rdi>, %775 : !x86.reg64<rsi>, %778 : !x86.reg64<rdx>, %559 : !x86.reg64<rbp>, %560 : !x86.reg64<rsp>, %561 : !x86.reg64<r11>, %564 : !x86.reg64<r10>, %563 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb4(%781: !x86.reg64<rdi>, %782: !x86.reg64<rsi>, %783: !x86.reg64<rdx>, %784: !x86.reg64<rbp>, %785: !x86.reg64<rsp>, %786: !x86.reg64<r11>, %787: !x86.reg64<r10>, %788: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      %789 = x86.ri.add %783, 1088 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %790 = x86.ri.add %782, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %791 = x86.ri.sub %781, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %792 = x86.si.cmp %786, 5 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %792 : !x86.rflags<rflags>, ^bb0(%791 : !x86.reg64<rdi>, %790 : !x86.reg64<rsi>, %789 : !x86.reg64<rdx>, %784 : !x86.reg64<rbp>, %785 : !x86.reg64<rsp>, %786 : !x86.reg64<r11>), ^bb5(%791 : !x86.reg64<rdi>, %790 : !x86.reg64<rsi>, %789 : !x86.reg64<rdx>, %784 : !x86.reg64<rbp>, %785 : !x86.reg64<rsp>, %786 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb5(%793: !x86.reg64<rdi>, %794: !x86.reg64<rsi>, %795: !x86.reg64<rdx>, %796: !x86.reg64<rbp>, %797: !x86.reg64<rsp>, %798: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %799 = x86.ds.mov %796 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %800, %801 = x86.d.pop %799 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }
