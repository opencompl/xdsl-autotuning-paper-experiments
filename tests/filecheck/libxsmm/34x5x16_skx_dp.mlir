// RUN: libxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push r15
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      mov rbp, rsp
// CHECK-MANUAL-NEXT:      sub rsp, 192
// CHECK-MANUAL-NEXT:      mov r10, -64
// CHECK-MANUAL-NEXT:      and rsp, r10
// CHECK-MANUAL-NEXT:      mov r11, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_0:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 5
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_1:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 32
// CHECK-MANUAL-NEXT:      vmovupd zmm12, [rdx]
// CHECK-MANUAL-NEXT:      vmovupd zmm13, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm14, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm15, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovupd zmm16, [rdx+272]
// CHECK-MANUAL-NEXT:      vmovupd zmm17, [rdx+336]
// CHECK-MANUAL-NEXT:      vmovupd zmm18, [rdx+400]
// CHECK-MANUAL-NEXT:      vmovupd zmm19, [rdx+464]
// CHECK-MANUAL-NEXT:      vmovupd zmm20, [rdx+544]
// CHECK-MANUAL-NEXT:      vmovupd zmm21, [rdx+608]
// CHECK-MANUAL-NEXT:      vmovupd zmm22, [rdx+672]
// CHECK-MANUAL-NEXT:      vmovupd zmm23, [rdx+736]
// CHECK-MANUAL-NEXT:      vmovupd zmm24, [rdx+816]
// CHECK-MANUAL-NEXT:      vmovupd zmm25, [rdx+880]
// CHECK-MANUAL-NEXT:      vmovupd zmm26, [rdx+944]
// CHECK-MANUAL-NEXT:      vmovupd zmm27, [rdx+1008]
// CHECK-MANUAL-NEXT:      vmovupd zmm28, [rdx+1088]
// CHECK-MANUAL-NEXT:      vmovupd zmm29, [rdx+1152]
// CHECK-MANUAL-NEXT:      vmovupd zmm30, [rdx+1216]
// CHECK-MANUAL-NEXT:      vmovupd zmm31, [rdx+1280]
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 272
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      sub rsi, 128
// CHECK-MANUAL-NEXT:      vmovupd [rdx], zmm12
// CHECK-MANUAL-NEXT:      vmovupd [rdx+64], zmm13
// CHECK-MANUAL-NEXT:      vmovupd [rdx+128], zmm14
// CHECK-MANUAL-NEXT:      vmovupd [rdx+192], zmm15
// CHECK-MANUAL-NEXT:      vmovupd [rdx+272], zmm16
// CHECK-MANUAL-NEXT:      vmovupd [rdx+336], zmm17
// CHECK-MANUAL-NEXT:      vmovupd [rdx+400], zmm18
// CHECK-MANUAL-NEXT:      vmovupd [rdx+464], zmm19
// CHECK-MANUAL-NEXT:      vmovupd [rdx+544], zmm20
// CHECK-MANUAL-NEXT:      vmovupd [rdx+608], zmm21
// CHECK-MANUAL-NEXT:      vmovupd [rdx+672], zmm22
// CHECK-MANUAL-NEXT:      vmovupd [rdx+736], zmm23
// CHECK-MANUAL-NEXT:      vmovupd [rdx+816], zmm24
// CHECK-MANUAL-NEXT:      vmovupd [rdx+880], zmm25
// CHECK-MANUAL-NEXT:      vmovupd [rdx+944], zmm26
// CHECK-MANUAL-NEXT:      vmovupd [rdx+1008], zmm27
// CHECK-MANUAL-NEXT:      vmovupd [rdx+1088], zmm28
// CHECK-MANUAL-NEXT:      vmovupd [rdx+1152], zmm29
// CHECK-MANUAL-NEXT:      vmovupd [rdx+1216], zmm30
// CHECK-MANUAL-NEXT:      vmovupd [rdx+1280], zmm31
// CHECK-MANUAL-NEXT:      add rdx, 256
// CHECK-MANUAL-NEXT:      sub rdi, 4096
// CHECK-MANUAL-NEXT:      cmp r10, 32
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_1]]
// CHECK-MANUAL-NEXT:      mov r15, 3
// CHECK-MANUAL-NEXT:      kmovb k1, r15d
// CHECK-MANUAL-NEXT:      mov r10, 32
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_2:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 2
// CHECK-MANUAL-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-MANUAL-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+272]
// CHECK-MANUAL-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+544]
// CHECK-MANUAL-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+816]
// CHECK-MANUAL-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+1088]
// CHECK-MANUAL-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-MANUAL-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-MANUAL-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-MANUAL-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-MANUAL-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-MANUAL-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-MANUAL-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-MANUAL-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-MANUAL-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-MANUAL-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-MANUAL-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-MANUAL-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-MANUAL-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-MANUAL-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-MANUAL-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+272]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm0, [rsi]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+128]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+256]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+384]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+512]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+544]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+8]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+136]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+264]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+392]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+520]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+816]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+16]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+144]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+272]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+400]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+528]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+1088]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+24]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+152]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+280]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+408]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+536]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+1360]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+32]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+160]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+288]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+416]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+544]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+1632]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+40]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+168]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+296]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+424]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+552]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+1904]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+48]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+176]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+304]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+432]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+560]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+2176]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+56]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+184]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+312]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+440]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+568]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+2448]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+64]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+192]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+320]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+448]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+576]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+2720]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+72]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+200]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+328]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+456]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+584]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+2992]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+80]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+208]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+336]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+464]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+592]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+3264]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+88]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+216]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+344]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+472]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+600]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+3536]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+96]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+224]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+352]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+480]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+608]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+3808]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+104]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+232]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+360]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+488]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+616]{1to8}
// CHECK-MANUAL-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+4080]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+112]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+240]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+368]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+496]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+624]{1to8}
// CHECK-MANUAL-NEXT:      add rdi, 4352
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+120]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+248]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+376]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+504]{1to8}
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+632]{1to8}
// CHECK-MANUAL-NEXT:      add rsi, 128
// CHECK-MANUAL-NEXT:      vaddpd zmm27, zmm22, zmm27
// CHECK-MANUAL-NEXT:      vaddpd zmm28, zmm23, zmm28
// CHECK-MANUAL-NEXT:      vaddpd zmm29, zmm24, zmm29
// CHECK-MANUAL-NEXT:      vaddpd zmm30, zmm25, zmm30
// CHECK-MANUAL-NEXT:      vaddpd zmm31, zmm26, zmm31
// CHECK-MANUAL-NEXT:      vaddpd zmm27, zmm17, zmm27
// CHECK-MANUAL-NEXT:      vaddpd zmm28, zmm18, zmm28
// CHECK-MANUAL-NEXT:      vaddpd zmm29, zmm19, zmm29
// CHECK-MANUAL-NEXT:      vaddpd zmm30, zmm20, zmm30
// CHECK-MANUAL-NEXT:      vaddpd zmm31, zmm21, zmm31
// CHECK-MANUAL-NEXT:      vaddpd zmm27, zmm12, zmm27
// CHECK-MANUAL-NEXT:      vaddpd zmm28, zmm13, zmm28
// CHECK-MANUAL-NEXT:      vaddpd zmm29, zmm14, zmm29
// CHECK-MANUAL-NEXT:      vaddpd zmm30, zmm15, zmm30
// CHECK-MANUAL-NEXT:      vaddpd zmm31, zmm16, zmm31
// CHECK-MANUAL-NEXT:      sub rsi, 128
// CHECK-MANUAL-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-MANUAL-NEXT:      vmovupd [rdx+272] {k1}, zmm28
// CHECK-MANUAL-NEXT:      vmovupd [rdx+544] {k1}, zmm29
// CHECK-MANUAL-NEXT:      vmovupd [rdx+816] {k1}, zmm30
// CHECK-MANUAL-NEXT:      vmovupd [rdx+1088] {k1}, zmm31
// CHECK-MANUAL-NEXT:      add rdx, 16
// CHECK-MANUAL-NEXT:      sub rdi, 4336
// CHECK-MANUAL-NEXT:      cmp r10, 34
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_2]]
// CHECK-LIBXSMM-NEXT:     add rdx, 1088
// CHECK-COMPXSMM-NEXT:    sub rdi, 272
// CHECK-MANUAL-NEXT:      add rsi, 640
// CHECK-LIBXSMM-NEXT:     sub rdi, 272
// CHECK-COMPXSMM-NEXT:    add rdx, 1088
// CHECK-MANUAL-NEXT:      cmp r11, 5
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_0]]
// CHECK-MANUAL-NEXT:      mov rsp, rbp
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      pop r15
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      ret

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:      %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:      %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:      %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:      %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      x86.fallthrough ^bb1(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb1(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-NEXT:      x86.label "l33"
// CHECK-NEXT:      %17 = x86.ri.add %16, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %26 = x86.ri.add %25, 32 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %27 = x86.dm.vmovupd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %28 = x86.dm.vmovupd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %29 = x86.dm.vmovupd [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %30 = x86.dm.vmovupd [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %31 = x86.dm.vmovupd [%21 + 272] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %32 = x86.dm.vmovupd [%21 + 336] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %33 = x86.dm.vmovupd [%21 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %34 = x86.dm.vmovupd [%21 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %35 = x86.dm.vmovupd [%21 + 544] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %36 = x86.dm.vmovupd [%21 + 608] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %37 = x86.dm.vmovupd [%21 + 672] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %38 = x86.dm.vmovupd [%21 + 736] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %39 = x86.dm.vmovupd [%21 + 816] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %40 = x86.dm.vmovupd [%21 + 880] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %41 = x86.dm.vmovupd [%21 + 944] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %42 = x86.dm.vmovupd [%21 + 1008] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %43 = x86.dm.vmovupd [%21 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %44 = x86.dm.vmovupd [%21 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %45 = x86.dm.vmovupd [%21 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %46 = x86.dm.vmovupd [%21 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %47 = x86.dm.vmovupd [%19] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %48 = x86.dm.vmovupd [%19 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %49 = x86.dm.vmovupd [%19 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %50 = x86.dm.vmovupd [%19 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %51 = x86.dm.vbroadcastsd [%20] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %52 = x86.rss.vfmadd231pd %27, %47, %51 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %53 = x86.rss.vfmadd231pd %28, %48, %51 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %54 = x86.rss.vfmadd231pd %29, %49, %51 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %55 = x86.rss.vfmadd231pd %30, %50, %51 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %56 = x86.dm.vbroadcastsd [%20 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %57 = x86.rss.vfmadd231pd %31, %47, %56 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %58 = x86.rss.vfmadd231pd %32, %48, %56 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %59 = x86.rss.vfmadd231pd %33, %49, %56 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %60 = x86.rss.vfmadd231pd %34, %50, %56 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %61 = x86.dm.vbroadcastsd [%20 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %62 = x86.rss.vfmadd231pd %35, %47, %61 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %63 = x86.rss.vfmadd231pd %36, %48, %61 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %64 = x86.rss.vfmadd231pd %37, %49, %61 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %65 = x86.rss.vfmadd231pd %38, %50, %61 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %66 = x86.dm.vbroadcastsd [%20 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %67 = x86.rss.vfmadd231pd %39, %47, %66 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %68 = x86.rss.vfmadd231pd %40, %48, %66 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %69 = x86.rss.vfmadd231pd %41, %49, %66 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %70 = x86.rss.vfmadd231pd %42, %50, %66 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %71 = x86.dm.vbroadcastsd [%20 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %72 = x86.ri.add %20, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %73 = x86.ri.add %19, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %74 = x86.rss.vfmadd231pd %43, %47, %71 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %75 = x86.rss.vfmadd231pd %44, %48, %71 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %76 = x86.rss.vfmadd231pd %45, %49, %71 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %77 = x86.rss.vfmadd231pd %46, %50, %71 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %78 = x86.dm.vmovupd [%73] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %79 = x86.dm.vmovupd [%73 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %80 = x86.dm.vmovupd [%73 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %81 = x86.dm.vmovupd [%73 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %82 = x86.dm.vbroadcastsd [%72] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %83 = x86.rss.vfmadd231pd %52, %78, %82 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %84 = x86.rss.vfmadd231pd %53, %79, %82 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %85 = x86.rss.vfmadd231pd %54, %80, %82 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %86 = x86.rss.vfmadd231pd %55, %81, %82 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %87 = x86.dm.vbroadcastsd [%72 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %88 = x86.rss.vfmadd231pd %57, %78, %87 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %89 = x86.rss.vfmadd231pd %58, %79, %87 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %90 = x86.rss.vfmadd231pd %59, %80, %87 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231pd %60, %81, %87 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %92 = x86.dm.vbroadcastsd [%72 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %93 = x86.rss.vfmadd231pd %62, %78, %92 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %94 = x86.rss.vfmadd231pd %63, %79, %92 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %95 = x86.rss.vfmadd231pd %64, %80, %92 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %96 = x86.rss.vfmadd231pd %65, %81, %92 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %97 = x86.dm.vbroadcastsd [%72 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %98 = x86.rss.vfmadd231pd %67, %78, %97 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %99 = x86.rss.vfmadd231pd %68, %79, %97 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %100 = x86.rss.vfmadd231pd %69, %80, %97 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %101 = x86.rss.vfmadd231pd %70, %81, %97 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %102 = x86.dm.vbroadcastsd [%72 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %103 = x86.ri.add %72, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %104 = x86.ri.add %73, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %105 = x86.rss.vfmadd231pd %74, %78, %102 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %106 = x86.rss.vfmadd231pd %75, %79, %102 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %107 = x86.rss.vfmadd231pd %76, %80, %102 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %108 = x86.rss.vfmadd231pd %77, %81, %102 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %109 = x86.dm.vmovupd [%104] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %110 = x86.dm.vmovupd [%104 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %111 = x86.dm.vmovupd [%104 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %112 = x86.dm.vmovupd [%104 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %113 = x86.dm.vbroadcastsd [%103] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %114 = x86.rss.vfmadd231pd %83, %109, %113 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %115 = x86.rss.vfmadd231pd %84, %110, %113 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %116 = x86.rss.vfmadd231pd %85, %111, %113 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %117 = x86.rss.vfmadd231pd %86, %112, %113 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %118 = x86.dm.vbroadcastsd [%103 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %119 = x86.rss.vfmadd231pd %88, %109, %118 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %120 = x86.rss.vfmadd231pd %89, %110, %118 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %121 = x86.rss.vfmadd231pd %90, %111, %118 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %122 = x86.rss.vfmadd231pd %91, %112, %118 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %123 = x86.dm.vbroadcastsd [%103 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %124 = x86.rss.vfmadd231pd %93, %109, %123 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %125 = x86.rss.vfmadd231pd %94, %110, %123 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %126 = x86.rss.vfmadd231pd %95, %111, %123 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %127 = x86.rss.vfmadd231pd %96, %112, %123 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %128 = x86.dm.vbroadcastsd [%103 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %129 = x86.rss.vfmadd231pd %98, %109, %128 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %130 = x86.rss.vfmadd231pd %99, %110, %128 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %131 = x86.rss.vfmadd231pd %100, %111, %128 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %132 = x86.rss.vfmadd231pd %101, %112, %128 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %133 = x86.dm.vbroadcastsd [%103 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %134 = x86.ri.add %103, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %135 = x86.ri.add %104, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %136 = x86.rss.vfmadd231pd %105, %109, %133 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %137 = x86.rss.vfmadd231pd %106, %110, %133 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %138 = x86.rss.vfmadd231pd %107, %111, %133 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %139 = x86.rss.vfmadd231pd %108, %112, %133 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %140 = x86.dm.vmovupd [%135] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %141 = x86.dm.vmovupd [%135 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %142 = x86.dm.vmovupd [%135 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %143 = x86.dm.vmovupd [%135 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %144 = x86.dm.vbroadcastsd [%134] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %145 = x86.rss.vfmadd231pd %114, %140, %144 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %146 = x86.rss.vfmadd231pd %115, %141, %144 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %147 = x86.rss.vfmadd231pd %116, %142, %144 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %148 = x86.rss.vfmadd231pd %117, %143, %144 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %149 = x86.dm.vbroadcastsd [%134 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %150 = x86.rss.vfmadd231pd %119, %140, %149 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %151 = x86.rss.vfmadd231pd %120, %141, %149 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %152 = x86.rss.vfmadd231pd %121, %142, %149 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %153 = x86.rss.vfmadd231pd %122, %143, %149 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %154 = x86.dm.vbroadcastsd [%134 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %155 = x86.rss.vfmadd231pd %124, %140, %154 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %156 = x86.rss.vfmadd231pd %125, %141, %154 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %157 = x86.rss.vfmadd231pd %126, %142, %154 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %158 = x86.rss.vfmadd231pd %127, %143, %154 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %159 = x86.dm.vbroadcastsd [%134 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %160 = x86.rss.vfmadd231pd %129, %140, %159 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %161 = x86.rss.vfmadd231pd %130, %141, %159 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %162 = x86.rss.vfmadd231pd %131, %142, %159 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %163 = x86.rss.vfmadd231pd %132, %143, %159 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %164 = x86.dm.vbroadcastsd [%134 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %165 = x86.ri.add %134, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %166 = x86.ri.add %135, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %167 = x86.rss.vfmadd231pd %136, %140, %164 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %168 = x86.rss.vfmadd231pd %137, %141, %164 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %169 = x86.rss.vfmadd231pd %138, %142, %164 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %170 = x86.rss.vfmadd231pd %139, %143, %164 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %171 = x86.dm.vmovupd [%166] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %172 = x86.dm.vmovupd [%166 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %173 = x86.dm.vmovupd [%166 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %174 = x86.dm.vmovupd [%166 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %175 = x86.dm.vbroadcastsd [%165] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %176 = x86.rss.vfmadd231pd %145, %171, %175 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %177 = x86.rss.vfmadd231pd %146, %172, %175 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %178 = x86.rss.vfmadd231pd %147, %173, %175 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %179 = x86.rss.vfmadd231pd %148, %174, %175 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %180 = x86.dm.vbroadcastsd [%165 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %181 = x86.rss.vfmadd231pd %150, %171, %180 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %182 = x86.rss.vfmadd231pd %151, %172, %180 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %183 = x86.rss.vfmadd231pd %152, %173, %180 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %184 = x86.rss.vfmadd231pd %153, %174, %180 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %185 = x86.dm.vbroadcastsd [%165 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %186 = x86.rss.vfmadd231pd %155, %171, %185 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %187 = x86.rss.vfmadd231pd %156, %172, %185 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %188 = x86.rss.vfmadd231pd %157, %173, %185 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %189 = x86.rss.vfmadd231pd %158, %174, %185 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %190 = x86.dm.vbroadcastsd [%165 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %191 = x86.rss.vfmadd231pd %160, %171, %190 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %192 = x86.rss.vfmadd231pd %161, %172, %190 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %193 = x86.rss.vfmadd231pd %162, %173, %190 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %194 = x86.rss.vfmadd231pd %163, %174, %190 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %195 = x86.dm.vbroadcastsd [%165 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %196 = x86.ri.add %165, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %197 = x86.ri.add %166, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %198 = x86.rss.vfmadd231pd %167, %171, %195 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %199 = x86.rss.vfmadd231pd %168, %172, %195 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %200 = x86.rss.vfmadd231pd %169, %173, %195 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %201 = x86.rss.vfmadd231pd %170, %174, %195 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %202 = x86.dm.vmovupd [%197] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %203 = x86.dm.vmovupd [%197 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %204 = x86.dm.vmovupd [%197 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %205 = x86.dm.vmovupd [%197 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %206 = x86.dm.vbroadcastsd [%196] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %207 = x86.rss.vfmadd231pd %176, %202, %206 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %208 = x86.rss.vfmadd231pd %177, %203, %206 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %209 = x86.rss.vfmadd231pd %178, %204, %206 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %210 = x86.rss.vfmadd231pd %179, %205, %206 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %211 = x86.dm.vbroadcastsd [%196 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %212 = x86.rss.vfmadd231pd %181, %202, %211 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %213 = x86.rss.vfmadd231pd %182, %203, %211 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %214 = x86.rss.vfmadd231pd %183, %204, %211 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %215 = x86.rss.vfmadd231pd %184, %205, %211 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %216 = x86.dm.vbroadcastsd [%196 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %217 = x86.rss.vfmadd231pd %186, %202, %216 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %218 = x86.rss.vfmadd231pd %187, %203, %216 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %219 = x86.rss.vfmadd231pd %188, %204, %216 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %220 = x86.rss.vfmadd231pd %189, %205, %216 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %221 = x86.dm.vbroadcastsd [%196 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %222 = x86.rss.vfmadd231pd %191, %202, %221 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %223 = x86.rss.vfmadd231pd %192, %203, %221 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %224 = x86.rss.vfmadd231pd %193, %204, %221 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %225 = x86.rss.vfmadd231pd %194, %205, %221 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %226 = x86.dm.vbroadcastsd [%196 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %227 = x86.ri.add %196, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %228 = x86.ri.add %197, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %229 = x86.rss.vfmadd231pd %198, %202, %226 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %230 = x86.rss.vfmadd231pd %199, %203, %226 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %231 = x86.rss.vfmadd231pd %200, %204, %226 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %232 = x86.rss.vfmadd231pd %201, %205, %226 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %233 = x86.dm.vmovupd [%228] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %234 = x86.dm.vmovupd [%228 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %235 = x86.dm.vmovupd [%228 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %236 = x86.dm.vmovupd [%228 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %237 = x86.dm.vbroadcastsd [%227] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %238 = x86.rss.vfmadd231pd %207, %233, %237 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %239 = x86.rss.vfmadd231pd %208, %234, %237 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %240 = x86.rss.vfmadd231pd %209, %235, %237 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %241 = x86.rss.vfmadd231pd %210, %236, %237 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %242 = x86.dm.vbroadcastsd [%227 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %243 = x86.rss.vfmadd231pd %212, %233, %242 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %244 = x86.rss.vfmadd231pd %213, %234, %242 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %245 = x86.rss.vfmadd231pd %214, %235, %242 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %246 = x86.rss.vfmadd231pd %215, %236, %242 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %247 = x86.dm.vbroadcastsd [%227 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %248 = x86.rss.vfmadd231pd %217, %233, %247 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %249 = x86.rss.vfmadd231pd %218, %234, %247 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %250 = x86.rss.vfmadd231pd %219, %235, %247 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %251 = x86.rss.vfmadd231pd %220, %236, %247 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %252 = x86.dm.vbroadcastsd [%227 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %253 = x86.rss.vfmadd231pd %222, %233, %252 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %254 = x86.rss.vfmadd231pd %223, %234, %252 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %255 = x86.rss.vfmadd231pd %224, %235, %252 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %256 = x86.rss.vfmadd231pd %225, %236, %252 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %257 = x86.dm.vbroadcastsd [%227 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %258 = x86.ri.add %227, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %259 = x86.ri.add %228, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %260 = x86.rss.vfmadd231pd %229, %233, %257 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %261 = x86.rss.vfmadd231pd %230, %234, %257 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %262 = x86.rss.vfmadd231pd %231, %235, %257 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %263 = x86.rss.vfmadd231pd %232, %236, %257 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %264 = x86.dm.vmovupd [%259] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %265 = x86.dm.vmovupd [%259 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %266 = x86.dm.vmovupd [%259 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %267 = x86.dm.vmovupd [%259 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %268 = x86.dm.vbroadcastsd [%258] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %269 = x86.rss.vfmadd231pd %238, %264, %268 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %270 = x86.rss.vfmadd231pd %239, %265, %268 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %271 = x86.rss.vfmadd231pd %240, %266, %268 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %272 = x86.rss.vfmadd231pd %241, %267, %268 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %273 = x86.dm.vbroadcastsd [%258 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %274 = x86.rss.vfmadd231pd %243, %264, %273 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %275 = x86.rss.vfmadd231pd %244, %265, %273 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %276 = x86.rss.vfmadd231pd %245, %266, %273 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %277 = x86.rss.vfmadd231pd %246, %267, %273 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %278 = x86.dm.vbroadcastsd [%258 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %279 = x86.rss.vfmadd231pd %248, %264, %278 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %280 = x86.rss.vfmadd231pd %249, %265, %278 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %281 = x86.rss.vfmadd231pd %250, %266, %278 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %282 = x86.rss.vfmadd231pd %251, %267, %278 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %283 = x86.dm.vbroadcastsd [%258 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %284 = x86.rss.vfmadd231pd %253, %264, %283 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %285 = x86.rss.vfmadd231pd %254, %265, %283 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %286 = x86.rss.vfmadd231pd %255, %266, %283 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %287 = x86.rss.vfmadd231pd %256, %267, %283 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %288 = x86.dm.vbroadcastsd [%258 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %289 = x86.ri.add %258, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %290 = x86.ri.add %259, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %291 = x86.rss.vfmadd231pd %260, %264, %288 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %292 = x86.rss.vfmadd231pd %261, %265, %288 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %293 = x86.rss.vfmadd231pd %262, %266, %288 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %294 = x86.rss.vfmadd231pd %263, %267, %288 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %295 = x86.dm.vmovupd [%290] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %296 = x86.dm.vmovupd [%290 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %297 = x86.dm.vmovupd [%290 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %298 = x86.dm.vmovupd [%290 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %299 = x86.dm.vbroadcastsd [%289] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %300 = x86.rss.vfmadd231pd %269, %295, %299 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %301 = x86.rss.vfmadd231pd %270, %296, %299 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %302 = x86.rss.vfmadd231pd %271, %297, %299 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %303 = x86.rss.vfmadd231pd %272, %298, %299 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %304 = x86.dm.vbroadcastsd [%289 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %305 = x86.rss.vfmadd231pd %274, %295, %304 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %306 = x86.rss.vfmadd231pd %275, %296, %304 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %307 = x86.rss.vfmadd231pd %276, %297, %304 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %308 = x86.rss.vfmadd231pd %277, %298, %304 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %309 = x86.dm.vbroadcastsd [%289 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %310 = x86.rss.vfmadd231pd %279, %295, %309 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %311 = x86.rss.vfmadd231pd %280, %296, %309 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %312 = x86.rss.vfmadd231pd %281, %297, %309 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %313 = x86.rss.vfmadd231pd %282, %298, %309 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %314 = x86.dm.vbroadcastsd [%289 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %315 = x86.rss.vfmadd231pd %284, %295, %314 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %316 = x86.rss.vfmadd231pd %285, %296, %314 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %317 = x86.rss.vfmadd231pd %286, %297, %314 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %318 = x86.rss.vfmadd231pd %287, %298, %314 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %319 = x86.dm.vbroadcastsd [%289 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %320 = x86.ri.add %289, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %321 = x86.ri.add %290, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %322 = x86.rss.vfmadd231pd %291, %295, %319 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %323 = x86.rss.vfmadd231pd %292, %296, %319 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %324 = x86.rss.vfmadd231pd %293, %297, %319 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %325 = x86.rss.vfmadd231pd %294, %298, %319 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %326 = x86.dm.vmovupd [%321] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %327 = x86.dm.vmovupd [%321 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %328 = x86.dm.vmovupd [%321 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %329 = x86.dm.vmovupd [%321 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %330 = x86.dm.vbroadcastsd [%320] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %331 = x86.rss.vfmadd231pd %300, %326, %330 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %332 = x86.rss.vfmadd231pd %301, %327, %330 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %333 = x86.rss.vfmadd231pd %302, %328, %330 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %334 = x86.rss.vfmadd231pd %303, %329, %330 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %335 = x86.dm.vbroadcastsd [%320 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %336 = x86.rss.vfmadd231pd %305, %326, %335 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %337 = x86.rss.vfmadd231pd %306, %327, %335 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %338 = x86.rss.vfmadd231pd %307, %328, %335 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %339 = x86.rss.vfmadd231pd %308, %329, %335 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %340 = x86.dm.vbroadcastsd [%320 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %341 = x86.rss.vfmadd231pd %310, %326, %340 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %342 = x86.rss.vfmadd231pd %311, %327, %340 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %343 = x86.rss.vfmadd231pd %312, %328, %340 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %344 = x86.rss.vfmadd231pd %313, %329, %340 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %345 = x86.dm.vbroadcastsd [%320 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %346 = x86.rss.vfmadd231pd %315, %326, %345 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %347 = x86.rss.vfmadd231pd %316, %327, %345 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %348 = x86.rss.vfmadd231pd %317, %328, %345 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %349 = x86.rss.vfmadd231pd %318, %329, %345 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %350 = x86.dm.vbroadcastsd [%320 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %351 = x86.ri.add %320, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %352 = x86.ri.add %321, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %353 = x86.rss.vfmadd231pd %322, %326, %350 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %354 = x86.rss.vfmadd231pd %323, %327, %350 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %355 = x86.rss.vfmadd231pd %324, %328, %350 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %356 = x86.rss.vfmadd231pd %325, %329, %350 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %357 = x86.dm.vmovupd [%352] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %358 = x86.dm.vmovupd [%352 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %359 = x86.dm.vmovupd [%352 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %360 = x86.dm.vmovupd [%352 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %361 = x86.dm.vbroadcastsd [%351] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %362 = x86.rss.vfmadd231pd %331, %357, %361 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %363 = x86.rss.vfmadd231pd %332, %358, %361 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %364 = x86.rss.vfmadd231pd %333, %359, %361 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %365 = x86.rss.vfmadd231pd %334, %360, %361 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %366 = x86.dm.vbroadcastsd [%351 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %367 = x86.rss.vfmadd231pd %336, %357, %366 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %368 = x86.rss.vfmadd231pd %337, %358, %366 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %369 = x86.rss.vfmadd231pd %338, %359, %366 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %370 = x86.rss.vfmadd231pd %339, %360, %366 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %371 = x86.dm.vbroadcastsd [%351 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %372 = x86.rss.vfmadd231pd %341, %357, %371 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %373 = x86.rss.vfmadd231pd %342, %358, %371 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %374 = x86.rss.vfmadd231pd %343, %359, %371 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %375 = x86.rss.vfmadd231pd %344, %360, %371 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %376 = x86.dm.vbroadcastsd [%351 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %377 = x86.rss.vfmadd231pd %346, %357, %376 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %378 = x86.rss.vfmadd231pd %347, %358, %376 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %379 = x86.rss.vfmadd231pd %348, %359, %376 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %380 = x86.rss.vfmadd231pd %349, %360, %376 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %381 = x86.dm.vbroadcastsd [%351 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %382 = x86.ri.add %351, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %383 = x86.ri.add %352, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %384 = x86.rss.vfmadd231pd %353, %357, %381 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %385 = x86.rss.vfmadd231pd %354, %358, %381 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %386 = x86.rss.vfmadd231pd %355, %359, %381 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %387 = x86.rss.vfmadd231pd %356, %360, %381 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %388 = x86.dm.vmovupd [%383] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %389 = x86.dm.vmovupd [%383 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %390 = x86.dm.vmovupd [%383 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %391 = x86.dm.vmovupd [%383 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %392 = x86.dm.vbroadcastsd [%382] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %393 = x86.rss.vfmadd231pd %362, %388, %392 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %394 = x86.rss.vfmadd231pd %363, %389, %392 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %395 = x86.rss.vfmadd231pd %364, %390, %392 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %396 = x86.rss.vfmadd231pd %365, %391, %392 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %397 = x86.dm.vbroadcastsd [%382 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %398 = x86.rss.vfmadd231pd %367, %388, %397 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %399 = x86.rss.vfmadd231pd %368, %389, %397 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %400 = x86.rss.vfmadd231pd %369, %390, %397 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %401 = x86.rss.vfmadd231pd %370, %391, %397 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %402 = x86.dm.vbroadcastsd [%382 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %403 = x86.rss.vfmadd231pd %372, %388, %402 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %404 = x86.rss.vfmadd231pd %373, %389, %402 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %405 = x86.rss.vfmadd231pd %374, %390, %402 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %406 = x86.rss.vfmadd231pd %375, %391, %402 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %407 = x86.dm.vbroadcastsd [%382 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %408 = x86.rss.vfmadd231pd %377, %388, %407 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %409 = x86.rss.vfmadd231pd %378, %389, %407 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %410 = x86.rss.vfmadd231pd %379, %390, %407 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %411 = x86.rss.vfmadd231pd %380, %391, %407 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %412 = x86.dm.vbroadcastsd [%382 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %413 = x86.ri.add %382, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %414 = x86.ri.add %383, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %415 = x86.rss.vfmadd231pd %384, %388, %412 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %416 = x86.rss.vfmadd231pd %385, %389, %412 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %417 = x86.rss.vfmadd231pd %386, %390, %412 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %418 = x86.rss.vfmadd231pd %387, %391, %412 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %419 = x86.dm.vmovupd [%414] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %420 = x86.dm.vmovupd [%414 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %421 = x86.dm.vmovupd [%414 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %422 = x86.dm.vmovupd [%414 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %423 = x86.dm.vbroadcastsd [%413] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %424 = x86.rss.vfmadd231pd %393, %419, %423 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %425 = x86.rss.vfmadd231pd %394, %420, %423 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %426 = x86.rss.vfmadd231pd %395, %421, %423 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %427 = x86.rss.vfmadd231pd %396, %422, %423 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %428 = x86.dm.vbroadcastsd [%413 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %429 = x86.rss.vfmadd231pd %398, %419, %428 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %430 = x86.rss.vfmadd231pd %399, %420, %428 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %431 = x86.rss.vfmadd231pd %400, %421, %428 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %432 = x86.rss.vfmadd231pd %401, %422, %428 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %433 = x86.dm.vbroadcastsd [%413 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %434 = x86.rss.vfmadd231pd %403, %419, %433 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %435 = x86.rss.vfmadd231pd %404, %420, %433 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %436 = x86.rss.vfmadd231pd %405, %421, %433 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %437 = x86.rss.vfmadd231pd %406, %422, %433 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %438 = x86.dm.vbroadcastsd [%413 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %439 = x86.rss.vfmadd231pd %408, %419, %438 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %440 = x86.rss.vfmadd231pd %409, %420, %438 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %441 = x86.rss.vfmadd231pd %410, %421, %438 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %442 = x86.rss.vfmadd231pd %411, %422, %438 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %443 = x86.dm.vbroadcastsd [%413 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %444 = x86.ri.add %413, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %445 = x86.ri.add %414, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %446 = x86.rss.vfmadd231pd %415, %419, %443 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %447 = x86.rss.vfmadd231pd %416, %420, %443 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %448 = x86.rss.vfmadd231pd %417, %421, %443 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %449 = x86.rss.vfmadd231pd %418, %422, %443 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %450 = x86.dm.vmovupd [%445] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %451 = x86.dm.vmovupd [%445 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %452 = x86.dm.vmovupd [%445 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %453 = x86.dm.vmovupd [%445 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %454 = x86.dm.vbroadcastsd [%444] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %455 = x86.rss.vfmadd231pd %424, %450, %454 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %456 = x86.rss.vfmadd231pd %425, %451, %454 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %457 = x86.rss.vfmadd231pd %426, %452, %454 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %458 = x86.rss.vfmadd231pd %427, %453, %454 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %459 = x86.dm.vbroadcastsd [%444 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %460 = x86.rss.vfmadd231pd %429, %450, %459 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %461 = x86.rss.vfmadd231pd %430, %451, %459 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %462 = x86.rss.vfmadd231pd %431, %452, %459 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %463 = x86.rss.vfmadd231pd %432, %453, %459 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %464 = x86.dm.vbroadcastsd [%444 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %465 = x86.rss.vfmadd231pd %434, %450, %464 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %466 = x86.rss.vfmadd231pd %435, %451, %464 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %467 = x86.rss.vfmadd231pd %436, %452, %464 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %468 = x86.rss.vfmadd231pd %437, %453, %464 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %469 = x86.dm.vbroadcastsd [%444 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %470 = x86.rss.vfmadd231pd %439, %450, %469 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %471 = x86.rss.vfmadd231pd %440, %451, %469 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %472 = x86.rss.vfmadd231pd %441, %452, %469 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %473 = x86.rss.vfmadd231pd %442, %453, %469 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %474 = x86.dm.vbroadcastsd [%444 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %475 = x86.ri.add %444, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %476 = x86.ri.add %445, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %477 = x86.rss.vfmadd231pd %446, %450, %474 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %478 = x86.rss.vfmadd231pd %447, %451, %474 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %479 = x86.rss.vfmadd231pd %448, %452, %474 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %480 = x86.rss.vfmadd231pd %449, %453, %474 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %481 = x86.dm.vmovupd [%476] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %482 = x86.dm.vmovupd [%476 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %483 = x86.dm.vmovupd [%476 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %484 = x86.dm.vmovupd [%476 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %485 = x86.dm.vbroadcastsd [%475] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %486 = x86.rss.vfmadd231pd %455, %481, %485 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %487 = x86.rss.vfmadd231pd %456, %482, %485 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %488 = x86.rss.vfmadd231pd %457, %483, %485 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %489 = x86.rss.vfmadd231pd %458, %484, %485 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %490 = x86.dm.vbroadcastsd [%475 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %491 = x86.rss.vfmadd231pd %460, %481, %490 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %492 = x86.rss.vfmadd231pd %461, %482, %490 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %493 = x86.rss.vfmadd231pd %462, %483, %490 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %494 = x86.rss.vfmadd231pd %463, %484, %490 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %495 = x86.dm.vbroadcastsd [%475 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %496 = x86.rss.vfmadd231pd %465, %481, %495 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %497 = x86.rss.vfmadd231pd %466, %482, %495 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %498 = x86.rss.vfmadd231pd %467, %483, %495 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %499 = x86.rss.vfmadd231pd %468, %484, %495 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %500 = x86.dm.vbroadcastsd [%475 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %501 = x86.rss.vfmadd231pd %470, %481, %500 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %502 = x86.rss.vfmadd231pd %471, %482, %500 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %503 = x86.rss.vfmadd231pd %472, %483, %500 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %504 = x86.rss.vfmadd231pd %473, %484, %500 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %505 = x86.dm.vbroadcastsd [%475 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %506 = x86.ri.add %475, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %507 = x86.ri.add %476, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %508 = x86.rss.vfmadd231pd %477, %481, %505 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %509 = x86.rss.vfmadd231pd %478, %482, %505 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %510 = x86.rss.vfmadd231pd %479, %483, %505 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %511 = x86.rss.vfmadd231pd %480, %484, %505 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %512 = x86.dm.vmovupd [%507] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %513 = x86.dm.vmovupd [%507 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %514 = x86.dm.vmovupd [%507 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %515 = x86.dm.vmovupd [%507 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %516 = x86.dm.vbroadcastsd [%506] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %517 = x86.rss.vfmadd231pd %486, %512, %516 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %518 = x86.rss.vfmadd231pd %487, %513, %516 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %519 = x86.rss.vfmadd231pd %488, %514, %516 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %520 = x86.rss.vfmadd231pd %489, %515, %516 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %521 = x86.dm.vbroadcastsd [%506 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %522 = x86.rss.vfmadd231pd %491, %512, %521 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %523 = x86.rss.vfmadd231pd %492, %513, %521 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %524 = x86.rss.vfmadd231pd %493, %514, %521 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %525 = x86.rss.vfmadd231pd %494, %515, %521 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %526 = x86.dm.vbroadcastsd [%506 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %527 = x86.rss.vfmadd231pd %496, %512, %526 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %528 = x86.rss.vfmadd231pd %497, %513, %526 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %529 = x86.rss.vfmadd231pd %498, %514, %526 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %530 = x86.rss.vfmadd231pd %499, %515, %526 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %531 = x86.dm.vbroadcastsd [%506 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %532 = x86.rss.vfmadd231pd %501, %512, %531 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %533 = x86.rss.vfmadd231pd %502, %513, %531 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %534 = x86.rss.vfmadd231pd %503, %514, %531 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %535 = x86.rss.vfmadd231pd %504, %515, %531 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %536 = x86.dm.vbroadcastsd [%506 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %537 = x86.ri.add %506, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %538 = x86.ri.add %507, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %539 = x86.rss.vfmadd231pd %508, %512, %536 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %540 = x86.rss.vfmadd231pd %509, %513, %536 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %541 = x86.rss.vfmadd231pd %510, %514, %536 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %542 = x86.rss.vfmadd231pd %511, %515, %536 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %543 = x86.ri.sub %537, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovupd [%21], %517 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 64], %518 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 128], %519 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 192], %520 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 272], %522 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 336], %523 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 400], %524 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 464], %525 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 544], %527 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 608], %528 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 672], %529 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 736], %530 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 816], %532 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 880], %533 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 944], %534 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 1008], %535 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 1088], %539 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 1152], %540 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 1216], %541 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%21 + 1280], %542 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %544 = x86.ri.add %21, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %545 = x86.ri.sub %538, 4096 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %546 = x86.si.cmp %26, 32 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %546 : !x86.rflags<rflags>, ^bb2(%545 : !x86.reg64<rdi>, %543 : !x86.reg64<rsi>, %544 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>), ^bb3(%545 : !x86.reg64<rdi>, %543 : !x86.reg64<rsi>, %544 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb3(%547: !x86.reg64<rdi>, %548: !x86.reg64<rsi>, %549: !x86.reg64<rdx>, %550: !x86.reg64<rbp>, %551: !x86.reg64<rsp>, %552: !x86.reg64<r11>, %553: !x86.reg64<r10>):
// CHECK-NEXT:      %554 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %555 = x86.ks.kmovb %554 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %556 = x86.di.mov 32 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb4(%547 : !x86.reg64<rdi>, %548 : !x86.reg64<rsi>, %549 : !x86.reg64<rdx>, %550 : !x86.reg64<rbp>, %551 : !x86.reg64<rsp>, %552 : !x86.reg64<r11>, %556 : !x86.reg64<r10>, %555 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb4(%557: !x86.reg64<rdi>, %558: !x86.reg64<rsi>, %559: !x86.reg64<rdx>, %560: !x86.reg64<rbp>, %561: !x86.reg64<rsp>, %562: !x86.reg64<r11>, %563: !x86.reg64<r10>, %564: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      x86.label "l35"
// CHECK-NEXT:      %565 = x86.ri.add %563, 2 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %566 = x86.dmk.vmovupd[%559], %564 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %567 = x86.dmk.vmovupd[%559 + 272], %564 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %568 = x86.dmk.vmovupd[%559 + 544], %564 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %569 = x86.dmk.vmovupd[%559 + 816], %564 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %570 = x86.dmk.vmovupd[%559 + 1088], %564 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %571 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-NEXT:      %572 = x86.dss.vpxord %571, %571 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %573 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-NEXT:      %574 = x86.dss.vpxord %573, %573 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %575 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-NEXT:      %576 = x86.dss.vpxord %575, %575 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %577 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-NEXT:      %578 = x86.dss.vpxord %577, %577 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %579 = x86.get_avx_register : !x86.avx512reg<zmm26>
// CHECK-NEXT:      %580 = x86.dss.vpxord %579, %579 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %581 = x86.get_avx_register : !x86.avx512reg<zmm17>
// CHECK-NEXT:      %582 = x86.dss.vpxord %581, %581 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm17>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %583 = x86.get_avx_register : !x86.avx512reg<zmm18>
// CHECK-NEXT:      %584 = x86.dss.vpxord %583, %583 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm18>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %585 = x86.get_avx_register : !x86.avx512reg<zmm19>
// CHECK-NEXT:      %586 = x86.dss.vpxord %585, %585 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm19>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %587 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-NEXT:      %588 = x86.dss.vpxord %587, %587 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %589 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-NEXT:      %590 = x86.dss.vpxord %589, %589 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %591 = x86.get_avx_register : !x86.avx512reg<zmm12>
// CHECK-NEXT:      %592 = x86.dss.vpxord %591, %591 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm12>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %593 = x86.get_avx_register : !x86.avx512reg<zmm13>
// CHECK-NEXT:      %594 = x86.dss.vpxord %593, %593 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm13>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %595 = x86.get_avx_register : !x86.avx512reg<zmm14>
// CHECK-NEXT:      %596 = x86.dss.vpxord %595, %595 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm14>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %597 = x86.get_avx_register : !x86.avx512reg<zmm15>
// CHECK-NEXT:      %598 = x86.dss.vpxord %597, %597 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm15>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %599 = x86.get_avx_register : !x86.avx512reg<zmm16>
// CHECK-NEXT:      %600 = x86.dss.vpxord %599, %599 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm16>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %601 = x86.dmk.vmovupd[%557], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %602 = x86.dmk.vmovupd[%557 + 272], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %603 = x86.rsm.vfmadd231pd %566, %601, [%558] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %604 = x86.rsm.vfmadd231pd %567, %601, [%558 + 128] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %605 = x86.rsm.vfmadd231pd %568, %601, [%558 + 256] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %606 = x86.rsm.vfmadd231pd %569, %601, [%558 + 384] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %607 = x86.rsm.vfmadd231pd %570, %601, [%558 + 512] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %608 = x86.dmk.vmovupd[%557 + 544], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %609 = x86.rsm.vfmadd231pd %572, %602, [%558 + 8] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %610 = x86.rsm.vfmadd231pd %574, %602, [%558 + 136] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %611 = x86.rsm.vfmadd231pd %576, %602, [%558 + 264] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %612 = x86.rsm.vfmadd231pd %578, %602, [%558 + 392] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %613 = x86.rsm.vfmadd231pd %580, %602, [%558 + 520] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %614 = x86.dmk.vmovupd[%557 + 816], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %615 = x86.rsm.vfmadd231pd %582, %608, [%558 + 16] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %616 = x86.rsm.vfmadd231pd %584, %608, [%558 + 144] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %617 = x86.rsm.vfmadd231pd %586, %608, [%558 + 272] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %618 = x86.rsm.vfmadd231pd %588, %608, [%558 + 400] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %619 = x86.rsm.vfmadd231pd %590, %608, [%558 + 528] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %620 = x86.dmk.vmovupd[%557 + 1088], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %621 = x86.rsm.vfmadd231pd %592, %614, [%558 + 24] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %622 = x86.rsm.vfmadd231pd %594, %614, [%558 + 152] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %623 = x86.rsm.vfmadd231pd %596, %614, [%558 + 280] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %624 = x86.rsm.vfmadd231pd %598, %614, [%558 + 408] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %625 = x86.rsm.vfmadd231pd %600, %614, [%558 + 536] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %626 = x86.dmk.vmovupd[%557 + 1360], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %627 = x86.rsm.vfmadd231pd %603, %620, [%558 + 32] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %628 = x86.rsm.vfmadd231pd %604, %620, [%558 + 160] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %629 = x86.rsm.vfmadd231pd %605, %620, [%558 + 288] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %630 = x86.rsm.vfmadd231pd %606, %620, [%558 + 416] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %631 = x86.rsm.vfmadd231pd %607, %620, [%558 + 544] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %632 = x86.dmk.vmovupd[%557 + 1632], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %633 = x86.rsm.vfmadd231pd %609, %626, [%558 + 40] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %634 = x86.rsm.vfmadd231pd %610, %626, [%558 + 168] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %635 = x86.rsm.vfmadd231pd %611, %626, [%558 + 296] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %636 = x86.rsm.vfmadd231pd %612, %626, [%558 + 424] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %637 = x86.rsm.vfmadd231pd %613, %626, [%558 + 552] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %638 = x86.dmk.vmovupd[%557 + 1904], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %639 = x86.rsm.vfmadd231pd %615, %632, [%558 + 48] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %640 = x86.rsm.vfmadd231pd %616, %632, [%558 + 176] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %641 = x86.rsm.vfmadd231pd %617, %632, [%558 + 304] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %642 = x86.rsm.vfmadd231pd %618, %632, [%558 + 432] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %643 = x86.rsm.vfmadd231pd %619, %632, [%558 + 560] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %644 = x86.dmk.vmovupd[%557 + 2176], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %645 = x86.rsm.vfmadd231pd %621, %638, [%558 + 56] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %646 = x86.rsm.vfmadd231pd %622, %638, [%558 + 184] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %647 = x86.rsm.vfmadd231pd %623, %638, [%558 + 312] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %648 = x86.rsm.vfmadd231pd %624, %638, [%558 + 440] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %649 = x86.rsm.vfmadd231pd %625, %638, [%558 + 568] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %650 = x86.dmk.vmovupd[%557 + 2448], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %651 = x86.rsm.vfmadd231pd %627, %644, [%558 + 64] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %652 = x86.rsm.vfmadd231pd %628, %644, [%558 + 192] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %653 = x86.rsm.vfmadd231pd %629, %644, [%558 + 320] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %654 = x86.rsm.vfmadd231pd %630, %644, [%558 + 448] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %655 = x86.rsm.vfmadd231pd %631, %644, [%558 + 576] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %656 = x86.dmk.vmovupd[%557 + 2720], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %657 = x86.rsm.vfmadd231pd %633, %650, [%558 + 72] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %658 = x86.rsm.vfmadd231pd %634, %650, [%558 + 200] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %659 = x86.rsm.vfmadd231pd %635, %650, [%558 + 328] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %660 = x86.rsm.vfmadd231pd %636, %650, [%558 + 456] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %661 = x86.rsm.vfmadd231pd %637, %650, [%558 + 584] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %662 = x86.dmk.vmovupd[%557 + 2992], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %663 = x86.rsm.vfmadd231pd %639, %656, [%558 + 80] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %664 = x86.rsm.vfmadd231pd %640, %656, [%558 + 208] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %665 = x86.rsm.vfmadd231pd %641, %656, [%558 + 336] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %666 = x86.rsm.vfmadd231pd %642, %656, [%558 + 464] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %667 = x86.rsm.vfmadd231pd %643, %656, [%558 + 592] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %668 = x86.dmk.vmovupd[%557 + 3264], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %669 = x86.rsm.vfmadd231pd %645, %662, [%558 + 88] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %670 = x86.rsm.vfmadd231pd %646, %662, [%558 + 216] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %671 = x86.rsm.vfmadd231pd %647, %662, [%558 + 344] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %672 = x86.rsm.vfmadd231pd %648, %662, [%558 + 472] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %673 = x86.rsm.vfmadd231pd %649, %662, [%558 + 600] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %674 = x86.dmk.vmovupd[%557 + 3536], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %675 = x86.rsm.vfmadd231pd %651, %668, [%558 + 96] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %676 = x86.rsm.vfmadd231pd %652, %668, [%558 + 224] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %677 = x86.rsm.vfmadd231pd %653, %668, [%558 + 352] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %678 = x86.rsm.vfmadd231pd %654, %668, [%558 + 480] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %679 = x86.rsm.vfmadd231pd %655, %668, [%558 + 608] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %680 = x86.dmk.vmovupd[%557 + 3808], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %681 = x86.rsm.vfmadd231pd %657, %674, [%558 + 104] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %682 = x86.rsm.vfmadd231pd %658, %674, [%558 + 232] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %683 = x86.rsm.vfmadd231pd %659, %674, [%558 + 360] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %684 = x86.rsm.vfmadd231pd %660, %674, [%558 + 488] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %685 = x86.rsm.vfmadd231pd %661, %674, [%558 + 616] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %686 = x86.dmk.vmovupd[%557 + 4080], %564 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %687 = x86.rsm.vfmadd231pd %663, %680, [%558 + 112] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %688 = x86.rsm.vfmadd231pd %664, %680, [%558 + 240] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %689 = x86.rsm.vfmadd231pd %665, %680, [%558 + 368] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %690 = x86.rsm.vfmadd231pd %666, %680, [%558 + 496] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %691 = x86.rsm.vfmadd231pd %667, %680, [%558 + 624] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %692 = x86.ri.add %557, 4352 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %693 = x86.rsm.vfmadd231pd %669, %686, [%558 + 120] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %694 = x86.rsm.vfmadd231pd %670, %686, [%558 + 248] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %695 = x86.rsm.vfmadd231pd %671, %686, [%558 + 376] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %696 = x86.rsm.vfmadd231pd %672, %686, [%558 + 504] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %697 = x86.rsm.vfmadd231pd %673, %686, [%558 + 632] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %698 = x86.ri.add %558, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %699 = x86.dss.vaddpd %681, %675 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %700 = x86.dss.vaddpd %682, %676 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %701 = x86.dss.vaddpd %683, %677 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %702 = x86.dss.vaddpd %684, %678 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %703 = x86.dss.vaddpd %685, %679 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %704 = x86.dss.vaddpd %687, %699 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %705 = x86.dss.vaddpd %688, %700 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %706 = x86.dss.vaddpd %689, %701 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %707 = x86.dss.vaddpd %690, %702 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %708 = x86.dss.vaddpd %691, %703 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %709 = x86.dss.vaddpd %693, %704 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %710 = x86.dss.vaddpd %694, %705 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %711 = x86.dss.vaddpd %695, %706 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %712 = x86.dss.vaddpd %696, %707 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %713 = x86.dss.vaddpd %697, %708 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %714 = x86.ri.sub %698, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.msk.vmovupd[%559], %709, %564 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%559 + 272], %710, %564 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%559 + 544], %711, %564 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%559 + 816], %712, %564 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%559 + 1088], %713, %564 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %715 = x86.ri.add %559, 16 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %716 = x86.ri.sub %692, 4336 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %717 = x86.si.cmp %565, 34 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %717 : !x86.rflags<rflags>, ^bb4(%716 : !x86.reg64<rdi>, %714 : !x86.reg64<rsi>, %715 : !x86.reg64<rdx>, %560 : !x86.reg64<rbp>, %561 : !x86.reg64<rsp>, %562 : !x86.reg64<r11>, %565 : !x86.reg64<r10>, %564 : !x86.avx512maskreg<k1>), ^bb5(%716 : !x86.reg64<rdi>, %714 : !x86.reg64<rsi>, %715 : !x86.reg64<rdx>, %560 : !x86.reg64<rbp>, %561 : !x86.reg64<rsp>, %562 : !x86.reg64<r11>, %565 : !x86.reg64<r10>, %564 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb5(%718: !x86.reg64<rdi>, %719: !x86.reg64<rsi>, %720: !x86.reg64<rdx>, %721: !x86.reg64<rbp>, %722: !x86.reg64<rsp>, %723: !x86.reg64<r11>, %724: !x86.reg64<r10>, %725: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      %726 = x86.ri.add %720, 1088 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %727 = x86.ri.add %719, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %728 = x86.ri.sub %718, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %729 = x86.si.cmp %723, 5 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %729 : !x86.rflags<rflags>, ^bb1(%728 : !x86.reg64<rdi>, %727 : !x86.reg64<rsi>, %726 : !x86.reg64<rdx>, %721 : !x86.reg64<rbp>, %722 : !x86.reg64<rsp>, %723 : !x86.reg64<r11>), ^bb6(%728 : !x86.reg64<rdi>, %727 : !x86.reg64<rsi>, %726 : !x86.reg64<rdx>, %721 : !x86.reg64<rbp>, %722 : !x86.reg64<rsp>, %723 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb6(%730: !x86.reg64<rdi>, %731: !x86.reg64<rsi>, %732: !x86.reg64<rdx>, %733: !x86.reg64<rbp>, %734: !x86.reg64<rsp>, %735: !x86.reg64<r11>):
// CHECK-NEXT:      %736 = x86.ds.mov %733 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %737, %738 = x86.d.pop %736 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
