// RUN: libxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

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
