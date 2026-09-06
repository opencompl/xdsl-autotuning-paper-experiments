// RUN: libxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-LIBXSMM-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push r15
// CHECK-LIBXSMM-NEXT:      push rbp
// CHECK-LIBXSMM-NEXT:      mov rbp, rsp
// CHECK-LIBXSMM-NEXT:      sub rsp, 192
// CHECK-LIBXSMM-NEXT:      mov r10, -64
// CHECK-LIBXSMM-NEXT:      and rsp, r10
// CHECK-LIBXSMM-NEXT:      mov r11, 0
// CHECK-LIBXSMM-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-LIBXSMM-NEXT:      add r11, 5
// CHECK-MANUAL-NEXT:      mov r15, 3
// CHECK-MANUAL-NEXT:      kmovb k1, r15d
// CHECK-LIBXSMM-NEXT:      mov r10, 0
// CHECK-LIBXSMM-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-LIBXSMM-NEXT:      add r10, 10
// CHECK-MANUAL-NEXT:      vmovupd zmm22, [rdx]
// CHECK-MANUAL-NEXT:      vmovupd zmm23 {k1}{z}, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovupd zmm24, [rdx+80]
// CHECK-MANUAL-NEXT:      vmovupd zmm25 {k1}{z}, [rdx+144]
// CHECK-MANUAL-NEXT:      vmovupd zmm26, [rdx+160]
// CHECK-MANUAL-NEXT:      vmovupd zmm27 {k1}{z}, [rdx+224]
// CHECK-MANUAL-NEXT:      vmovupd zmm28, [rdx+240]
// CHECK-MANUAL-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+304]
// CHECK-MANUAL-NEXT:      vmovupd zmm30, [rdx+320]
// CHECK-MANUAL-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+384]
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovupd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 80
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-LIBXSMM-NEXT:     sub rsi, 128
// CHECK-MANUAL-NEXT:      vmovupd [rdx], zmm22
// CHECK-MANUAL-NEXT:      vmovupd [rdx+64] {k1}, zmm23
// CHECK-MANUAL-NEXT:      vmovupd [rdx+80], zmm24
// CHECK-MANUAL-NEXT:      vmovupd [rdx+144] {k1}, zmm25
// CHECK-MANUAL-NEXT:      vmovupd [rdx+160], zmm26
// CHECK-MANUAL-NEXT:      vmovupd [rdx+224] {k1}, zmm27
// CHECK-MANUAL-NEXT:      vmovupd [rdx+240], zmm28
// CHECK-MANUAL-NEXT:      vmovupd [rdx+304] {k1}, zmm29
// CHECK-MANUAL-NEXT:      vmovupd [rdx+320], zmm30
// CHECK-MANUAL-NEXT:      vmovupd [rdx+384] {k1}, zmm31
// CHECK-COMPXSMM-NEXT:    sub rdi, 1200
// CHECK-COMPXSMM-NEXT:    sub rsi, 128
// CHECK-MANUAL-NEXT:      add rdx, 80
// CHECK-LIBXSMM-NEXT:     sub rdi, 1200
// CHECK-LIBXSMM-NEXT:      cmp r10, 10
// CHECK-LIBXSMM-NEXT:      jl [[SCF_M_BODY]]
// CHECK-LIBXSMM-NEXT:     add rdx, 320
// CHECK-COMPXSMM-NEXT:    sub rdi, 80
// CHECK-MANUAL-NEXT:      add rsi, 640
// CHECK-LIBXSMM-NEXT:     sub rdi, 80
// CHECK-COMPXSMM-NEXT:    add rdx, 320
// CHECK-LIBXSMM-NEXT:      cmp r11, 5
// CHECK-LIBXSMM-NEXT:      jl [[SCF_N_BODY]]
// CHECK-LIBXSMM-NEXT:      mov rsp, rbp
// CHECK-LIBXSMM-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      pop r15
// CHECK-LIBXSMM-NEXT:      pop rbp
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
// CHECK-NEXT:      %18 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %19 = x86.ks.kmovb %18 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %20 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %20 : !x86.reg64<r10>, %19 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb2(%21: !x86.reg64<rdi>, %22: !x86.reg64<rsi>, %23: !x86.reg64<rdx>, %24: !x86.reg64<rbp>, %25: !x86.reg64<rsp>, %26: !x86.reg64<r11>, %27: !x86.reg64<r10>, %28: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %29 = x86.ri.add %27, 10 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %30 = x86.dm.vmovupd [%23] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %31 = x86.dmk.vmovupd[%23 + 64], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %32 = x86.dm.vmovupd [%23 + 80] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %33 = x86.dmk.vmovupd[%23 + 144], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %34 = x86.dm.vmovupd [%23 + 160] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %35 = x86.dmk.vmovupd[%23 + 224], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %36 = x86.dm.vmovupd [%23 + 240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %37 = x86.dmk.vmovupd[%23 + 304], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %38 = x86.dm.vmovupd [%23 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %39 = x86.dmk.vmovupd[%23 + 384], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %40 = x86.dm.vmovupd [%21] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %41 = x86.dmk.vmovupd[%21 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %42 = x86.dm.vbroadcastsd [%22] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %43 = x86.rss.vfmadd231pd %30, %40, %42 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %44 = x86.rss.vfmadd231pd %31, %41, %42 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %45 = x86.dm.vbroadcastsd [%22 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %46 = x86.rss.vfmadd231pd %32, %40, %45 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %47 = x86.rss.vfmadd231pd %33, %41, %45 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %48 = x86.dm.vbroadcastsd [%22 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %49 = x86.rss.vfmadd231pd %34, %40, %48 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %50 = x86.rss.vfmadd231pd %35, %41, %48 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %51 = x86.dm.vbroadcastsd [%22 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %52 = x86.rss.vfmadd231pd %36, %40, %51 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %53 = x86.rss.vfmadd231pd %37, %41, %51 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %54 = x86.dm.vbroadcastsd [%22 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %55 = x86.ri.add %22, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %56 = x86.ri.add %21, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %57 = x86.rss.vfmadd231pd %38, %40, %54 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %58 = x86.rss.vfmadd231pd %39, %41, %54 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %59 = x86.dm.vmovupd [%56] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %60 = x86.dmk.vmovupd[%56 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %61 = x86.dm.vbroadcastsd [%55] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %62 = x86.rss.vfmadd231pd %43, %59, %61 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %63 = x86.rss.vfmadd231pd %44, %60, %61 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %64 = x86.dm.vbroadcastsd [%55 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %65 = x86.rss.vfmadd231pd %46, %59, %64 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %66 = x86.rss.vfmadd231pd %47, %60, %64 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %67 = x86.dm.vbroadcastsd [%55 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %68 = x86.rss.vfmadd231pd %49, %59, %67 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %69 = x86.rss.vfmadd231pd %50, %60, %67 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %70 = x86.dm.vbroadcastsd [%55 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %71 = x86.rss.vfmadd231pd %52, %59, %70 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %72 = x86.rss.vfmadd231pd %53, %60, %70 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %73 = x86.dm.vbroadcastsd [%55 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %74 = x86.ri.add %55, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %75 = x86.ri.add %56, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %76 = x86.rss.vfmadd231pd %57, %59, %73 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %77 = x86.rss.vfmadd231pd %58, %60, %73 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %78 = x86.dm.vmovupd [%75] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %79 = x86.dmk.vmovupd[%75 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %80 = x86.dm.vbroadcastsd [%74] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %81 = x86.rss.vfmadd231pd %62, %78, %80 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %82 = x86.rss.vfmadd231pd %63, %79, %80 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %83 = x86.dm.vbroadcastsd [%74 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %84 = x86.rss.vfmadd231pd %65, %78, %83 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %85 = x86.rss.vfmadd231pd %66, %79, %83 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %86 = x86.dm.vbroadcastsd [%74 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %87 = x86.rss.vfmadd231pd %68, %78, %86 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %88 = x86.rss.vfmadd231pd %69, %79, %86 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %89 = x86.dm.vbroadcastsd [%74 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %90 = x86.rss.vfmadd231pd %71, %78, %89 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231pd %72, %79, %89 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %92 = x86.dm.vbroadcastsd [%74 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %93 = x86.ri.add %74, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %94 = x86.ri.add %75, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %95 = x86.rss.vfmadd231pd %76, %78, %92 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %96 = x86.rss.vfmadd231pd %77, %79, %92 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %97 = x86.dm.vmovupd [%94] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %98 = x86.dmk.vmovupd[%94 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %99 = x86.dm.vbroadcastsd [%93] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %100 = x86.rss.vfmadd231pd %81, %97, %99 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %101 = x86.rss.vfmadd231pd %82, %98, %99 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %102 = x86.dm.vbroadcastsd [%93 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %103 = x86.rss.vfmadd231pd %84, %97, %102 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %104 = x86.rss.vfmadd231pd %85, %98, %102 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %105 = x86.dm.vbroadcastsd [%93 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %106 = x86.rss.vfmadd231pd %87, %97, %105 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %107 = x86.rss.vfmadd231pd %88, %98, %105 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %108 = x86.dm.vbroadcastsd [%93 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %109 = x86.rss.vfmadd231pd %90, %97, %108 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %110 = x86.rss.vfmadd231pd %91, %98, %108 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %111 = x86.dm.vbroadcastsd [%93 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %112 = x86.ri.add %93, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %113 = x86.ri.add %94, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %114 = x86.rss.vfmadd231pd %95, %97, %111 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %115 = x86.rss.vfmadd231pd %96, %98, %111 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %116 = x86.dm.vmovupd [%113] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %117 = x86.dmk.vmovupd[%113 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %118 = x86.dm.vbroadcastsd [%112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %119 = x86.rss.vfmadd231pd %100, %116, %118 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %120 = x86.rss.vfmadd231pd %101, %117, %118 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %121 = x86.dm.vbroadcastsd [%112 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %122 = x86.rss.vfmadd231pd %103, %116, %121 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %123 = x86.rss.vfmadd231pd %104, %117, %121 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %124 = x86.dm.vbroadcastsd [%112 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %125 = x86.rss.vfmadd231pd %106, %116, %124 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %126 = x86.rss.vfmadd231pd %107, %117, %124 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %127 = x86.dm.vbroadcastsd [%112 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %128 = x86.rss.vfmadd231pd %109, %116, %127 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %129 = x86.rss.vfmadd231pd %110, %117, %127 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %130 = x86.dm.vbroadcastsd [%112 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %131 = x86.ri.add %112, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %132 = x86.ri.add %113, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %133 = x86.rss.vfmadd231pd %114, %116, %130 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %134 = x86.rss.vfmadd231pd %115, %117, %130 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %135 = x86.dm.vmovupd [%132] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %136 = x86.dmk.vmovupd[%132 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %137 = x86.dm.vbroadcastsd [%131] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %138 = x86.rss.vfmadd231pd %119, %135, %137 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %139 = x86.rss.vfmadd231pd %120, %136, %137 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %140 = x86.dm.vbroadcastsd [%131 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %141 = x86.rss.vfmadd231pd %122, %135, %140 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %142 = x86.rss.vfmadd231pd %123, %136, %140 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %143 = x86.dm.vbroadcastsd [%131 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %144 = x86.rss.vfmadd231pd %125, %135, %143 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %145 = x86.rss.vfmadd231pd %126, %136, %143 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %146 = x86.dm.vbroadcastsd [%131 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %147 = x86.rss.vfmadd231pd %128, %135, %146 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %148 = x86.rss.vfmadd231pd %129, %136, %146 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %149 = x86.dm.vbroadcastsd [%131 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %150 = x86.ri.add %131, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %151 = x86.ri.add %132, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %152 = x86.rss.vfmadd231pd %133, %135, %149 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %153 = x86.rss.vfmadd231pd %134, %136, %149 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %154 = x86.dm.vmovupd [%151] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %155 = x86.dmk.vmovupd[%151 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %156 = x86.dm.vbroadcastsd [%150] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %157 = x86.rss.vfmadd231pd %138, %154, %156 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %158 = x86.rss.vfmadd231pd %139, %155, %156 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %159 = x86.dm.vbroadcastsd [%150 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %160 = x86.rss.vfmadd231pd %141, %154, %159 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %161 = x86.rss.vfmadd231pd %142, %155, %159 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %162 = x86.dm.vbroadcastsd [%150 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %163 = x86.rss.vfmadd231pd %144, %154, %162 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %164 = x86.rss.vfmadd231pd %145, %155, %162 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %165 = x86.dm.vbroadcastsd [%150 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %166 = x86.rss.vfmadd231pd %147, %154, %165 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %167 = x86.rss.vfmadd231pd %148, %155, %165 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %168 = x86.dm.vbroadcastsd [%150 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %169 = x86.ri.add %150, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %170 = x86.ri.add %151, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %171 = x86.rss.vfmadd231pd %152, %154, %168 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %172 = x86.rss.vfmadd231pd %153, %155, %168 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %173 = x86.dm.vmovupd [%170] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %174 = x86.dmk.vmovupd[%170 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %175 = x86.dm.vbroadcastsd [%169] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %176 = x86.rss.vfmadd231pd %157, %173, %175 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %177 = x86.rss.vfmadd231pd %158, %174, %175 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %178 = x86.dm.vbroadcastsd [%169 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %179 = x86.rss.vfmadd231pd %160, %173, %178 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %180 = x86.rss.vfmadd231pd %161, %174, %178 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %181 = x86.dm.vbroadcastsd [%169 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %182 = x86.rss.vfmadd231pd %163, %173, %181 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %183 = x86.rss.vfmadd231pd %164, %174, %181 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %184 = x86.dm.vbroadcastsd [%169 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %185 = x86.rss.vfmadd231pd %166, %173, %184 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %186 = x86.rss.vfmadd231pd %167, %174, %184 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %187 = x86.dm.vbroadcastsd [%169 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %188 = x86.ri.add %169, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %189 = x86.ri.add %170, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %190 = x86.rss.vfmadd231pd %171, %173, %187 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %191 = x86.rss.vfmadd231pd %172, %174, %187 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %192 = x86.dm.vmovupd [%189] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %193 = x86.dmk.vmovupd[%189 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %194 = x86.dm.vbroadcastsd [%188] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %195 = x86.rss.vfmadd231pd %176, %192, %194 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %196 = x86.rss.vfmadd231pd %177, %193, %194 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %197 = x86.dm.vbroadcastsd [%188 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %198 = x86.rss.vfmadd231pd %179, %192, %197 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %199 = x86.rss.vfmadd231pd %180, %193, %197 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %200 = x86.dm.vbroadcastsd [%188 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %201 = x86.rss.vfmadd231pd %182, %192, %200 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %202 = x86.rss.vfmadd231pd %183, %193, %200 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %203 = x86.dm.vbroadcastsd [%188 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %204 = x86.rss.vfmadd231pd %185, %192, %203 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %205 = x86.rss.vfmadd231pd %186, %193, %203 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %206 = x86.dm.vbroadcastsd [%188 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %207 = x86.ri.add %188, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %208 = x86.ri.add %189, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %209 = x86.rss.vfmadd231pd %190, %192, %206 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %210 = x86.rss.vfmadd231pd %191, %193, %206 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %211 = x86.dm.vmovupd [%208] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %212 = x86.dmk.vmovupd[%208 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %213 = x86.dm.vbroadcastsd [%207] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %214 = x86.rss.vfmadd231pd %195, %211, %213 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %215 = x86.rss.vfmadd231pd %196, %212, %213 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %216 = x86.dm.vbroadcastsd [%207 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %217 = x86.rss.vfmadd231pd %198, %211, %216 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %218 = x86.rss.vfmadd231pd %199, %212, %216 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %219 = x86.dm.vbroadcastsd [%207 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %220 = x86.rss.vfmadd231pd %201, %211, %219 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %221 = x86.rss.vfmadd231pd %202, %212, %219 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %222 = x86.dm.vbroadcastsd [%207 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %223 = x86.rss.vfmadd231pd %204, %211, %222 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %224 = x86.rss.vfmadd231pd %205, %212, %222 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %225 = x86.dm.vbroadcastsd [%207 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %226 = x86.ri.add %207, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %227 = x86.ri.add %208, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %228 = x86.rss.vfmadd231pd %209, %211, %225 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %229 = x86.rss.vfmadd231pd %210, %212, %225 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %230 = x86.dm.vmovupd [%227] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %231 = x86.dmk.vmovupd[%227 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %232 = x86.dm.vbroadcastsd [%226] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %233 = x86.rss.vfmadd231pd %214, %230, %232 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %234 = x86.rss.vfmadd231pd %215, %231, %232 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %235 = x86.dm.vbroadcastsd [%226 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %236 = x86.rss.vfmadd231pd %217, %230, %235 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %237 = x86.rss.vfmadd231pd %218, %231, %235 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %238 = x86.dm.vbroadcastsd [%226 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %239 = x86.rss.vfmadd231pd %220, %230, %238 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %240 = x86.rss.vfmadd231pd %221, %231, %238 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %241 = x86.dm.vbroadcastsd [%226 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %242 = x86.rss.vfmadd231pd %223, %230, %241 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %243 = x86.rss.vfmadd231pd %224, %231, %241 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %244 = x86.dm.vbroadcastsd [%226 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %245 = x86.ri.add %226, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %246 = x86.ri.add %227, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %247 = x86.rss.vfmadd231pd %228, %230, %244 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %248 = x86.rss.vfmadd231pd %229, %231, %244 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %249 = x86.dm.vmovupd [%246] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %250 = x86.dmk.vmovupd[%246 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %251 = x86.dm.vbroadcastsd [%245] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %252 = x86.rss.vfmadd231pd %233, %249, %251 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %253 = x86.rss.vfmadd231pd %234, %250, %251 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %254 = x86.dm.vbroadcastsd [%245 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %255 = x86.rss.vfmadd231pd %236, %249, %254 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %256 = x86.rss.vfmadd231pd %237, %250, %254 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %257 = x86.dm.vbroadcastsd [%245 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %258 = x86.rss.vfmadd231pd %239, %249, %257 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %259 = x86.rss.vfmadd231pd %240, %250, %257 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %260 = x86.dm.vbroadcastsd [%245 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %261 = x86.rss.vfmadd231pd %242, %249, %260 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %262 = x86.rss.vfmadd231pd %243, %250, %260 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %263 = x86.dm.vbroadcastsd [%245 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %264 = x86.ri.add %245, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %265 = x86.ri.add %246, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %266 = x86.rss.vfmadd231pd %247, %249, %263 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %267 = x86.rss.vfmadd231pd %248, %250, %263 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %268 = x86.dm.vmovupd [%265] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %269 = x86.dmk.vmovupd[%265 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %270 = x86.dm.vbroadcastsd [%264] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %271 = x86.rss.vfmadd231pd %252, %268, %270 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %272 = x86.rss.vfmadd231pd %253, %269, %270 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %273 = x86.dm.vbroadcastsd [%264 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %274 = x86.rss.vfmadd231pd %255, %268, %273 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %275 = x86.rss.vfmadd231pd %256, %269, %273 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %276 = x86.dm.vbroadcastsd [%264 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %277 = x86.rss.vfmadd231pd %258, %268, %276 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %278 = x86.rss.vfmadd231pd %259, %269, %276 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %279 = x86.dm.vbroadcastsd [%264 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %280 = x86.rss.vfmadd231pd %261, %268, %279 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %281 = x86.rss.vfmadd231pd %262, %269, %279 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %282 = x86.dm.vbroadcastsd [%264 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %283 = x86.ri.add %264, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %284 = x86.ri.add %265, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %285 = x86.rss.vfmadd231pd %266, %268, %282 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %286 = x86.rss.vfmadd231pd %267, %269, %282 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %287 = x86.dm.vmovupd [%284] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %288 = x86.dmk.vmovupd[%284 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %289 = x86.dm.vbroadcastsd [%283] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %290 = x86.rss.vfmadd231pd %271, %287, %289 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %291 = x86.rss.vfmadd231pd %272, %288, %289 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %292 = x86.dm.vbroadcastsd [%283 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %293 = x86.rss.vfmadd231pd %274, %287, %292 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %294 = x86.rss.vfmadd231pd %275, %288, %292 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %295 = x86.dm.vbroadcastsd [%283 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %296 = x86.rss.vfmadd231pd %277, %287, %295 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %297 = x86.rss.vfmadd231pd %278, %288, %295 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %298 = x86.dm.vbroadcastsd [%283 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %299 = x86.rss.vfmadd231pd %280, %287, %298 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %300 = x86.rss.vfmadd231pd %281, %288, %298 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %301 = x86.dm.vbroadcastsd [%283 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %302 = x86.ri.add %283, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %303 = x86.ri.add %284, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %304 = x86.rss.vfmadd231pd %285, %287, %301 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %305 = x86.rss.vfmadd231pd %286, %288, %301 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %306 = x86.dm.vmovupd [%303] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %307 = x86.dmk.vmovupd[%303 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %308 = x86.dm.vbroadcastsd [%302] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %309 = x86.rss.vfmadd231pd %290, %306, %308 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %310 = x86.rss.vfmadd231pd %291, %307, %308 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %311 = x86.dm.vbroadcastsd [%302 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %312 = x86.rss.vfmadd231pd %293, %306, %311 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %313 = x86.rss.vfmadd231pd %294, %307, %311 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %314 = x86.dm.vbroadcastsd [%302 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %315 = x86.rss.vfmadd231pd %296, %306, %314 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %316 = x86.rss.vfmadd231pd %297, %307, %314 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %317 = x86.dm.vbroadcastsd [%302 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %318 = x86.rss.vfmadd231pd %299, %306, %317 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %319 = x86.rss.vfmadd231pd %300, %307, %317 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %320 = x86.dm.vbroadcastsd [%302 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %321 = x86.ri.add %302, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %322 = x86.ri.add %303, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %323 = x86.rss.vfmadd231pd %304, %306, %320 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %324 = x86.rss.vfmadd231pd %305, %307, %320 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %325 = x86.dm.vmovupd [%322] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %326 = x86.dmk.vmovupd[%322 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %327 = x86.dm.vbroadcastsd [%321] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %328 = x86.rss.vfmadd231pd %309, %325, %327 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %329 = x86.rss.vfmadd231pd %310, %326, %327 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %330 = x86.dm.vbroadcastsd [%321 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %331 = x86.rss.vfmadd231pd %312, %325, %330 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %332 = x86.rss.vfmadd231pd %313, %326, %330 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %333 = x86.dm.vbroadcastsd [%321 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %334 = x86.rss.vfmadd231pd %315, %325, %333 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %335 = x86.rss.vfmadd231pd %316, %326, %333 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %336 = x86.dm.vbroadcastsd [%321 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %337 = x86.rss.vfmadd231pd %318, %325, %336 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %338 = x86.rss.vfmadd231pd %319, %326, %336 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %339 = x86.dm.vbroadcastsd [%321 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %340 = x86.ri.add %321, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %341 = x86.ri.add %322, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %342 = x86.rss.vfmadd231pd %323, %325, %339 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %343 = x86.rss.vfmadd231pd %324, %326, %339 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %344 = x86.ri.sub %340, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovupd [%23], %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%23 + 64], %329, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%23 + 80], %331 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%23 + 144], %332, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%23 + 160], %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%23 + 224], %335, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%23 + 240], %337 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%23 + 304], %338, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%23 + 320], %342 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%23 + 384], %343, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %345 = x86.ri.add %23, 80 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %346 = x86.ri.sub %341, 1200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %347 = x86.si.cmp %29, 10 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %347 : !x86.rflags<rflags>, ^bb2(%346 : !x86.reg64<rdi>, %344 : !x86.reg64<rsi>, %345 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %29 : !x86.reg64<r10>, %28 : !x86.avx512maskreg<k1>), ^bb3(%346 : !x86.reg64<rdi>, %344 : !x86.reg64<rsi>, %345 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %29 : !x86.reg64<r10>, %28 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb3(%348: !x86.reg64<rdi>, %349: !x86.reg64<rsi>, %350: !x86.reg64<rdx>, %351: !x86.reg64<rbp>, %352: !x86.reg64<rsp>, %353: !x86.reg64<r11>, %354: !x86.reg64<r10>, %355: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      %356 = x86.ri.add %350, 320 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %357 = x86.ri.add %349, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %358 = x86.ri.sub %348, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %359 = x86.si.cmp %353, 5 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %359 : !x86.rflags<rflags>, ^bb1(%358 : !x86.reg64<rdi>, %357 : !x86.reg64<rsi>, %356 : !x86.reg64<rdx>, %351 : !x86.reg64<rbp>, %352 : !x86.reg64<rsp>, %353 : !x86.reg64<r11>), ^bb4(%358 : !x86.reg64<rdi>, %357 : !x86.reg64<rsi>, %356 : !x86.reg64<rdx>, %351 : !x86.reg64<rbp>, %352 : !x86.reg64<rsp>, %353 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb4(%360: !x86.reg64<rdi>, %361: !x86.reg64<rsi>, %362: !x86.reg64<rdx>, %363: !x86.reg64<rbp>, %364: !x86.reg64<rsp>, %365: !x86.reg64<r11>):
// CHECK-NEXT:      %366 = x86.ds.mov %363 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %367, %368 = x86.d.pop %366 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
