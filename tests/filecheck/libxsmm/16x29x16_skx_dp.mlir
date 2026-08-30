// RUN: libxsmm-gemm dense %t matmul_bac 16 29 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 16 29 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 16 29 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      mov rbp, rsp
// CHECK-MANUAL-NEXT:      sub rsp, 192
// CHECK-MANUAL-NEXT:      mov r10, -64
// CHECK-MANUAL-NEXT:      and rsp, r10
// CHECK-MANUAL-NEXT:      mov r11, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_33:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 10
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_34:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 16
// CHECK-MANUAL-NEXT:      vmovapd zmm12, [rdx]
// CHECK-MANUAL-NEXT:      vmovapd zmm13, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovapd zmm14, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovapd zmm15, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovapd zmm16, [rdx+256]
// CHECK-MANUAL-NEXT:      vmovapd zmm17, [rdx+320]
// CHECK-MANUAL-NEXT:      vmovapd zmm18, [rdx+384]
// CHECK-MANUAL-NEXT:      vmovapd zmm19, [rdx+448]
// CHECK-MANUAL-NEXT:      vmovapd zmm20, [rdx+512]
// CHECK-MANUAL-NEXT:      vmovapd zmm21, [rdx+576]
// CHECK-MANUAL-NEXT:      vmovapd zmm22, [rdx+640]
// CHECK-MANUAL-NEXT:      vmovapd zmm23, [rdx+704]
// CHECK-MANUAL-NEXT:      vmovapd zmm24, [rdx+768]
// CHECK-MANUAL-NEXT:      vmovapd zmm25, [rdx+832]
// CHECK-MANUAL-NEXT:      vmovapd zmm26, [rdx+896]
// CHECK-MANUAL-NEXT:      vmovapd zmm27, [rdx+960]
// CHECK-MANUAL-NEXT:      vmovapd zmm28, [rdx+1024]
// CHECK-MANUAL-NEXT:      vmovapd zmm29, [rdx+1088]
// CHECK-MANUAL-NEXT:      vmovapd zmm30, [rdx+1152]
// CHECK-MANUAL-NEXT:      vmovapd zmm31, [rdx+1216]
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      sub rsi, 128
// CHECK-MANUAL-NEXT:      vmovapd [rdx], zmm12
// CHECK-MANUAL-NEXT:      vmovapd [rdx+64], zmm13
// CHECK-MANUAL-NEXT:      vmovapd [rdx+128], zmm14
// CHECK-MANUAL-NEXT:      vmovapd [rdx+192], zmm15
// CHECK-MANUAL-NEXT:      vmovapd [rdx+256], zmm16
// CHECK-MANUAL-NEXT:      vmovapd [rdx+320], zmm17
// CHECK-MANUAL-NEXT:      vmovapd [rdx+384], zmm18
// CHECK-MANUAL-NEXT:      vmovapd [rdx+448], zmm19
// CHECK-MANUAL-NEXT:      vmovapd [rdx+512], zmm20
// CHECK-MANUAL-NEXT:      vmovapd [rdx+576], zmm21
// CHECK-MANUAL-NEXT:      vmovapd [rdx+640], zmm22
// CHECK-MANUAL-NEXT:      vmovapd [rdx+704], zmm23
// CHECK-MANUAL-NEXT:      vmovapd [rdx+768], zmm24
// CHECK-MANUAL-NEXT:      vmovapd [rdx+832], zmm25
// CHECK-MANUAL-NEXT:      vmovapd [rdx+896], zmm26
// CHECK-MANUAL-NEXT:      vmovapd [rdx+960], zmm27
// CHECK-MANUAL-NEXT:      vmovapd [rdx+1024], zmm28
// CHECK-MANUAL-NEXT:      vmovapd [rdx+1088], zmm29
// CHECK-MANUAL-NEXT:      vmovapd [rdx+1152], zmm30
// CHECK-MANUAL-NEXT:      vmovapd [rdx+1216], zmm31
// CHECK-MANUAL-NEXT:      add rdx, 128
// CHECK-MANUAL-NEXT:      sub rdi, 1920
// CHECK-MANUAL-NEXT:      cmp r10, 16
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_34]]
// CHECK-LIBXSMM-NEXT:     add rdx, 1152
// CHECK-COMPXSMM-NEXT:    sub rdi, 128
// CHECK-MANUAL-NEXT:      add rsi, 1280
// CHECK-LIBXSMM-NEXT:     sub rdi, 128
// CHECK-COMPXSMM-NEXT:    add rdx, 1152
// CHECK-MANUAL-NEXT:      cmp r11, 20
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_33]]
// CHECK-MANUAL-NEXT:      mov r11, 20
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_35:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 9
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_36:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 16
// CHECK-MANUAL-NEXT:      vmovapd zmm14, [rdx]
// CHECK-MANUAL-NEXT:      vmovapd zmm15, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovapd zmm16, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovapd zmm17, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovapd zmm18, [rdx+256]
// CHECK-MANUAL-NEXT:      vmovapd zmm19, [rdx+320]
// CHECK-MANUAL-NEXT:      vmovapd zmm20, [rdx+384]
// CHECK-MANUAL-NEXT:      vmovapd zmm21, [rdx+448]
// CHECK-MANUAL-NEXT:      vmovapd zmm22, [rdx+512]
// CHECK-MANUAL-NEXT:      vmovapd zmm23, [rdx+576]
// CHECK-MANUAL-NEXT:      vmovapd zmm24, [rdx+640]
// CHECK-MANUAL-NEXT:      vmovapd zmm25, [rdx+704]
// CHECK-MANUAL-NEXT:      vmovapd zmm26, [rdx+768]
// CHECK-MANUAL-NEXT:      vmovapd zmm27, [rdx+832]
// CHECK-MANUAL-NEXT:      vmovapd zmm28, [rdx+896]
// CHECK-MANUAL-NEXT:      vmovapd zmm29, [rdx+960]
// CHECK-MANUAL-NEXT:      vmovapd zmm30, [rdx+1024]
// CHECK-MANUAL-NEXT:      vmovapd zmm31, [rdx+1088]
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+640]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+896]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      sub rsi, 128
// CHECK-MANUAL-NEXT:      vmovapd [rdx], zmm14
// CHECK-MANUAL-NEXT:      vmovapd [rdx+64], zmm15
// CHECK-MANUAL-NEXT:      vmovapd [rdx+128], zmm16
// CHECK-MANUAL-NEXT:      vmovapd [rdx+192], zmm17
// CHECK-MANUAL-NEXT:      vmovapd [rdx+256], zmm18
// CHECK-MANUAL-NEXT:      vmovapd [rdx+320], zmm19
// CHECK-MANUAL-NEXT:      vmovapd [rdx+384], zmm20
// CHECK-MANUAL-NEXT:      vmovapd [rdx+448], zmm21
// CHECK-MANUAL-NEXT:      vmovapd [rdx+512], zmm22
// CHECK-MANUAL-NEXT:      vmovapd [rdx+576], zmm23
// CHECK-MANUAL-NEXT:      vmovapd [rdx+640], zmm24
// CHECK-MANUAL-NEXT:      vmovapd [rdx+704], zmm25
// CHECK-MANUAL-NEXT:      vmovapd [rdx+768], zmm26
// CHECK-MANUAL-NEXT:      vmovapd [rdx+832], zmm27
// CHECK-MANUAL-NEXT:      vmovapd [rdx+896], zmm28
// CHECK-MANUAL-NEXT:      vmovapd [rdx+960], zmm29
// CHECK-MANUAL-NEXT:      vmovapd [rdx+1024], zmm30
// CHECK-MANUAL-NEXT:      vmovapd [rdx+1088], zmm31
// CHECK-MANUAL-NEXT:      add rdx, 128
// CHECK-MANUAL-NEXT:      sub rdi, 1920
// CHECK-MANUAL-NEXT:      cmp r10, 16
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_36]]
// CHECK-LIBXSMM-NEXT:     add rdx, 1024
// CHECK-COMPXSMM-NEXT:    sub rdi, 128
// CHECK-MANUAL-NEXT:      add rsi, 1152
// CHECK-LIBXSMM-NEXT:     sub rdi, 128
// CHECK-COMPXSMM-NEXT:    add rdx, 1024
// CHECK-MANUAL-NEXT:      cmp r11, 29
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_35]]
// CHECK-MANUAL-NEXT:      mov rsp, rbp
// CHECK-MANUAL-NEXT:      pop rbp
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
// CHECK-NEXT:      %17 = x86.ri.add %16, 10 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %26 = x86.ri.add %25, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %27 = x86.dm.vmovapd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %28 = x86.dm.vmovapd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %29 = x86.dm.vmovapd [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %30 = x86.dm.vmovapd [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %31 = x86.dm.vmovapd [%21 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %32 = x86.dm.vmovapd [%21 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %33 = x86.dm.vmovapd [%21 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %34 = x86.dm.vmovapd [%21 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %35 = x86.dm.vmovapd [%21 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %36 = x86.dm.vmovapd [%21 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %37 = x86.dm.vmovapd [%21 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %38 = x86.dm.vmovapd [%21 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %39 = x86.dm.vmovapd [%21 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %40 = x86.dm.vmovapd [%21 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %41 = x86.dm.vmovapd [%21 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %42 = x86.dm.vmovapd [%21 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %43 = x86.dm.vmovapd [%21 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %44 = x86.dm.vmovapd [%21 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %45 = x86.dm.vmovapd [%21 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %46 = x86.dm.vmovapd [%21 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %47 = x86.dm.vmovapd [%19] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %48 = x86.dm.vmovapd [%19 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %49 = x86.dm.vbroadcastsd [%20] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %50 = x86.rss.vfmadd231pd %27, %47, %49 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %51 = x86.rss.vfmadd231pd %28, %48, %49 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %52 = x86.dm.vbroadcastsd [%20 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %53 = x86.rss.vfmadd231pd %29, %47, %52 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %54 = x86.rss.vfmadd231pd %30, %48, %52 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %55 = x86.dm.vbroadcastsd [%20 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %56 = x86.rss.vfmadd231pd %31, %47, %55 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %57 = x86.rss.vfmadd231pd %32, %48, %55 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %58 = x86.dm.vbroadcastsd [%20 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %59 = x86.rss.vfmadd231pd %33, %47, %58 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %60 = x86.rss.vfmadd231pd %34, %48, %58 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %61 = x86.dm.vbroadcastsd [%20 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %62 = x86.rss.vfmadd231pd %35, %47, %61 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %63 = x86.rss.vfmadd231pd %36, %48, %61 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %64 = x86.dm.vbroadcastsd [%20 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %65 = x86.rss.vfmadd231pd %37, %47, %64 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %66 = x86.rss.vfmadd231pd %38, %48, %64 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %67 = x86.dm.vbroadcastsd [%20 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %68 = x86.rss.vfmadd231pd %39, %47, %67 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %69 = x86.rss.vfmadd231pd %40, %48, %67 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %70 = x86.dm.vbroadcastsd [%20 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %71 = x86.rss.vfmadd231pd %41, %47, %70 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %72 = x86.rss.vfmadd231pd %42, %48, %70 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %73 = x86.dm.vbroadcastsd [%20 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %74 = x86.rss.vfmadd231pd %43, %47, %73 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %75 = x86.rss.vfmadd231pd %44, %48, %73 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %76 = x86.dm.vbroadcastsd [%20 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %77 = x86.ri.add %20, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %78 = x86.ri.add %19, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %79 = x86.rss.vfmadd231pd %45, %47, %76 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %80 = x86.rss.vfmadd231pd %46, %48, %76 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %81 = x86.dm.vmovapd [%78] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %82 = x86.dm.vmovapd [%78 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %83 = x86.dm.vbroadcastsd [%77] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %84 = x86.rss.vfmadd231pd %50, %81, %83 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %85 = x86.rss.vfmadd231pd %51, %82, %83 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %86 = x86.dm.vbroadcastsd [%77 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %87 = x86.rss.vfmadd231pd %53, %81, %86 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %88 = x86.rss.vfmadd231pd %54, %82, %86 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %89 = x86.dm.vbroadcastsd [%77 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %90 = x86.rss.vfmadd231pd %56, %81, %89 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231pd %57, %82, %89 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %92 = x86.dm.vbroadcastsd [%77 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %93 = x86.rss.vfmadd231pd %59, %81, %92 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %94 = x86.rss.vfmadd231pd %60, %82, %92 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %95 = x86.dm.vbroadcastsd [%77 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %96 = x86.rss.vfmadd231pd %62, %81, %95 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %97 = x86.rss.vfmadd231pd %63, %82, %95 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %98 = x86.dm.vbroadcastsd [%77 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %99 = x86.rss.vfmadd231pd %65, %81, %98 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %100 = x86.rss.vfmadd231pd %66, %82, %98 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %101 = x86.dm.vbroadcastsd [%77 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %102 = x86.rss.vfmadd231pd %68, %81, %101 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %103 = x86.rss.vfmadd231pd %69, %82, %101 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %104 = x86.dm.vbroadcastsd [%77 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %105 = x86.rss.vfmadd231pd %71, %81, %104 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %106 = x86.rss.vfmadd231pd %72, %82, %104 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %107 = x86.dm.vbroadcastsd [%77 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %108 = x86.rss.vfmadd231pd %74, %81, %107 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %109 = x86.rss.vfmadd231pd %75, %82, %107 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %110 = x86.dm.vbroadcastsd [%77 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %111 = x86.ri.add %77, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %112 = x86.ri.add %78, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %113 = x86.rss.vfmadd231pd %79, %81, %110 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %114 = x86.rss.vfmadd231pd %80, %82, %110 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %115 = x86.dm.vmovapd [%112] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %116 = x86.dm.vmovapd [%112 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %117 = x86.dm.vbroadcastsd [%111] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %118 = x86.rss.vfmadd231pd %84, %115, %117 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %119 = x86.rss.vfmadd231pd %85, %116, %117 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %120 = x86.dm.vbroadcastsd [%111 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %121 = x86.rss.vfmadd231pd %87, %115, %120 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %122 = x86.rss.vfmadd231pd %88, %116, %120 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %123 = x86.dm.vbroadcastsd [%111 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %124 = x86.rss.vfmadd231pd %90, %115, %123 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %125 = x86.rss.vfmadd231pd %91, %116, %123 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %126 = x86.dm.vbroadcastsd [%111 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %127 = x86.rss.vfmadd231pd %93, %115, %126 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %128 = x86.rss.vfmadd231pd %94, %116, %126 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %129 = x86.dm.vbroadcastsd [%111 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %130 = x86.rss.vfmadd231pd %96, %115, %129 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %131 = x86.rss.vfmadd231pd %97, %116, %129 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %132 = x86.dm.vbroadcastsd [%111 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %133 = x86.rss.vfmadd231pd %99, %115, %132 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %134 = x86.rss.vfmadd231pd %100, %116, %132 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %135 = x86.dm.vbroadcastsd [%111 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %136 = x86.rss.vfmadd231pd %102, %115, %135 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %137 = x86.rss.vfmadd231pd %103, %116, %135 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %138 = x86.dm.vbroadcastsd [%111 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %139 = x86.rss.vfmadd231pd %105, %115, %138 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %140 = x86.rss.vfmadd231pd %106, %116, %138 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %141 = x86.dm.vbroadcastsd [%111 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %142 = x86.rss.vfmadd231pd %108, %115, %141 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %143 = x86.rss.vfmadd231pd %109, %116, %141 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %144 = x86.dm.vbroadcastsd [%111 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %145 = x86.ri.add %111, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %146 = x86.ri.add %112, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %147 = x86.rss.vfmadd231pd %113, %115, %144 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %148 = x86.rss.vfmadd231pd %114, %116, %144 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %149 = x86.dm.vmovapd [%146] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %150 = x86.dm.vmovapd [%146 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %151 = x86.dm.vbroadcastsd [%145] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %152 = x86.rss.vfmadd231pd %118, %149, %151 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %153 = x86.rss.vfmadd231pd %119, %150, %151 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %154 = x86.dm.vbroadcastsd [%145 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %155 = x86.rss.vfmadd231pd %121, %149, %154 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %156 = x86.rss.vfmadd231pd %122, %150, %154 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %157 = x86.dm.vbroadcastsd [%145 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %158 = x86.rss.vfmadd231pd %124, %149, %157 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %159 = x86.rss.vfmadd231pd %125, %150, %157 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %160 = x86.dm.vbroadcastsd [%145 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %161 = x86.rss.vfmadd231pd %127, %149, %160 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %162 = x86.rss.vfmadd231pd %128, %150, %160 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %163 = x86.dm.vbroadcastsd [%145 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %164 = x86.rss.vfmadd231pd %130, %149, %163 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %165 = x86.rss.vfmadd231pd %131, %150, %163 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %166 = x86.dm.vbroadcastsd [%145 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %167 = x86.rss.vfmadd231pd %133, %149, %166 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %168 = x86.rss.vfmadd231pd %134, %150, %166 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %169 = x86.dm.vbroadcastsd [%145 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %170 = x86.rss.vfmadd231pd %136, %149, %169 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %171 = x86.rss.vfmadd231pd %137, %150, %169 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %172 = x86.dm.vbroadcastsd [%145 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %173 = x86.rss.vfmadd231pd %139, %149, %172 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %174 = x86.rss.vfmadd231pd %140, %150, %172 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %175 = x86.dm.vbroadcastsd [%145 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %176 = x86.rss.vfmadd231pd %142, %149, %175 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %177 = x86.rss.vfmadd231pd %143, %150, %175 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %178 = x86.dm.vbroadcastsd [%145 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %179 = x86.ri.add %145, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %180 = x86.ri.add %146, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %181 = x86.rss.vfmadd231pd %147, %149, %178 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %182 = x86.rss.vfmadd231pd %148, %150, %178 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %183 = x86.dm.vmovapd [%180] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %184 = x86.dm.vmovapd [%180 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %185 = x86.dm.vbroadcastsd [%179] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %186 = x86.rss.vfmadd231pd %152, %183, %185 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %187 = x86.rss.vfmadd231pd %153, %184, %185 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %188 = x86.dm.vbroadcastsd [%179 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %189 = x86.rss.vfmadd231pd %155, %183, %188 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %190 = x86.rss.vfmadd231pd %156, %184, %188 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %191 = x86.dm.vbroadcastsd [%179 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %192 = x86.rss.vfmadd231pd %158, %183, %191 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %193 = x86.rss.vfmadd231pd %159, %184, %191 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %194 = x86.dm.vbroadcastsd [%179 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %195 = x86.rss.vfmadd231pd %161, %183, %194 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %196 = x86.rss.vfmadd231pd %162, %184, %194 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %197 = x86.dm.vbroadcastsd [%179 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %198 = x86.rss.vfmadd231pd %164, %183, %197 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %199 = x86.rss.vfmadd231pd %165, %184, %197 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %200 = x86.dm.vbroadcastsd [%179 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %201 = x86.rss.vfmadd231pd %167, %183, %200 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %202 = x86.rss.vfmadd231pd %168, %184, %200 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %203 = x86.dm.vbroadcastsd [%179 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %204 = x86.rss.vfmadd231pd %170, %183, %203 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %205 = x86.rss.vfmadd231pd %171, %184, %203 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %206 = x86.dm.vbroadcastsd [%179 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %207 = x86.rss.vfmadd231pd %173, %183, %206 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %208 = x86.rss.vfmadd231pd %174, %184, %206 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %209 = x86.dm.vbroadcastsd [%179 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %210 = x86.rss.vfmadd231pd %176, %183, %209 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %211 = x86.rss.vfmadd231pd %177, %184, %209 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %212 = x86.dm.vbroadcastsd [%179 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %213 = x86.ri.add %179, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %214 = x86.ri.add %180, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %215 = x86.rss.vfmadd231pd %181, %183, %212 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %216 = x86.rss.vfmadd231pd %182, %184, %212 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %217 = x86.dm.vmovapd [%214] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %218 = x86.dm.vmovapd [%214 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %219 = x86.dm.vbroadcastsd [%213] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %220 = x86.rss.vfmadd231pd %186, %217, %219 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %221 = x86.rss.vfmadd231pd %187, %218, %219 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %222 = x86.dm.vbroadcastsd [%213 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %223 = x86.rss.vfmadd231pd %189, %217, %222 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %224 = x86.rss.vfmadd231pd %190, %218, %222 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %225 = x86.dm.vbroadcastsd [%213 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %226 = x86.rss.vfmadd231pd %192, %217, %225 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %227 = x86.rss.vfmadd231pd %193, %218, %225 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %228 = x86.dm.vbroadcastsd [%213 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %229 = x86.rss.vfmadd231pd %195, %217, %228 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %230 = x86.rss.vfmadd231pd %196, %218, %228 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %231 = x86.dm.vbroadcastsd [%213 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %232 = x86.rss.vfmadd231pd %198, %217, %231 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %233 = x86.rss.vfmadd231pd %199, %218, %231 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %234 = x86.dm.vbroadcastsd [%213 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %235 = x86.rss.vfmadd231pd %201, %217, %234 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %236 = x86.rss.vfmadd231pd %202, %218, %234 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %237 = x86.dm.vbroadcastsd [%213 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %238 = x86.rss.vfmadd231pd %204, %217, %237 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %239 = x86.rss.vfmadd231pd %205, %218, %237 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %240 = x86.dm.vbroadcastsd [%213 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %241 = x86.rss.vfmadd231pd %207, %217, %240 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %242 = x86.rss.vfmadd231pd %208, %218, %240 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %243 = x86.dm.vbroadcastsd [%213 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %244 = x86.rss.vfmadd231pd %210, %217, %243 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %245 = x86.rss.vfmadd231pd %211, %218, %243 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %246 = x86.dm.vbroadcastsd [%213 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %247 = x86.ri.add %213, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %248 = x86.ri.add %214, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %249 = x86.rss.vfmadd231pd %215, %217, %246 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %250 = x86.rss.vfmadd231pd %216, %218, %246 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %251 = x86.dm.vmovapd [%248] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %252 = x86.dm.vmovapd [%248 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %253 = x86.dm.vbroadcastsd [%247] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %254 = x86.rss.vfmadd231pd %220, %251, %253 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %255 = x86.rss.vfmadd231pd %221, %252, %253 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %256 = x86.dm.vbroadcastsd [%247 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %257 = x86.rss.vfmadd231pd %223, %251, %256 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %258 = x86.rss.vfmadd231pd %224, %252, %256 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %259 = x86.dm.vbroadcastsd [%247 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %260 = x86.rss.vfmadd231pd %226, %251, %259 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %261 = x86.rss.vfmadd231pd %227, %252, %259 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %262 = x86.dm.vbroadcastsd [%247 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %263 = x86.rss.vfmadd231pd %229, %251, %262 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %264 = x86.rss.vfmadd231pd %230, %252, %262 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %265 = x86.dm.vbroadcastsd [%247 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %266 = x86.rss.vfmadd231pd %232, %251, %265 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %267 = x86.rss.vfmadd231pd %233, %252, %265 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %268 = x86.dm.vbroadcastsd [%247 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %269 = x86.rss.vfmadd231pd %235, %251, %268 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %270 = x86.rss.vfmadd231pd %236, %252, %268 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %271 = x86.dm.vbroadcastsd [%247 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %272 = x86.rss.vfmadd231pd %238, %251, %271 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %273 = x86.rss.vfmadd231pd %239, %252, %271 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %274 = x86.dm.vbroadcastsd [%247 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %275 = x86.rss.vfmadd231pd %241, %251, %274 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %276 = x86.rss.vfmadd231pd %242, %252, %274 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %277 = x86.dm.vbroadcastsd [%247 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %278 = x86.rss.vfmadd231pd %244, %251, %277 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %279 = x86.rss.vfmadd231pd %245, %252, %277 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %280 = x86.dm.vbroadcastsd [%247 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %281 = x86.ri.add %247, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %282 = x86.ri.add %248, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %283 = x86.rss.vfmadd231pd %249, %251, %280 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %284 = x86.rss.vfmadd231pd %250, %252, %280 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %285 = x86.dm.vmovapd [%282] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %286 = x86.dm.vmovapd [%282 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %287 = x86.dm.vbroadcastsd [%281] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %288 = x86.rss.vfmadd231pd %254, %285, %287 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %289 = x86.rss.vfmadd231pd %255, %286, %287 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %290 = x86.dm.vbroadcastsd [%281 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %291 = x86.rss.vfmadd231pd %257, %285, %290 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %292 = x86.rss.vfmadd231pd %258, %286, %290 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %293 = x86.dm.vbroadcastsd [%281 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %294 = x86.rss.vfmadd231pd %260, %285, %293 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %295 = x86.rss.vfmadd231pd %261, %286, %293 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %296 = x86.dm.vbroadcastsd [%281 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %297 = x86.rss.vfmadd231pd %263, %285, %296 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %298 = x86.rss.vfmadd231pd %264, %286, %296 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %299 = x86.dm.vbroadcastsd [%281 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %300 = x86.rss.vfmadd231pd %266, %285, %299 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %301 = x86.rss.vfmadd231pd %267, %286, %299 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %302 = x86.dm.vbroadcastsd [%281 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %303 = x86.rss.vfmadd231pd %269, %285, %302 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %304 = x86.rss.vfmadd231pd %270, %286, %302 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %305 = x86.dm.vbroadcastsd [%281 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %306 = x86.rss.vfmadd231pd %272, %285, %305 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %307 = x86.rss.vfmadd231pd %273, %286, %305 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %308 = x86.dm.vbroadcastsd [%281 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %309 = x86.rss.vfmadd231pd %275, %285, %308 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %310 = x86.rss.vfmadd231pd %276, %286, %308 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %311 = x86.dm.vbroadcastsd [%281 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %312 = x86.rss.vfmadd231pd %278, %285, %311 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %313 = x86.rss.vfmadd231pd %279, %286, %311 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %314 = x86.dm.vbroadcastsd [%281 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %315 = x86.ri.add %281, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %316 = x86.ri.add %282, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %317 = x86.rss.vfmadd231pd %283, %285, %314 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %318 = x86.rss.vfmadd231pd %284, %286, %314 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %319 = x86.dm.vmovapd [%316] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %320 = x86.dm.vmovapd [%316 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %321 = x86.dm.vbroadcastsd [%315] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %322 = x86.rss.vfmadd231pd %288, %319, %321 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %323 = x86.rss.vfmadd231pd %289, %320, %321 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %324 = x86.dm.vbroadcastsd [%315 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %325 = x86.rss.vfmadd231pd %291, %319, %324 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %326 = x86.rss.vfmadd231pd %292, %320, %324 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %327 = x86.dm.vbroadcastsd [%315 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %328 = x86.rss.vfmadd231pd %294, %319, %327 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %329 = x86.rss.vfmadd231pd %295, %320, %327 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %330 = x86.dm.vbroadcastsd [%315 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %331 = x86.rss.vfmadd231pd %297, %319, %330 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %332 = x86.rss.vfmadd231pd %298, %320, %330 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %333 = x86.dm.vbroadcastsd [%315 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %334 = x86.rss.vfmadd231pd %300, %319, %333 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %335 = x86.rss.vfmadd231pd %301, %320, %333 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %336 = x86.dm.vbroadcastsd [%315 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %337 = x86.rss.vfmadd231pd %303, %319, %336 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %338 = x86.rss.vfmadd231pd %304, %320, %336 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %339 = x86.dm.vbroadcastsd [%315 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %340 = x86.rss.vfmadd231pd %306, %319, %339 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %341 = x86.rss.vfmadd231pd %307, %320, %339 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %342 = x86.dm.vbroadcastsd [%315 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %343 = x86.rss.vfmadd231pd %309, %319, %342 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %344 = x86.rss.vfmadd231pd %310, %320, %342 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %345 = x86.dm.vbroadcastsd [%315 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %346 = x86.rss.vfmadd231pd %312, %319, %345 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %347 = x86.rss.vfmadd231pd %313, %320, %345 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %348 = x86.dm.vbroadcastsd [%315 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %349 = x86.ri.add %315, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %350 = x86.ri.add %316, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %351 = x86.rss.vfmadd231pd %317, %319, %348 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %352 = x86.rss.vfmadd231pd %318, %320, %348 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %353 = x86.dm.vmovapd [%350] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %354 = x86.dm.vmovapd [%350 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %355 = x86.dm.vbroadcastsd [%349] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %356 = x86.rss.vfmadd231pd %322, %353, %355 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %357 = x86.rss.vfmadd231pd %323, %354, %355 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %358 = x86.dm.vbroadcastsd [%349 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %359 = x86.rss.vfmadd231pd %325, %353, %358 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %360 = x86.rss.vfmadd231pd %326, %354, %358 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %361 = x86.dm.vbroadcastsd [%349 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %362 = x86.rss.vfmadd231pd %328, %353, %361 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %363 = x86.rss.vfmadd231pd %329, %354, %361 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %364 = x86.dm.vbroadcastsd [%349 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %365 = x86.rss.vfmadd231pd %331, %353, %364 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %366 = x86.rss.vfmadd231pd %332, %354, %364 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %367 = x86.dm.vbroadcastsd [%349 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %368 = x86.rss.vfmadd231pd %334, %353, %367 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %369 = x86.rss.vfmadd231pd %335, %354, %367 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %370 = x86.dm.vbroadcastsd [%349 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %371 = x86.rss.vfmadd231pd %337, %353, %370 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %372 = x86.rss.vfmadd231pd %338, %354, %370 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %373 = x86.dm.vbroadcastsd [%349 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %374 = x86.rss.vfmadd231pd %340, %353, %373 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %375 = x86.rss.vfmadd231pd %341, %354, %373 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %376 = x86.dm.vbroadcastsd [%349 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %377 = x86.rss.vfmadd231pd %343, %353, %376 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %378 = x86.rss.vfmadd231pd %344, %354, %376 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %379 = x86.dm.vbroadcastsd [%349 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %380 = x86.rss.vfmadd231pd %346, %353, %379 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %381 = x86.rss.vfmadd231pd %347, %354, %379 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %382 = x86.dm.vbroadcastsd [%349 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %383 = x86.ri.add %349, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %384 = x86.ri.add %350, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %385 = x86.rss.vfmadd231pd %351, %353, %382 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %386 = x86.rss.vfmadd231pd %352, %354, %382 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %387 = x86.dm.vmovapd [%384] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %388 = x86.dm.vmovapd [%384 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %389 = x86.dm.vbroadcastsd [%383] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %390 = x86.rss.vfmadd231pd %356, %387, %389 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %391 = x86.rss.vfmadd231pd %357, %388, %389 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %392 = x86.dm.vbroadcastsd [%383 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %393 = x86.rss.vfmadd231pd %359, %387, %392 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %394 = x86.rss.vfmadd231pd %360, %388, %392 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %395 = x86.dm.vbroadcastsd [%383 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %396 = x86.rss.vfmadd231pd %362, %387, %395 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %397 = x86.rss.vfmadd231pd %363, %388, %395 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %398 = x86.dm.vbroadcastsd [%383 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %399 = x86.rss.vfmadd231pd %365, %387, %398 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %400 = x86.rss.vfmadd231pd %366, %388, %398 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %401 = x86.dm.vbroadcastsd [%383 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %402 = x86.rss.vfmadd231pd %368, %387, %401 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %403 = x86.rss.vfmadd231pd %369, %388, %401 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %404 = x86.dm.vbroadcastsd [%383 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %405 = x86.rss.vfmadd231pd %371, %387, %404 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %406 = x86.rss.vfmadd231pd %372, %388, %404 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %407 = x86.dm.vbroadcastsd [%383 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %408 = x86.rss.vfmadd231pd %374, %387, %407 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %409 = x86.rss.vfmadd231pd %375, %388, %407 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %410 = x86.dm.vbroadcastsd [%383 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %411 = x86.rss.vfmadd231pd %377, %387, %410 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %412 = x86.rss.vfmadd231pd %378, %388, %410 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %413 = x86.dm.vbroadcastsd [%383 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %414 = x86.rss.vfmadd231pd %380, %387, %413 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %415 = x86.rss.vfmadd231pd %381, %388, %413 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %416 = x86.dm.vbroadcastsd [%383 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %417 = x86.ri.add %383, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %418 = x86.ri.add %384, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %419 = x86.rss.vfmadd231pd %385, %387, %416 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %420 = x86.rss.vfmadd231pd %386, %388, %416 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %421 = x86.dm.vmovapd [%418] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %422 = x86.dm.vmovapd [%418 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %423 = x86.dm.vbroadcastsd [%417] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %424 = x86.rss.vfmadd231pd %390, %421, %423 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %425 = x86.rss.vfmadd231pd %391, %422, %423 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %426 = x86.dm.vbroadcastsd [%417 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %427 = x86.rss.vfmadd231pd %393, %421, %426 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %428 = x86.rss.vfmadd231pd %394, %422, %426 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %429 = x86.dm.vbroadcastsd [%417 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %430 = x86.rss.vfmadd231pd %396, %421, %429 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %431 = x86.rss.vfmadd231pd %397, %422, %429 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %432 = x86.dm.vbroadcastsd [%417 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %433 = x86.rss.vfmadd231pd %399, %421, %432 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %434 = x86.rss.vfmadd231pd %400, %422, %432 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %435 = x86.dm.vbroadcastsd [%417 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %436 = x86.rss.vfmadd231pd %402, %421, %435 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %437 = x86.rss.vfmadd231pd %403, %422, %435 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %438 = x86.dm.vbroadcastsd [%417 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %439 = x86.rss.vfmadd231pd %405, %421, %438 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %440 = x86.rss.vfmadd231pd %406, %422, %438 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %441 = x86.dm.vbroadcastsd [%417 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %442 = x86.rss.vfmadd231pd %408, %421, %441 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %443 = x86.rss.vfmadd231pd %409, %422, %441 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %444 = x86.dm.vbroadcastsd [%417 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %445 = x86.rss.vfmadd231pd %411, %421, %444 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %446 = x86.rss.vfmadd231pd %412, %422, %444 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %447 = x86.dm.vbroadcastsd [%417 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %448 = x86.rss.vfmadd231pd %414, %421, %447 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %449 = x86.rss.vfmadd231pd %415, %422, %447 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %450 = x86.dm.vbroadcastsd [%417 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %451 = x86.ri.add %417, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %452 = x86.ri.add %418, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %453 = x86.rss.vfmadd231pd %419, %421, %450 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %454 = x86.rss.vfmadd231pd %420, %422, %450 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %455 = x86.dm.vmovapd [%452] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %456 = x86.dm.vmovapd [%452 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %457 = x86.dm.vbroadcastsd [%451] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %458 = x86.rss.vfmadd231pd %424, %455, %457 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %459 = x86.rss.vfmadd231pd %425, %456, %457 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %460 = x86.dm.vbroadcastsd [%451 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %461 = x86.rss.vfmadd231pd %427, %455, %460 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %462 = x86.rss.vfmadd231pd %428, %456, %460 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %463 = x86.dm.vbroadcastsd [%451 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %464 = x86.rss.vfmadd231pd %430, %455, %463 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %465 = x86.rss.vfmadd231pd %431, %456, %463 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %466 = x86.dm.vbroadcastsd [%451 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %467 = x86.rss.vfmadd231pd %433, %455, %466 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %468 = x86.rss.vfmadd231pd %434, %456, %466 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %469 = x86.dm.vbroadcastsd [%451 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %470 = x86.rss.vfmadd231pd %436, %455, %469 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %471 = x86.rss.vfmadd231pd %437, %456, %469 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %472 = x86.dm.vbroadcastsd [%451 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %473 = x86.rss.vfmadd231pd %439, %455, %472 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %474 = x86.rss.vfmadd231pd %440, %456, %472 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %475 = x86.dm.vbroadcastsd [%451 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %476 = x86.rss.vfmadd231pd %442, %455, %475 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %477 = x86.rss.vfmadd231pd %443, %456, %475 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %478 = x86.dm.vbroadcastsd [%451 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %479 = x86.rss.vfmadd231pd %445, %455, %478 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %480 = x86.rss.vfmadd231pd %446, %456, %478 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %481 = x86.dm.vbroadcastsd [%451 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %482 = x86.rss.vfmadd231pd %448, %455, %481 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %483 = x86.rss.vfmadd231pd %449, %456, %481 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %484 = x86.dm.vbroadcastsd [%451 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %485 = x86.ri.add %451, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %486 = x86.ri.add %452, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %487 = x86.rss.vfmadd231pd %453, %455, %484 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %488 = x86.rss.vfmadd231pd %454, %456, %484 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %489 = x86.dm.vmovapd [%486] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %490 = x86.dm.vmovapd [%486 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %491 = x86.dm.vbroadcastsd [%485] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %492 = x86.rss.vfmadd231pd %458, %489, %491 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %493 = x86.rss.vfmadd231pd %459, %490, %491 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %494 = x86.dm.vbroadcastsd [%485 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %495 = x86.rss.vfmadd231pd %461, %489, %494 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %496 = x86.rss.vfmadd231pd %462, %490, %494 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %497 = x86.dm.vbroadcastsd [%485 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %498 = x86.rss.vfmadd231pd %464, %489, %497 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %499 = x86.rss.vfmadd231pd %465, %490, %497 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %500 = x86.dm.vbroadcastsd [%485 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %501 = x86.rss.vfmadd231pd %467, %489, %500 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %502 = x86.rss.vfmadd231pd %468, %490, %500 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %503 = x86.dm.vbroadcastsd [%485 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %504 = x86.rss.vfmadd231pd %470, %489, %503 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %505 = x86.rss.vfmadd231pd %471, %490, %503 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %506 = x86.dm.vbroadcastsd [%485 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %507 = x86.rss.vfmadd231pd %473, %489, %506 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %508 = x86.rss.vfmadd231pd %474, %490, %506 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %509 = x86.dm.vbroadcastsd [%485 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %510 = x86.rss.vfmadd231pd %476, %489, %509 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %511 = x86.rss.vfmadd231pd %477, %490, %509 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %512 = x86.dm.vbroadcastsd [%485 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %513 = x86.rss.vfmadd231pd %479, %489, %512 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %514 = x86.rss.vfmadd231pd %480, %490, %512 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %515 = x86.dm.vbroadcastsd [%485 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %516 = x86.rss.vfmadd231pd %482, %489, %515 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %517 = x86.rss.vfmadd231pd %483, %490, %515 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %518 = x86.dm.vbroadcastsd [%485 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %519 = x86.ri.add %485, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %520 = x86.ri.add %486, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %521 = x86.rss.vfmadd231pd %487, %489, %518 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %522 = x86.rss.vfmadd231pd %488, %490, %518 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %523 = x86.dm.vmovapd [%520] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %524 = x86.dm.vmovapd [%520 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %525 = x86.dm.vbroadcastsd [%519] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %526 = x86.rss.vfmadd231pd %492, %523, %525 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %527 = x86.rss.vfmadd231pd %493, %524, %525 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %528 = x86.dm.vbroadcastsd [%519 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %529 = x86.rss.vfmadd231pd %495, %523, %528 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %530 = x86.rss.vfmadd231pd %496, %524, %528 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %531 = x86.dm.vbroadcastsd [%519 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %532 = x86.rss.vfmadd231pd %498, %523, %531 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %533 = x86.rss.vfmadd231pd %499, %524, %531 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %534 = x86.dm.vbroadcastsd [%519 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %535 = x86.rss.vfmadd231pd %501, %523, %534 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %536 = x86.rss.vfmadd231pd %502, %524, %534 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %537 = x86.dm.vbroadcastsd [%519 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %538 = x86.rss.vfmadd231pd %504, %523, %537 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %539 = x86.rss.vfmadd231pd %505, %524, %537 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %540 = x86.dm.vbroadcastsd [%519 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %541 = x86.rss.vfmadd231pd %507, %523, %540 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %542 = x86.rss.vfmadd231pd %508, %524, %540 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %543 = x86.dm.vbroadcastsd [%519 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %544 = x86.rss.vfmadd231pd %510, %523, %543 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %545 = x86.rss.vfmadd231pd %511, %524, %543 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %546 = x86.dm.vbroadcastsd [%519 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %547 = x86.rss.vfmadd231pd %513, %523, %546 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %548 = x86.rss.vfmadd231pd %514, %524, %546 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %549 = x86.dm.vbroadcastsd [%519 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %550 = x86.rss.vfmadd231pd %516, %523, %549 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %551 = x86.rss.vfmadd231pd %517, %524, %549 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %552 = x86.dm.vbroadcastsd [%519 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %553 = x86.ri.add %519, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %554 = x86.ri.add %520, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %555 = x86.rss.vfmadd231pd %521, %523, %552 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %556 = x86.rss.vfmadd231pd %522, %524, %552 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %557 = x86.dm.vmovapd [%554] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %558 = x86.dm.vmovapd [%554 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %559 = x86.dm.vbroadcastsd [%553] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %560 = x86.rss.vfmadd231pd %526, %557, %559 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %561 = x86.rss.vfmadd231pd %527, %558, %559 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %562 = x86.dm.vbroadcastsd [%553 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %563 = x86.rss.vfmadd231pd %529, %557, %562 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %564 = x86.rss.vfmadd231pd %530, %558, %562 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %565 = x86.dm.vbroadcastsd [%553 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %566 = x86.rss.vfmadd231pd %532, %557, %565 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %567 = x86.rss.vfmadd231pd %533, %558, %565 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %568 = x86.dm.vbroadcastsd [%553 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %569 = x86.rss.vfmadd231pd %535, %557, %568 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %570 = x86.rss.vfmadd231pd %536, %558, %568 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %571 = x86.dm.vbroadcastsd [%553 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %572 = x86.rss.vfmadd231pd %538, %557, %571 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %573 = x86.rss.vfmadd231pd %539, %558, %571 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %574 = x86.dm.vbroadcastsd [%553 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %575 = x86.rss.vfmadd231pd %541, %557, %574 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %576 = x86.rss.vfmadd231pd %542, %558, %574 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %577 = x86.dm.vbroadcastsd [%553 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %578 = x86.rss.vfmadd231pd %544, %557, %577 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %579 = x86.rss.vfmadd231pd %545, %558, %577 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %580 = x86.dm.vbroadcastsd [%553 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %581 = x86.rss.vfmadd231pd %547, %557, %580 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %582 = x86.rss.vfmadd231pd %548, %558, %580 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %583 = x86.dm.vbroadcastsd [%553 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %584 = x86.rss.vfmadd231pd %550, %557, %583 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %585 = x86.rss.vfmadd231pd %551, %558, %583 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %586 = x86.dm.vbroadcastsd [%553 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %587 = x86.ri.add %553, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %588 = x86.ri.add %554, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %589 = x86.rss.vfmadd231pd %555, %557, %586 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %590 = x86.rss.vfmadd231pd %556, %558, %586 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %591 = x86.ri.sub %587, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%21], %560 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 64], %561 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 128], %563 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 192], %564 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 256], %566 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 320], %567 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 384], %569 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 448], %570 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 512], %572 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 576], %573 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 640], %575 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 704], %576 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 768], %578 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 832], %579 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 896], %581 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 960], %582 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 1024], %584 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 1088], %585 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 1152], %589 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 1216], %590 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %592 = x86.ri.add %21, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %593 = x86.ri.sub %588, 1920 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %594 = x86.si.cmp %26, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %594 : !x86.rflags<rflags>, ^bb2(%593 : !x86.reg64<rdi>, %591 : !x86.reg64<rsi>, %592 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>), ^bb3(%593 : !x86.reg64<rdi>, %591 : !x86.reg64<rsi>, %592 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb3(%595: !x86.reg64<rdi>, %596: !x86.reg64<rsi>, %597: !x86.reg64<rdx>, %598: !x86.reg64<rbp>, %599: !x86.reg64<rsp>, %600: !x86.reg64<r11>, %601: !x86.reg64<r10>):
// CHECK-NEXT:      %602 = x86.ri.add %597, 1152 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %603 = x86.ri.add %596, 1280 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %604 = x86.ri.sub %595, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %605 = x86.si.cmp %600, 20 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %605 : !x86.rflags<rflags>, ^bb1(%604 : !x86.reg64<rdi>, %603 : !x86.reg64<rsi>, %602 : !x86.reg64<rdx>, %598 : !x86.reg64<rbp>, %599 : !x86.reg64<rsp>, %600 : !x86.reg64<r11>), ^bb4(%604 : !x86.reg64<rdi>, %603 : !x86.reg64<rsi>, %602 : !x86.reg64<rdx>, %598 : !x86.reg64<rbp>, %599 : !x86.reg64<rsp>, %600 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb4(%606: !x86.reg64<rdi>, %607: !x86.reg64<rsi>, %608: !x86.reg64<rdx>, %609: !x86.reg64<rbp>, %610: !x86.reg64<rsp>, %611: !x86.reg64<r11>):
// CHECK-NEXT:      %612 = x86.di.mov 20 : () -> !x86.reg64<r11>
// CHECK-NEXT:      x86.fallthrough ^bb5(%606 : !x86.reg64<rdi>, %607 : !x86.reg64<rsi>, %608 : !x86.reg64<rdx>, %609 : !x86.reg64<rbp>, %610 : !x86.reg64<rsp>, %612 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb5(%613: !x86.reg64<rdi>, %614: !x86.reg64<rsi>, %615: !x86.reg64<rdx>, %616: !x86.reg64<rbp>, %617: !x86.reg64<rsp>, %618: !x86.reg64<r11>):
// CHECK-NEXT:      x86.label "l35"
// CHECK-NEXT:      %619 = x86.ri.add %618, 9 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %620 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb6(%613 : !x86.reg64<rdi>, %614 : !x86.reg64<rsi>, %615 : !x86.reg64<rdx>, %616 : !x86.reg64<rbp>, %617 : !x86.reg64<rsp>, %619 : !x86.reg64<r11>, %620 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb6(%621: !x86.reg64<rdi>, %622: !x86.reg64<rsi>, %623: !x86.reg64<rdx>, %624: !x86.reg64<rbp>, %625: !x86.reg64<rsp>, %626: !x86.reg64<r11>, %627: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l36"
// CHECK-NEXT:      %628 = x86.ri.add %627, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %629 = x86.dm.vmovapd [%623] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %630 = x86.dm.vmovapd [%623 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %631 = x86.dm.vmovapd [%623 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %632 = x86.dm.vmovapd [%623 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %633 = x86.dm.vmovapd [%623 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %634 = x86.dm.vmovapd [%623 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %635 = x86.dm.vmovapd [%623 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %636 = x86.dm.vmovapd [%623 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %637 = x86.dm.vmovapd [%623 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %638 = x86.dm.vmovapd [%623 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %639 = x86.dm.vmovapd [%623 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %640 = x86.dm.vmovapd [%623 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %641 = x86.dm.vmovapd [%623 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %642 = x86.dm.vmovapd [%623 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %643 = x86.dm.vmovapd [%623 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %644 = x86.dm.vmovapd [%623 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %645 = x86.dm.vmovapd [%623 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %646 = x86.dm.vmovapd [%623 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %647 = x86.dm.vmovapd [%621] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %648 = x86.dm.vmovapd [%621 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %649 = x86.dm.vbroadcastsd [%622] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %650 = x86.rss.vfmadd231pd %629, %647, %649 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %651 = x86.rss.vfmadd231pd %630, %648, %649 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %652 = x86.dm.vbroadcastsd [%622 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %653 = x86.rss.vfmadd231pd %631, %647, %652 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %654 = x86.rss.vfmadd231pd %632, %648, %652 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %655 = x86.dm.vbroadcastsd [%622 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %656 = x86.rss.vfmadd231pd %633, %647, %655 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %657 = x86.rss.vfmadd231pd %634, %648, %655 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %658 = x86.dm.vbroadcastsd [%622 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %659 = x86.rss.vfmadd231pd %635, %647, %658 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %660 = x86.rss.vfmadd231pd %636, %648, %658 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %661 = x86.dm.vbroadcastsd [%622 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %662 = x86.rss.vfmadd231pd %637, %647, %661 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %663 = x86.rss.vfmadd231pd %638, %648, %661 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %664 = x86.dm.vbroadcastsd [%622 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %665 = x86.rss.vfmadd231pd %639, %647, %664 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %666 = x86.rss.vfmadd231pd %640, %648, %664 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %667 = x86.dm.vbroadcastsd [%622 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %668 = x86.rss.vfmadd231pd %641, %647, %667 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %669 = x86.rss.vfmadd231pd %642, %648, %667 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %670 = x86.dm.vbroadcastsd [%622 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %671 = x86.rss.vfmadd231pd %643, %647, %670 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %672 = x86.rss.vfmadd231pd %644, %648, %670 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %673 = x86.dm.vbroadcastsd [%622 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %674 = x86.ri.add %622, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %675 = x86.ri.add %621, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %676 = x86.rss.vfmadd231pd %645, %647, %673 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %677 = x86.rss.vfmadd231pd %646, %648, %673 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %678 = x86.dm.vmovapd [%675] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %679 = x86.dm.vmovapd [%675 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %680 = x86.dm.vbroadcastsd [%674] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %681 = x86.rss.vfmadd231pd %650, %678, %680 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %682 = x86.rss.vfmadd231pd %651, %679, %680 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %683 = x86.dm.vbroadcastsd [%674 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %684 = x86.rss.vfmadd231pd %653, %678, %683 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %685 = x86.rss.vfmadd231pd %654, %679, %683 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %686 = x86.dm.vbroadcastsd [%674 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %687 = x86.rss.vfmadd231pd %656, %678, %686 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %688 = x86.rss.vfmadd231pd %657, %679, %686 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %689 = x86.dm.vbroadcastsd [%674 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %690 = x86.rss.vfmadd231pd %659, %678, %689 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %691 = x86.rss.vfmadd231pd %660, %679, %689 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %692 = x86.dm.vbroadcastsd [%674 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %693 = x86.rss.vfmadd231pd %662, %678, %692 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %694 = x86.rss.vfmadd231pd %663, %679, %692 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %695 = x86.dm.vbroadcastsd [%674 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %696 = x86.rss.vfmadd231pd %665, %678, %695 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %697 = x86.rss.vfmadd231pd %666, %679, %695 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %698 = x86.dm.vbroadcastsd [%674 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %699 = x86.rss.vfmadd231pd %668, %678, %698 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %700 = x86.rss.vfmadd231pd %669, %679, %698 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %701 = x86.dm.vbroadcastsd [%674 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %702 = x86.rss.vfmadd231pd %671, %678, %701 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %703 = x86.rss.vfmadd231pd %672, %679, %701 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %704 = x86.dm.vbroadcastsd [%674 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %705 = x86.ri.add %674, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %706 = x86.ri.add %675, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %707 = x86.rss.vfmadd231pd %676, %678, %704 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %708 = x86.rss.vfmadd231pd %677, %679, %704 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %709 = x86.dm.vmovapd [%706] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %710 = x86.dm.vmovapd [%706 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %711 = x86.dm.vbroadcastsd [%705] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %712 = x86.rss.vfmadd231pd %681, %709, %711 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %713 = x86.rss.vfmadd231pd %682, %710, %711 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %714 = x86.dm.vbroadcastsd [%705 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %715 = x86.rss.vfmadd231pd %684, %709, %714 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %716 = x86.rss.vfmadd231pd %685, %710, %714 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %717 = x86.dm.vbroadcastsd [%705 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %718 = x86.rss.vfmadd231pd %687, %709, %717 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %719 = x86.rss.vfmadd231pd %688, %710, %717 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %720 = x86.dm.vbroadcastsd [%705 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %721 = x86.rss.vfmadd231pd %690, %709, %720 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %722 = x86.rss.vfmadd231pd %691, %710, %720 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %723 = x86.dm.vbroadcastsd [%705 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %724 = x86.rss.vfmadd231pd %693, %709, %723 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %725 = x86.rss.vfmadd231pd %694, %710, %723 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %726 = x86.dm.vbroadcastsd [%705 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %727 = x86.rss.vfmadd231pd %696, %709, %726 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %728 = x86.rss.vfmadd231pd %697, %710, %726 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %729 = x86.dm.vbroadcastsd [%705 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %730 = x86.rss.vfmadd231pd %699, %709, %729 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %731 = x86.rss.vfmadd231pd %700, %710, %729 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %732 = x86.dm.vbroadcastsd [%705 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %733 = x86.rss.vfmadd231pd %702, %709, %732 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %734 = x86.rss.vfmadd231pd %703, %710, %732 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %735 = x86.dm.vbroadcastsd [%705 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %736 = x86.ri.add %705, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %737 = x86.ri.add %706, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %738 = x86.rss.vfmadd231pd %707, %709, %735 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %739 = x86.rss.vfmadd231pd %708, %710, %735 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %740 = x86.dm.vmovapd [%737] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %741 = x86.dm.vmovapd [%737 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %742 = x86.dm.vbroadcastsd [%736] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %743 = x86.rss.vfmadd231pd %712, %740, %742 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %744 = x86.rss.vfmadd231pd %713, %741, %742 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %745 = x86.dm.vbroadcastsd [%736 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %746 = x86.rss.vfmadd231pd %715, %740, %745 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %747 = x86.rss.vfmadd231pd %716, %741, %745 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %748 = x86.dm.vbroadcastsd [%736 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %749 = x86.rss.vfmadd231pd %718, %740, %748 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %750 = x86.rss.vfmadd231pd %719, %741, %748 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %751 = x86.dm.vbroadcastsd [%736 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %752 = x86.rss.vfmadd231pd %721, %740, %751 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %753 = x86.rss.vfmadd231pd %722, %741, %751 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %754 = x86.dm.vbroadcastsd [%736 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %755 = x86.rss.vfmadd231pd %724, %740, %754 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %756 = x86.rss.vfmadd231pd %725, %741, %754 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %757 = x86.dm.vbroadcastsd [%736 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %758 = x86.rss.vfmadd231pd %727, %740, %757 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %759 = x86.rss.vfmadd231pd %728, %741, %757 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %760 = x86.dm.vbroadcastsd [%736 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %761 = x86.rss.vfmadd231pd %730, %740, %760 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %762 = x86.rss.vfmadd231pd %731, %741, %760 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %763 = x86.dm.vbroadcastsd [%736 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %764 = x86.rss.vfmadd231pd %733, %740, %763 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %765 = x86.rss.vfmadd231pd %734, %741, %763 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %766 = x86.dm.vbroadcastsd [%736 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %767 = x86.ri.add %736, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %768 = x86.ri.add %737, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %769 = x86.rss.vfmadd231pd %738, %740, %766 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %770 = x86.rss.vfmadd231pd %739, %741, %766 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %771 = x86.dm.vmovapd [%768] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %772 = x86.dm.vmovapd [%768 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %773 = x86.dm.vbroadcastsd [%767] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %774 = x86.rss.vfmadd231pd %743, %771, %773 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %775 = x86.rss.vfmadd231pd %744, %772, %773 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %776 = x86.dm.vbroadcastsd [%767 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %777 = x86.rss.vfmadd231pd %746, %771, %776 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %778 = x86.rss.vfmadd231pd %747, %772, %776 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %779 = x86.dm.vbroadcastsd [%767 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %780 = x86.rss.vfmadd231pd %749, %771, %779 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %781 = x86.rss.vfmadd231pd %750, %772, %779 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %782 = x86.dm.vbroadcastsd [%767 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %783 = x86.rss.vfmadd231pd %752, %771, %782 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %784 = x86.rss.vfmadd231pd %753, %772, %782 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %785 = x86.dm.vbroadcastsd [%767 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %786 = x86.rss.vfmadd231pd %755, %771, %785 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %787 = x86.rss.vfmadd231pd %756, %772, %785 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %788 = x86.dm.vbroadcastsd [%767 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %789 = x86.rss.vfmadd231pd %758, %771, %788 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %790 = x86.rss.vfmadd231pd %759, %772, %788 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %791 = x86.dm.vbroadcastsd [%767 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %792 = x86.rss.vfmadd231pd %761, %771, %791 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %793 = x86.rss.vfmadd231pd %762, %772, %791 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %794 = x86.dm.vbroadcastsd [%767 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %795 = x86.rss.vfmadd231pd %764, %771, %794 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %796 = x86.rss.vfmadd231pd %765, %772, %794 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %797 = x86.dm.vbroadcastsd [%767 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %798 = x86.ri.add %767, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %799 = x86.ri.add %768, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %800 = x86.rss.vfmadd231pd %769, %771, %797 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %801 = x86.rss.vfmadd231pd %770, %772, %797 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %802 = x86.dm.vmovapd [%799] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %803 = x86.dm.vmovapd [%799 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %804 = x86.dm.vbroadcastsd [%798] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %805 = x86.rss.vfmadd231pd %774, %802, %804 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %806 = x86.rss.vfmadd231pd %775, %803, %804 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %807 = x86.dm.vbroadcastsd [%798 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %808 = x86.rss.vfmadd231pd %777, %802, %807 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %809 = x86.rss.vfmadd231pd %778, %803, %807 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %810 = x86.dm.vbroadcastsd [%798 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %811 = x86.rss.vfmadd231pd %780, %802, %810 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %812 = x86.rss.vfmadd231pd %781, %803, %810 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %813 = x86.dm.vbroadcastsd [%798 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %814 = x86.rss.vfmadd231pd %783, %802, %813 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %815 = x86.rss.vfmadd231pd %784, %803, %813 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %816 = x86.dm.vbroadcastsd [%798 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %817 = x86.rss.vfmadd231pd %786, %802, %816 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %818 = x86.rss.vfmadd231pd %787, %803, %816 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %819 = x86.dm.vbroadcastsd [%798 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %820 = x86.rss.vfmadd231pd %789, %802, %819 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %821 = x86.rss.vfmadd231pd %790, %803, %819 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %822 = x86.dm.vbroadcastsd [%798 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %823 = x86.rss.vfmadd231pd %792, %802, %822 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %824 = x86.rss.vfmadd231pd %793, %803, %822 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %825 = x86.dm.vbroadcastsd [%798 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %826 = x86.rss.vfmadd231pd %795, %802, %825 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %827 = x86.rss.vfmadd231pd %796, %803, %825 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %828 = x86.dm.vbroadcastsd [%798 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %829 = x86.ri.add %798, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %830 = x86.ri.add %799, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %831 = x86.rss.vfmadd231pd %800, %802, %828 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %832 = x86.rss.vfmadd231pd %801, %803, %828 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %833 = x86.dm.vmovapd [%830] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %834 = x86.dm.vmovapd [%830 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %835 = x86.dm.vbroadcastsd [%829] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %836 = x86.rss.vfmadd231pd %805, %833, %835 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %837 = x86.rss.vfmadd231pd %806, %834, %835 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %838 = x86.dm.vbroadcastsd [%829 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %839 = x86.rss.vfmadd231pd %808, %833, %838 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %840 = x86.rss.vfmadd231pd %809, %834, %838 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %841 = x86.dm.vbroadcastsd [%829 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %842 = x86.rss.vfmadd231pd %811, %833, %841 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %843 = x86.rss.vfmadd231pd %812, %834, %841 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %844 = x86.dm.vbroadcastsd [%829 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %845 = x86.rss.vfmadd231pd %814, %833, %844 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %846 = x86.rss.vfmadd231pd %815, %834, %844 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %847 = x86.dm.vbroadcastsd [%829 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %848 = x86.rss.vfmadd231pd %817, %833, %847 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %849 = x86.rss.vfmadd231pd %818, %834, %847 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %850 = x86.dm.vbroadcastsd [%829 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %851 = x86.rss.vfmadd231pd %820, %833, %850 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %852 = x86.rss.vfmadd231pd %821, %834, %850 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %853 = x86.dm.vbroadcastsd [%829 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %854 = x86.rss.vfmadd231pd %823, %833, %853 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %855 = x86.rss.vfmadd231pd %824, %834, %853 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %856 = x86.dm.vbroadcastsd [%829 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %857 = x86.rss.vfmadd231pd %826, %833, %856 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %858 = x86.rss.vfmadd231pd %827, %834, %856 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %859 = x86.dm.vbroadcastsd [%829 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %860 = x86.ri.add %829, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %861 = x86.ri.add %830, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %862 = x86.rss.vfmadd231pd %831, %833, %859 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %863 = x86.rss.vfmadd231pd %832, %834, %859 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %864 = x86.dm.vmovapd [%861] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %865 = x86.dm.vmovapd [%861 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %866 = x86.dm.vbroadcastsd [%860] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %867 = x86.rss.vfmadd231pd %836, %864, %866 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %868 = x86.rss.vfmadd231pd %837, %865, %866 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %869 = x86.dm.vbroadcastsd [%860 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %870 = x86.rss.vfmadd231pd %839, %864, %869 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %871 = x86.rss.vfmadd231pd %840, %865, %869 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %872 = x86.dm.vbroadcastsd [%860 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %873 = x86.rss.vfmadd231pd %842, %864, %872 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %874 = x86.rss.vfmadd231pd %843, %865, %872 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %875 = x86.dm.vbroadcastsd [%860 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %876 = x86.rss.vfmadd231pd %845, %864, %875 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %877 = x86.rss.vfmadd231pd %846, %865, %875 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %878 = x86.dm.vbroadcastsd [%860 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %879 = x86.rss.vfmadd231pd %848, %864, %878 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %880 = x86.rss.vfmadd231pd %849, %865, %878 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %881 = x86.dm.vbroadcastsd [%860 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %882 = x86.rss.vfmadd231pd %851, %864, %881 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %883 = x86.rss.vfmadd231pd %852, %865, %881 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %884 = x86.dm.vbroadcastsd [%860 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %885 = x86.rss.vfmadd231pd %854, %864, %884 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %886 = x86.rss.vfmadd231pd %855, %865, %884 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %887 = x86.dm.vbroadcastsd [%860 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %888 = x86.rss.vfmadd231pd %857, %864, %887 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %889 = x86.rss.vfmadd231pd %858, %865, %887 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %890 = x86.dm.vbroadcastsd [%860 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %891 = x86.ri.add %860, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %892 = x86.ri.add %861, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %893 = x86.rss.vfmadd231pd %862, %864, %890 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %894 = x86.rss.vfmadd231pd %863, %865, %890 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %895 = x86.dm.vmovapd [%892] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %896 = x86.dm.vmovapd [%892 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %897 = x86.dm.vbroadcastsd [%891] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %898 = x86.rss.vfmadd231pd %867, %895, %897 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %899 = x86.rss.vfmadd231pd %868, %896, %897 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %900 = x86.dm.vbroadcastsd [%891 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %901 = x86.rss.vfmadd231pd %870, %895, %900 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %902 = x86.rss.vfmadd231pd %871, %896, %900 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %903 = x86.dm.vbroadcastsd [%891 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %904 = x86.rss.vfmadd231pd %873, %895, %903 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %905 = x86.rss.vfmadd231pd %874, %896, %903 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %906 = x86.dm.vbroadcastsd [%891 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %907 = x86.rss.vfmadd231pd %876, %895, %906 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %908 = x86.rss.vfmadd231pd %877, %896, %906 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %909 = x86.dm.vbroadcastsd [%891 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %910 = x86.rss.vfmadd231pd %879, %895, %909 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %911 = x86.rss.vfmadd231pd %880, %896, %909 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %912 = x86.dm.vbroadcastsd [%891 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %913 = x86.rss.vfmadd231pd %882, %895, %912 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %914 = x86.rss.vfmadd231pd %883, %896, %912 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %915 = x86.dm.vbroadcastsd [%891 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %916 = x86.rss.vfmadd231pd %885, %895, %915 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %917 = x86.rss.vfmadd231pd %886, %896, %915 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %918 = x86.dm.vbroadcastsd [%891 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %919 = x86.rss.vfmadd231pd %888, %895, %918 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %920 = x86.rss.vfmadd231pd %889, %896, %918 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %921 = x86.dm.vbroadcastsd [%891 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %922 = x86.ri.add %891, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %923 = x86.ri.add %892, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %924 = x86.rss.vfmadd231pd %893, %895, %921 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %925 = x86.rss.vfmadd231pd %894, %896, %921 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %926 = x86.dm.vmovapd [%923] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %927 = x86.dm.vmovapd [%923 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %928 = x86.dm.vbroadcastsd [%922] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %929 = x86.rss.vfmadd231pd %898, %926, %928 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %930 = x86.rss.vfmadd231pd %899, %927, %928 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %931 = x86.dm.vbroadcastsd [%922 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %932 = x86.rss.vfmadd231pd %901, %926, %931 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %933 = x86.rss.vfmadd231pd %902, %927, %931 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %934 = x86.dm.vbroadcastsd [%922 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %935 = x86.rss.vfmadd231pd %904, %926, %934 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %936 = x86.rss.vfmadd231pd %905, %927, %934 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %937 = x86.dm.vbroadcastsd [%922 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %938 = x86.rss.vfmadd231pd %907, %926, %937 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %939 = x86.rss.vfmadd231pd %908, %927, %937 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %940 = x86.dm.vbroadcastsd [%922 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %941 = x86.rss.vfmadd231pd %910, %926, %940 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %942 = x86.rss.vfmadd231pd %911, %927, %940 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %943 = x86.dm.vbroadcastsd [%922 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %944 = x86.rss.vfmadd231pd %913, %926, %943 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %945 = x86.rss.vfmadd231pd %914, %927, %943 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %946 = x86.dm.vbroadcastsd [%922 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %947 = x86.rss.vfmadd231pd %916, %926, %946 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %948 = x86.rss.vfmadd231pd %917, %927, %946 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %949 = x86.dm.vbroadcastsd [%922 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %950 = x86.rss.vfmadd231pd %919, %926, %949 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %951 = x86.rss.vfmadd231pd %920, %927, %949 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %952 = x86.dm.vbroadcastsd [%922 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %953 = x86.ri.add %922, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %954 = x86.ri.add %923, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %955 = x86.rss.vfmadd231pd %924, %926, %952 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %956 = x86.rss.vfmadd231pd %925, %927, %952 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %957 = x86.dm.vmovapd [%954] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %958 = x86.dm.vmovapd [%954 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %959 = x86.dm.vbroadcastsd [%953] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %960 = x86.rss.vfmadd231pd %929, %957, %959 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %961 = x86.rss.vfmadd231pd %930, %958, %959 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %962 = x86.dm.vbroadcastsd [%953 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %963 = x86.rss.vfmadd231pd %932, %957, %962 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %964 = x86.rss.vfmadd231pd %933, %958, %962 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %965 = x86.dm.vbroadcastsd [%953 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %966 = x86.rss.vfmadd231pd %935, %957, %965 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %967 = x86.rss.vfmadd231pd %936, %958, %965 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %968 = x86.dm.vbroadcastsd [%953 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %969 = x86.rss.vfmadd231pd %938, %957, %968 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %970 = x86.rss.vfmadd231pd %939, %958, %968 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %971 = x86.dm.vbroadcastsd [%953 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %972 = x86.rss.vfmadd231pd %941, %957, %971 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %973 = x86.rss.vfmadd231pd %942, %958, %971 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %974 = x86.dm.vbroadcastsd [%953 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %975 = x86.rss.vfmadd231pd %944, %957, %974 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %976 = x86.rss.vfmadd231pd %945, %958, %974 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %977 = x86.dm.vbroadcastsd [%953 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %978 = x86.rss.vfmadd231pd %947, %957, %977 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %979 = x86.rss.vfmadd231pd %948, %958, %977 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %980 = x86.dm.vbroadcastsd [%953 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %981 = x86.rss.vfmadd231pd %950, %957, %980 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %982 = x86.rss.vfmadd231pd %951, %958, %980 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %983 = x86.dm.vbroadcastsd [%953 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %984 = x86.ri.add %953, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %985 = x86.ri.add %954, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %986 = x86.rss.vfmadd231pd %955, %957, %983 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %987 = x86.rss.vfmadd231pd %956, %958, %983 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %988 = x86.dm.vmovapd [%985] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %989 = x86.dm.vmovapd [%985 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %990 = x86.dm.vbroadcastsd [%984] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %991 = x86.rss.vfmadd231pd %960, %988, %990 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %992 = x86.rss.vfmadd231pd %961, %989, %990 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %993 = x86.dm.vbroadcastsd [%984 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %994 = x86.rss.vfmadd231pd %963, %988, %993 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %995 = x86.rss.vfmadd231pd %964, %989, %993 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %996 = x86.dm.vbroadcastsd [%984 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %997 = x86.rss.vfmadd231pd %966, %988, %996 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %998 = x86.rss.vfmadd231pd %967, %989, %996 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %999 = x86.dm.vbroadcastsd [%984 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1000 = x86.rss.vfmadd231pd %969, %988, %999 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %1001 = x86.rss.vfmadd231pd %970, %989, %999 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %1002 = x86.dm.vbroadcastsd [%984 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1003 = x86.rss.vfmadd231pd %972, %988, %1002 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %1004 = x86.rss.vfmadd231pd %973, %989, %1002 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %1005 = x86.dm.vbroadcastsd [%984 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1006 = x86.rss.vfmadd231pd %975, %988, %1005 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %1007 = x86.rss.vfmadd231pd %976, %989, %1005 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %1008 = x86.dm.vbroadcastsd [%984 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1009 = x86.rss.vfmadd231pd %978, %988, %1008 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %1010 = x86.rss.vfmadd231pd %979, %989, %1008 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %1011 = x86.dm.vbroadcastsd [%984 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1012 = x86.rss.vfmadd231pd %981, %988, %1011 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %1013 = x86.rss.vfmadd231pd %982, %989, %1011 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %1014 = x86.dm.vbroadcastsd [%984 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1015 = x86.ri.add %984, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %1016 = x86.ri.add %985, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1017 = x86.rss.vfmadd231pd %986, %988, %1014 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1018 = x86.rss.vfmadd231pd %987, %989, %1014 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %1019 = x86.dm.vmovapd [%1016] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %1020 = x86.dm.vmovapd [%1016 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %1021 = x86.dm.vbroadcastsd [%1015] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1022 = x86.rss.vfmadd231pd %991, %1019, %1021 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %1023 = x86.rss.vfmadd231pd %992, %1020, %1021 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %1024 = x86.dm.vbroadcastsd [%1015 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1025 = x86.rss.vfmadd231pd %994, %1019, %1024 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %1026 = x86.rss.vfmadd231pd %995, %1020, %1024 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %1027 = x86.dm.vbroadcastsd [%1015 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1028 = x86.rss.vfmadd231pd %997, %1019, %1027 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %1029 = x86.rss.vfmadd231pd %998, %1020, %1027 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %1030 = x86.dm.vbroadcastsd [%1015 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1031 = x86.rss.vfmadd231pd %1000, %1019, %1030 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %1032 = x86.rss.vfmadd231pd %1001, %1020, %1030 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %1033 = x86.dm.vbroadcastsd [%1015 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1034 = x86.rss.vfmadd231pd %1003, %1019, %1033 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %1035 = x86.rss.vfmadd231pd %1004, %1020, %1033 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %1036 = x86.dm.vbroadcastsd [%1015 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1037 = x86.rss.vfmadd231pd %1006, %1019, %1036 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %1038 = x86.rss.vfmadd231pd %1007, %1020, %1036 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %1039 = x86.dm.vbroadcastsd [%1015 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1040 = x86.rss.vfmadd231pd %1009, %1019, %1039 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %1041 = x86.rss.vfmadd231pd %1010, %1020, %1039 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %1042 = x86.dm.vbroadcastsd [%1015 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1043 = x86.rss.vfmadd231pd %1012, %1019, %1042 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %1044 = x86.rss.vfmadd231pd %1013, %1020, %1042 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %1045 = x86.dm.vbroadcastsd [%1015 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1046 = x86.ri.add %1015, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %1047 = x86.ri.add %1016, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1048 = x86.rss.vfmadd231pd %1017, %1019, %1045 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1049 = x86.rss.vfmadd231pd %1018, %1020, %1045 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %1050 = x86.dm.vmovapd [%1047] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %1051 = x86.dm.vmovapd [%1047 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %1052 = x86.dm.vbroadcastsd [%1046] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1053 = x86.rss.vfmadd231pd %1022, %1050, %1052 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %1054 = x86.rss.vfmadd231pd %1023, %1051, %1052 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %1055 = x86.dm.vbroadcastsd [%1046 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1056 = x86.rss.vfmadd231pd %1025, %1050, %1055 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %1057 = x86.rss.vfmadd231pd %1026, %1051, %1055 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %1058 = x86.dm.vbroadcastsd [%1046 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1059 = x86.rss.vfmadd231pd %1028, %1050, %1058 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %1060 = x86.rss.vfmadd231pd %1029, %1051, %1058 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %1061 = x86.dm.vbroadcastsd [%1046 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1062 = x86.rss.vfmadd231pd %1031, %1050, %1061 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %1063 = x86.rss.vfmadd231pd %1032, %1051, %1061 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %1064 = x86.dm.vbroadcastsd [%1046 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1065 = x86.rss.vfmadd231pd %1034, %1050, %1064 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %1066 = x86.rss.vfmadd231pd %1035, %1051, %1064 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %1067 = x86.dm.vbroadcastsd [%1046 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1068 = x86.rss.vfmadd231pd %1037, %1050, %1067 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %1069 = x86.rss.vfmadd231pd %1038, %1051, %1067 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %1070 = x86.dm.vbroadcastsd [%1046 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1071 = x86.rss.vfmadd231pd %1040, %1050, %1070 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %1072 = x86.rss.vfmadd231pd %1041, %1051, %1070 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %1073 = x86.dm.vbroadcastsd [%1046 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1074 = x86.rss.vfmadd231pd %1043, %1050, %1073 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %1075 = x86.rss.vfmadd231pd %1044, %1051, %1073 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %1076 = x86.dm.vbroadcastsd [%1046 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1077 = x86.ri.add %1046, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %1078 = x86.ri.add %1047, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1079 = x86.rss.vfmadd231pd %1048, %1050, %1076 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1080 = x86.rss.vfmadd231pd %1049, %1051, %1076 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %1081 = x86.dm.vmovapd [%1078] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %1082 = x86.dm.vmovapd [%1078 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %1083 = x86.dm.vbroadcastsd [%1077] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1084 = x86.rss.vfmadd231pd %1053, %1081, %1083 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %1085 = x86.rss.vfmadd231pd %1054, %1082, %1083 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %1086 = x86.dm.vbroadcastsd [%1077 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1087 = x86.rss.vfmadd231pd %1056, %1081, %1086 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %1088 = x86.rss.vfmadd231pd %1057, %1082, %1086 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %1089 = x86.dm.vbroadcastsd [%1077 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1090 = x86.rss.vfmadd231pd %1059, %1081, %1089 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %1091 = x86.rss.vfmadd231pd %1060, %1082, %1089 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %1092 = x86.dm.vbroadcastsd [%1077 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1093 = x86.rss.vfmadd231pd %1062, %1081, %1092 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %1094 = x86.rss.vfmadd231pd %1063, %1082, %1092 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %1095 = x86.dm.vbroadcastsd [%1077 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1096 = x86.rss.vfmadd231pd %1065, %1081, %1095 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %1097 = x86.rss.vfmadd231pd %1066, %1082, %1095 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %1098 = x86.dm.vbroadcastsd [%1077 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1099 = x86.rss.vfmadd231pd %1068, %1081, %1098 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %1100 = x86.rss.vfmadd231pd %1069, %1082, %1098 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %1101 = x86.dm.vbroadcastsd [%1077 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1102 = x86.rss.vfmadd231pd %1071, %1081, %1101 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %1103 = x86.rss.vfmadd231pd %1072, %1082, %1101 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %1104 = x86.dm.vbroadcastsd [%1077 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1105 = x86.rss.vfmadd231pd %1074, %1081, %1104 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %1106 = x86.rss.vfmadd231pd %1075, %1082, %1104 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %1107 = x86.dm.vbroadcastsd [%1077 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1108 = x86.ri.add %1077, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %1109 = x86.ri.add %1078, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1110 = x86.rss.vfmadd231pd %1079, %1081, %1107 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1111 = x86.rss.vfmadd231pd %1080, %1082, %1107 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %1112 = x86.dm.vmovapd [%1109] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %1113 = x86.dm.vmovapd [%1109 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %1114 = x86.dm.vbroadcastsd [%1108] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1115 = x86.rss.vfmadd231pd %1084, %1112, %1114 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %1116 = x86.rss.vfmadd231pd %1085, %1113, %1114 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %1117 = x86.dm.vbroadcastsd [%1108 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1118 = x86.rss.vfmadd231pd %1087, %1112, %1117 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %1119 = x86.rss.vfmadd231pd %1088, %1113, %1117 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %1120 = x86.dm.vbroadcastsd [%1108 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1121 = x86.rss.vfmadd231pd %1090, %1112, %1120 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %1122 = x86.rss.vfmadd231pd %1091, %1113, %1120 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %1123 = x86.dm.vbroadcastsd [%1108 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1124 = x86.rss.vfmadd231pd %1093, %1112, %1123 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %1125 = x86.rss.vfmadd231pd %1094, %1113, %1123 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %1126 = x86.dm.vbroadcastsd [%1108 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1127 = x86.rss.vfmadd231pd %1096, %1112, %1126 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %1128 = x86.rss.vfmadd231pd %1097, %1113, %1126 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %1129 = x86.dm.vbroadcastsd [%1108 + 640] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1130 = x86.rss.vfmadd231pd %1099, %1112, %1129 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %1131 = x86.rss.vfmadd231pd %1100, %1113, %1129 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %1132 = x86.dm.vbroadcastsd [%1108 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1133 = x86.rss.vfmadd231pd %1102, %1112, %1132 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %1134 = x86.rss.vfmadd231pd %1103, %1113, %1132 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %1135 = x86.dm.vbroadcastsd [%1108 + 896] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1136 = x86.rss.vfmadd231pd %1105, %1112, %1135 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %1137 = x86.rss.vfmadd231pd %1106, %1113, %1135 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %1138 = x86.dm.vbroadcastsd [%1108 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %1139 = x86.ri.add %1108, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %1140 = x86.ri.add %1109, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1141 = x86.rss.vfmadd231pd %1110, %1112, %1138 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1142 = x86.rss.vfmadd231pd %1111, %1113, %1138 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %1143 = x86.ri.sub %1139, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%623], %1115 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 64], %1116 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 128], %1118 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 192], %1119 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 256], %1121 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 320], %1122 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 384], %1124 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 448], %1125 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 512], %1127 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 576], %1128 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 640], %1130 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 704], %1131 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 768], %1133 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 832], %1134 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 896], %1136 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 960], %1137 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 1024], %1141 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%623 + 1088], %1142 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %1144 = x86.ri.add %623, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %1145 = x86.ri.sub %1140, 1920 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1146 = x86.si.cmp %628, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %1146 : !x86.rflags<rflags>, ^bb6(%1145 : !x86.reg64<rdi>, %1143 : !x86.reg64<rsi>, %1144 : !x86.reg64<rdx>, %624 : !x86.reg64<rbp>, %625 : !x86.reg64<rsp>, %626 : !x86.reg64<r11>, %628 : !x86.reg64<r10>), ^bb7(%1145 : !x86.reg64<rdi>, %1143 : !x86.reg64<rsi>, %1144 : !x86.reg64<rdx>, %624 : !x86.reg64<rbp>, %625 : !x86.reg64<rsp>, %626 : !x86.reg64<r11>, %628 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb7(%1147: !x86.reg64<rdi>, %1148: !x86.reg64<rsi>, %1149: !x86.reg64<rdx>, %1150: !x86.reg64<rbp>, %1151: !x86.reg64<rsp>, %1152: !x86.reg64<r11>, %1153: !x86.reg64<r10>):
// CHECK-NEXT:      %1154 = x86.ri.add %1149, 1024 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %1155 = x86.ri.add %1148, 1152 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %1156 = x86.ri.sub %1147, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %1157 = x86.si.cmp %1152, 29 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %1157 : !x86.rflags<rflags>, ^bb5(%1156 : !x86.reg64<rdi>, %1155 : !x86.reg64<rsi>, %1154 : !x86.reg64<rdx>, %1150 : !x86.reg64<rbp>, %1151 : !x86.reg64<rsp>, %1152 : !x86.reg64<r11>), ^bb8(%1156 : !x86.reg64<rdi>, %1155 : !x86.reg64<rsi>, %1154 : !x86.reg64<rdx>, %1150 : !x86.reg64<rbp>, %1151 : !x86.reg64<rsp>, %1152 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb8(%1158: !x86.reg64<rdi>, %1159: !x86.reg64<rsi>, %1160: !x86.reg64<rdx>, %1161: !x86.reg64<rbp>, %1162: !x86.reg64<rsp>, %1163: !x86.reg64<r11>):
// CHECK-NEXT:      %1164 = x86.ds.mov %1161 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %1165, %1166 = x86.d.pop %1164 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
