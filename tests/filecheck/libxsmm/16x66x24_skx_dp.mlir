// RUN: libxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r12
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  l33:
// CHECK-NEXT:      add r11, 14
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  l34:
// CHECK-NEXT:      add r10, 16
// CHECK-NEXT:      vmovapd zmm4, [rdx]
// CHECK-NEXT:      vmovapd zmm5, [rdx+64]
// CHECK-NEXT:      vmovapd zmm6, [rdx+128]
// CHECK-NEXT:      vmovapd zmm7, [rdx+192]
// CHECK-NEXT:      vmovapd zmm8, [rdx+256]
// CHECK-NEXT:      vmovapd zmm9, [rdx+320]
// CHECK-NEXT:      vmovapd zmm10, [rdx+384]
// CHECK-NEXT:      vmovapd zmm11, [rdx+448]
// CHECK-NEXT:      vmovapd zmm12, [rdx+512]
// CHECK-NEXT:      vmovapd zmm13, [rdx+576]
// CHECK-NEXT:      vmovapd zmm14, [rdx+640]
// CHECK-NEXT:      vmovapd zmm15, [rdx+704]
// CHECK-NEXT:      vmovapd zmm16, [rdx+768]
// CHECK-NEXT:      vmovapd zmm17, [rdx+832]
// CHECK-NEXT:      vmovapd zmm18, [rdx+896]
// CHECK-NEXT:      vmovapd zmm19, [rdx+960]
// CHECK-NEXT:      vmovapd zmm20, [rdx+1024]
// CHECK-NEXT:      vmovapd zmm21, [rdx+1088]
// CHECK-NEXT:      vmovapd zmm22, [rdx+1152]
// CHECK-NEXT:      vmovapd zmm23, [rdx+1216]
// CHECK-NEXT:      vmovapd zmm24, [rdx+1280]
// CHECK-NEXT:      vmovapd zmm25, [rdx+1344]
// CHECK-NEXT:      vmovapd zmm26, [rdx+1408]
// CHECK-NEXT:      vmovapd zmm27, [rdx+1472]
// CHECK-NEXT:      vmovapd zmm28, [rdx+1536]
// CHECK-NEXT:      vmovapd zmm29, [rdx+1600]
// CHECK-NEXT:      vmovapd zmm30, [rdx+1664]
// CHECK-NEXT:      vmovapd zmm31, [rdx+1728]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  l35:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      cmp r12, 24
// CHECK-NEXT:      jl l35
// CHECK-NEXT:      sub rsi, 192
// CHECK-NEXT:      vmovapd [rdx], zmm4
// CHECK-NEXT:      vmovapd [rdx+64], zmm5
// CHECK-NEXT:      vmovapd [rdx+128], zmm6
// CHECK-NEXT:      vmovapd [rdx+192], zmm7
// CHECK-NEXT:      vmovapd [rdx+256], zmm8
// CHECK-NEXT:      vmovapd [rdx+320], zmm9
// CHECK-NEXT:      vmovapd [rdx+384], zmm10
// CHECK-NEXT:      vmovapd [rdx+448], zmm11
// CHECK-NEXT:      vmovapd [rdx+512], zmm12
// CHECK-NEXT:      vmovapd [rdx+576], zmm13
// CHECK-NEXT:      vmovapd [rdx+640], zmm14
// CHECK-NEXT:      vmovapd [rdx+704], zmm15
// CHECK-NEXT:      vmovapd [rdx+768], zmm16
// CHECK-NEXT:      vmovapd [rdx+832], zmm17
// CHECK-NEXT:      vmovapd [rdx+896], zmm18
// CHECK-NEXT:      vmovapd [rdx+960], zmm19
// CHECK-NEXT:      vmovapd [rdx+1024], zmm20
// CHECK-NEXT:      vmovapd [rdx+1088], zmm21
// CHECK-NEXT:      vmovapd [rdx+1152], zmm22
// CHECK-NEXT:      vmovapd [rdx+1216], zmm23
// CHECK-NEXT:      vmovapd [rdx+1280], zmm24
// CHECK-NEXT:      vmovapd [rdx+1344], zmm25
// CHECK-NEXT:      vmovapd [rdx+1408], zmm26
// CHECK-NEXT:      vmovapd [rdx+1472], zmm27
// CHECK-NEXT:      vmovapd [rdx+1536], zmm28
// CHECK-NEXT:      vmovapd [rdx+1600], zmm29
// CHECK-NEXT:      vmovapd [rdx+1664], zmm30
// CHECK-NEXT:      vmovapd [rdx+1728], zmm31
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      sub rdi, 2944
// CHECK-NEXT:      cmp r10, 16
// CHECK-NEXT:      jl l34
// CHECK-NEXT:      add rdx, 1664
// CHECK-NEXT:      add rsi, 2688
// CHECK-NEXT:      sub rdi, 128
// CHECK-NEXT:      cmp r11, 14
// CHECK-NEXT:      jl l33
// CHECK-NEXT:      mov r11, 14
// CHECK-NEXT:  l36:
// CHECK-NEXT:      add r11, 13
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  l37:
// CHECK-NEXT:      add r10, 16
// CHECK-NEXT:      vmovapd zmm6, [rdx]
// CHECK-NEXT:      vmovapd zmm7, [rdx+64]
// CHECK-NEXT:      vmovapd zmm8, [rdx+128]
// CHECK-NEXT:      vmovapd zmm9, [rdx+192]
// CHECK-NEXT:      vmovapd zmm10, [rdx+256]
// CHECK-NEXT:      vmovapd zmm11, [rdx+320]
// CHECK-NEXT:      vmovapd zmm12, [rdx+384]
// CHECK-NEXT:      vmovapd zmm13, [rdx+448]
// CHECK-NEXT:      vmovapd zmm14, [rdx+512]
// CHECK-NEXT:      vmovapd zmm15, [rdx+576]
// CHECK-NEXT:      vmovapd zmm16, [rdx+640]
// CHECK-NEXT:      vmovapd zmm17, [rdx+704]
// CHECK-NEXT:      vmovapd zmm18, [rdx+768]
// CHECK-NEXT:      vmovapd zmm19, [rdx+832]
// CHECK-NEXT:      vmovapd zmm20, [rdx+896]
// CHECK-NEXT:      vmovapd zmm21, [rdx+960]
// CHECK-NEXT:      vmovapd zmm22, [rdx+1024]
// CHECK-NEXT:      vmovapd zmm23, [rdx+1088]
// CHECK-NEXT:      vmovapd zmm24, [rdx+1152]
// CHECK-NEXT:      vmovapd zmm25, [rdx+1216]
// CHECK-NEXT:      vmovapd zmm26, [rdx+1280]
// CHECK-NEXT:      vmovapd zmm27, [rdx+1344]
// CHECK-NEXT:      vmovapd zmm28, [rdx+1408]
// CHECK-NEXT:      vmovapd zmm29, [rdx+1472]
// CHECK-NEXT:      vmovapd zmm30, [rdx+1536]
// CHECK-NEXT:      vmovapd zmm31, [rdx+1600]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  l38:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      cmp r12, 24
// CHECK-NEXT:      jl l38
// CHECK-NEXT:      sub rsi, 192
// CHECK-NEXT:      vmovapd [rdx], zmm6
// CHECK-NEXT:      vmovapd [rdx+64], zmm7
// CHECK-NEXT:      vmovapd [rdx+128], zmm8
// CHECK-NEXT:      vmovapd [rdx+192], zmm9
// CHECK-NEXT:      vmovapd [rdx+256], zmm10
// CHECK-NEXT:      vmovapd [rdx+320], zmm11
// CHECK-NEXT:      vmovapd [rdx+384], zmm12
// CHECK-NEXT:      vmovapd [rdx+448], zmm13
// CHECK-NEXT:      vmovapd [rdx+512], zmm14
// CHECK-NEXT:      vmovapd [rdx+576], zmm15
// CHECK-NEXT:      vmovapd [rdx+640], zmm16
// CHECK-NEXT:      vmovapd [rdx+704], zmm17
// CHECK-NEXT:      vmovapd [rdx+768], zmm18
// CHECK-NEXT:      vmovapd [rdx+832], zmm19
// CHECK-NEXT:      vmovapd [rdx+896], zmm20
// CHECK-NEXT:      vmovapd [rdx+960], zmm21
// CHECK-NEXT:      vmovapd [rdx+1024], zmm22
// CHECK-NEXT:      vmovapd [rdx+1088], zmm23
// CHECK-NEXT:      vmovapd [rdx+1152], zmm24
// CHECK-NEXT:      vmovapd [rdx+1216], zmm25
// CHECK-NEXT:      vmovapd [rdx+1280], zmm26
// CHECK-NEXT:      vmovapd [rdx+1344], zmm27
// CHECK-NEXT:      vmovapd [rdx+1408], zmm28
// CHECK-NEXT:      vmovapd [rdx+1472], zmm29
// CHECK-NEXT:      vmovapd [rdx+1536], zmm30
// CHECK-NEXT:      vmovapd [rdx+1600], zmm31
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      sub rdi, 2944
// CHECK-NEXT:      cmp r10, 16
// CHECK-NEXT:      jl l37
// CHECK-NEXT:      add rdx, 1536
// CHECK-NEXT:      add rsi, 2496
// CHECK-NEXT:      sub rdi, 128
// CHECK-NEXT:      cmp r11, 66
// CHECK-NEXT:      jl l36
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r12
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
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 14 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %26 = x86.ri.add %25, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %27 = x86.dm.vmovapd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %28 = x86.dm.vmovapd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm5>
// CHECK-IR-LIBXSMM-NEXT:      %29 = x86.dm.vmovapd [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.dm.vmovapd [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vmovapd [%21 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dm.vmovapd [%21 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dm.vmovapd [%21 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dm.vmovapd [%21 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.dm.vmovapd [%21 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dm.vmovapd [%21 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.dm.vmovapd [%21 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dm.vmovapd [%21 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.dm.vmovapd [%21 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dm.vmovapd [%21 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.dm.vmovapd [%21 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dm.vmovapd [%21 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.dm.vmovapd [%21 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.dm.vmovapd [%21 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.dm.vmovapd [%21 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dm.vmovapd [%21 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.dm.vmovapd [%21 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.dm.vmovapd [%21 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.dm.vmovapd [%21 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.dm.vmovapd [%21 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.dm.vmovapd [%21 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %52 = x86.dm.vmovapd [%21 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %53 = x86.dm.vmovapd [%21 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %54 = x86.dm.vmovapd [%21 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %55 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb2(%19 : !x86.reg64<rdi>, %20 : !x86.reg64<rsi>, %21 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>, %27 : !x86.avx512reg<zmm4>, %28 : !x86.avx512reg<zmm5>, %29 : !x86.avx512reg<zmm6>, %30 : !x86.avx512reg<zmm7>, %31 : !x86.avx512reg<zmm8>, %32 : !x86.avx512reg<zmm9>, %33 : !x86.avx512reg<zmm10>, %34 : !x86.avx512reg<zmm11>, %35 : !x86.avx512reg<zmm12>, %36 : !x86.avx512reg<zmm13>, %37 : !x86.avx512reg<zmm14>, %38 : !x86.avx512reg<zmm15>, %39 : !x86.avx512reg<zmm16>, %40 : !x86.avx512reg<zmm17>, %41 : !x86.avx512reg<zmm18>, %42 : !x86.avx512reg<zmm19>, %43 : !x86.avx512reg<zmm20>, %44 : !x86.avx512reg<zmm21>, %45 : !x86.avx512reg<zmm22>, %46 : !x86.avx512reg<zmm23>, %47 : !x86.avx512reg<zmm24>, %48 : !x86.avx512reg<zmm25>, %49 : !x86.avx512reg<zmm26>, %50 : !x86.avx512reg<zmm27>, %51 : !x86.avx512reg<zmm28>, %52 : !x86.avx512reg<zmm29>, %53 : !x86.avx512reg<zmm30>, %54 : !x86.avx512reg<zmm31>, %55 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%56: !x86.reg64<rdi>, %57: !x86.reg64<rsi>, %58: !x86.reg64<rdx>, %59: !x86.reg64<rbp>, %60: !x86.reg64<rsp>, %61: !x86.reg64<r11>, %62: !x86.reg64<r10>, %63: !x86.avx512reg<zmm4>, %64: !x86.avx512reg<zmm5>, %65: !x86.avx512reg<zmm6>, %66: !x86.avx512reg<zmm7>, %67: !x86.avx512reg<zmm8>, %68: !x86.avx512reg<zmm9>, %69: !x86.avx512reg<zmm10>, %70: !x86.avx512reg<zmm11>, %71: !x86.avx512reg<zmm12>, %72: !x86.avx512reg<zmm13>, %73: !x86.avx512reg<zmm14>, %74: !x86.avx512reg<zmm15>, %75: !x86.avx512reg<zmm16>, %76: !x86.avx512reg<zmm17>, %77: !x86.avx512reg<zmm18>, %78: !x86.avx512reg<zmm19>, %79: !x86.avx512reg<zmm20>, %80: !x86.avx512reg<zmm21>, %81: !x86.avx512reg<zmm22>, %82: !x86.avx512reg<zmm23>, %83: !x86.avx512reg<zmm24>, %84: !x86.avx512reg<zmm25>, %85: !x86.avx512reg<zmm26>, %86: !x86.avx512reg<zmm27>, %87: !x86.avx512reg<zmm28>, %88: !x86.avx512reg<zmm29>, %89: !x86.avx512reg<zmm30>, %90: !x86.avx512reg<zmm31>, %91: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l35"
// CHECK-IR-LIBXSMM-NEXT:      %92 = x86.ri.add %91, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %93 = x86.dm.vmovapd [%56] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %94 = x86.dm.vmovapd [%56 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %95 = x86.dm.vbroadcastsd [%57] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %96 = x86.rss.vfmadd231pd %63, %93, %95 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %97 = x86.rss.vfmadd231pd %64, %94, %95 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-IR-LIBXSMM-NEXT:      %98 = x86.dm.vbroadcastsd [%57 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %99 = x86.rss.vfmadd231pd %65, %93, %98 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %100 = x86.rss.vfmadd231pd %66, %94, %98 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %101 = x86.dm.vbroadcastsd [%57 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %102 = x86.rss.vfmadd231pd %67, %93, %101 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %103 = x86.rss.vfmadd231pd %68, %94, %101 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %104 = x86.dm.vbroadcastsd [%57 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %105 = x86.rss.vfmadd231pd %69, %93, %104 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %106 = x86.rss.vfmadd231pd %70, %94, %104 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %107 = x86.dm.vbroadcastsd [%57 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %108 = x86.rss.vfmadd231pd %71, %93, %107 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %109 = x86.rss.vfmadd231pd %72, %94, %107 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %110 = x86.dm.vbroadcastsd [%57 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %111 = x86.rss.vfmadd231pd %73, %93, %110 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %112 = x86.rss.vfmadd231pd %74, %94, %110 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %113 = x86.dm.vbroadcastsd [%57 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %114 = x86.rss.vfmadd231pd %75, %93, %113 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %115 = x86.rss.vfmadd231pd %76, %94, %113 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.dm.vbroadcastsd [%57 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.rss.vfmadd231pd %77, %93, %116 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.rss.vfmadd231pd %78, %94, %116 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.dm.vbroadcastsd [%57 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.rss.vfmadd231pd %79, %93, %119 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.rss.vfmadd231pd %80, %94, %119 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.dm.vbroadcastsd [%57 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.rss.vfmadd231pd %81, %93, %122 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.rss.vfmadd231pd %82, %94, %122 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %125 = x86.dm.vbroadcastsd [%57 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %126 = x86.rss.vfmadd231pd %83, %93, %125 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %127 = x86.rss.vfmadd231pd %84, %94, %125 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %128 = x86.dm.vbroadcastsd [%57 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %129 = x86.rss.vfmadd231pd %85, %93, %128 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %130 = x86.rss.vfmadd231pd %86, %94, %128 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %131 = x86.dm.vbroadcastsd [%57 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %132 = x86.rss.vfmadd231pd %87, %93, %131 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %133 = x86.rss.vfmadd231pd %88, %94, %131 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %134 = x86.dm.vbroadcastsd [%57 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %135 = x86.ri.add %57, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %136 = x86.ri.add %56, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %137 = x86.rss.vfmadd231pd %89, %93, %134 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %138 = x86.rss.vfmadd231pd %90, %94, %134 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %139 = x86.dm.vmovapd [%136] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %140 = x86.dm.vmovapd [%136 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %141 = x86.dm.vbroadcastsd [%135] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %142 = x86.rss.vfmadd231pd %96, %139, %141 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %143 = x86.rss.vfmadd231pd %97, %140, %141 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-IR-LIBXSMM-NEXT:      %144 = x86.dm.vbroadcastsd [%135 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %145 = x86.rss.vfmadd231pd %99, %139, %144 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %146 = x86.rss.vfmadd231pd %100, %140, %144 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %147 = x86.dm.vbroadcastsd [%135 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %148 = x86.rss.vfmadd231pd %102, %139, %147 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %149 = x86.rss.vfmadd231pd %103, %140, %147 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %150 = x86.dm.vbroadcastsd [%135 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %151 = x86.rss.vfmadd231pd %105, %139, %150 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.rss.vfmadd231pd %106, %140, %150 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.dm.vbroadcastsd [%135 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.rss.vfmadd231pd %108, %139, %153 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.rss.vfmadd231pd %109, %140, %153 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.dm.vbroadcastsd [%135 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.rss.vfmadd231pd %111, %139, %156 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.rss.vfmadd231pd %112, %140, %156 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.dm.vbroadcastsd [%135 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.rss.vfmadd231pd %114, %139, %159 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %161 = x86.rss.vfmadd231pd %115, %140, %159 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %162 = x86.dm.vbroadcastsd [%135 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %163 = x86.rss.vfmadd231pd %117, %139, %162 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %164 = x86.rss.vfmadd231pd %118, %140, %162 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %165 = x86.dm.vbroadcastsd [%135 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %166 = x86.rss.vfmadd231pd %120, %139, %165 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %167 = x86.rss.vfmadd231pd %121, %140, %165 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %168 = x86.dm.vbroadcastsd [%135 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %169 = x86.rss.vfmadd231pd %123, %139, %168 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %170 = x86.rss.vfmadd231pd %124, %140, %168 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %171 = x86.dm.vbroadcastsd [%135 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %172 = x86.rss.vfmadd231pd %126, %139, %171 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %173 = x86.rss.vfmadd231pd %127, %140, %171 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %174 = x86.dm.vbroadcastsd [%135 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %175 = x86.rss.vfmadd231pd %129, %139, %174 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %176 = x86.rss.vfmadd231pd %130, %140, %174 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %177 = x86.dm.vbroadcastsd [%135 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %178 = x86.rss.vfmadd231pd %132, %139, %177 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %179 = x86.rss.vfmadd231pd %133, %140, %177 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %180 = x86.dm.vbroadcastsd [%135 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %181 = x86.ri.add %135, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %182 = x86.ri.add %136, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %183 = x86.rss.vfmadd231pd %137, %139, %180 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %184 = x86.rss.vfmadd231pd %138, %140, %180 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %185 = x86.dm.vmovapd [%182] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %186 = x86.dm.vmovapd [%182 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %187 = x86.dm.vbroadcastsd [%181] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %188 = x86.rss.vfmadd231pd %142, %185, %187 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.rss.vfmadd231pd %143, %186, %187 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.dm.vbroadcastsd [%181 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.rss.vfmadd231pd %145, %185, %190 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.rss.vfmadd231pd %146, %186, %190 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %193 = x86.dm.vbroadcastsd [%181 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %194 = x86.rss.vfmadd231pd %148, %185, %193 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %195 = x86.rss.vfmadd231pd %149, %186, %193 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %196 = x86.dm.vbroadcastsd [%181 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %197 = x86.rss.vfmadd231pd %151, %185, %196 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %198 = x86.rss.vfmadd231pd %152, %186, %196 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %199 = x86.dm.vbroadcastsd [%181 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %200 = x86.rss.vfmadd231pd %154, %185, %199 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %201 = x86.rss.vfmadd231pd %155, %186, %199 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %202 = x86.dm.vbroadcastsd [%181 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %203 = x86.rss.vfmadd231pd %157, %185, %202 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %204 = x86.rss.vfmadd231pd %158, %186, %202 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %205 = x86.dm.vbroadcastsd [%181 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %206 = x86.rss.vfmadd231pd %160, %185, %205 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %207 = x86.rss.vfmadd231pd %161, %186, %205 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %208 = x86.dm.vbroadcastsd [%181 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %209 = x86.rss.vfmadd231pd %163, %185, %208 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %210 = x86.rss.vfmadd231pd %164, %186, %208 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %211 = x86.dm.vbroadcastsd [%181 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %212 = x86.rss.vfmadd231pd %166, %185, %211 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %213 = x86.rss.vfmadd231pd %167, %186, %211 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %214 = x86.dm.vbroadcastsd [%181 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %215 = x86.rss.vfmadd231pd %169, %185, %214 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %216 = x86.rss.vfmadd231pd %170, %186, %214 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %217 = x86.dm.vbroadcastsd [%181 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %218 = x86.rss.vfmadd231pd %172, %185, %217 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %219 = x86.rss.vfmadd231pd %173, %186, %217 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %220 = x86.dm.vbroadcastsd [%181 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %221 = x86.rss.vfmadd231pd %175, %185, %220 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %222 = x86.rss.vfmadd231pd %176, %186, %220 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %223 = x86.dm.vbroadcastsd [%181 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %224 = x86.rss.vfmadd231pd %178, %185, %223 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %225 = x86.rss.vfmadd231pd %179, %186, %223 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %226 = x86.dm.vbroadcastsd [%181 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %227 = x86.ri.add %181, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %228 = x86.ri.add %182, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %229 = x86.rss.vfmadd231pd %183, %185, %226 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %230 = x86.rss.vfmadd231pd %184, %186, %226 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %231 = x86.dm.vmovapd [%228] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %232 = x86.dm.vmovapd [%228 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %233 = x86.dm.vbroadcastsd [%227] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %234 = x86.rss.vfmadd231pd %188, %231, %233 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %235 = x86.rss.vfmadd231pd %189, %232, %233 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-IR-LIBXSMM-NEXT:      %236 = x86.dm.vbroadcastsd [%227 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %237 = x86.rss.vfmadd231pd %191, %231, %236 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %238 = x86.rss.vfmadd231pd %192, %232, %236 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %239 = x86.dm.vbroadcastsd [%227 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %240 = x86.rss.vfmadd231pd %194, %231, %239 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %241 = x86.rss.vfmadd231pd %195, %232, %239 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %242 = x86.dm.vbroadcastsd [%227 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %243 = x86.rss.vfmadd231pd %197, %231, %242 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %244 = x86.rss.vfmadd231pd %198, %232, %242 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %245 = x86.dm.vbroadcastsd [%227 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %246 = x86.rss.vfmadd231pd %200, %231, %245 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %247 = x86.rss.vfmadd231pd %201, %232, %245 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %248 = x86.dm.vbroadcastsd [%227 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %249 = x86.rss.vfmadd231pd %203, %231, %248 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %250 = x86.rss.vfmadd231pd %204, %232, %248 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %251 = x86.dm.vbroadcastsd [%227 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %252 = x86.rss.vfmadd231pd %206, %231, %251 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %253 = x86.rss.vfmadd231pd %207, %232, %251 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %254 = x86.dm.vbroadcastsd [%227 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %255 = x86.rss.vfmadd231pd %209, %231, %254 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %256 = x86.rss.vfmadd231pd %210, %232, %254 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %257 = x86.dm.vbroadcastsd [%227 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %258 = x86.rss.vfmadd231pd %212, %231, %257 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %259 = x86.rss.vfmadd231pd %213, %232, %257 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %260 = x86.dm.vbroadcastsd [%227 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %261 = x86.rss.vfmadd231pd %215, %231, %260 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %262 = x86.rss.vfmadd231pd %216, %232, %260 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %263 = x86.dm.vbroadcastsd [%227 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %264 = x86.rss.vfmadd231pd %218, %231, %263 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %265 = x86.rss.vfmadd231pd %219, %232, %263 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %266 = x86.dm.vbroadcastsd [%227 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %267 = x86.rss.vfmadd231pd %221, %231, %266 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %268 = x86.rss.vfmadd231pd %222, %232, %266 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %269 = x86.dm.vbroadcastsd [%227 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %270 = x86.rss.vfmadd231pd %224, %231, %269 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %271 = x86.rss.vfmadd231pd %225, %232, %269 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %272 = x86.dm.vbroadcastsd [%227 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %273 = x86.ri.add %227, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %274 = x86.ri.add %228, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %275 = x86.rss.vfmadd231pd %229, %231, %272 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %276 = x86.rss.vfmadd231pd %230, %232, %272 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %277 = x86.si.cmp %92, 24 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %277 : !x86.rflags<rflags>, ^bb2(%274 : !x86.reg64<rdi>, %273 : !x86.reg64<rsi>, %58 : !x86.reg64<rdx>, %59 : !x86.reg64<rbp>, %60 : !x86.reg64<rsp>, %61 : !x86.reg64<r11>, %62 : !x86.reg64<r10>, %234 : !x86.avx512reg<zmm4>, %235 : !x86.avx512reg<zmm5>, %237 : !x86.avx512reg<zmm6>, %238 : !x86.avx512reg<zmm7>, %240 : !x86.avx512reg<zmm8>, %241 : !x86.avx512reg<zmm9>, %243 : !x86.avx512reg<zmm10>, %244 : !x86.avx512reg<zmm11>, %246 : !x86.avx512reg<zmm12>, %247 : !x86.avx512reg<zmm13>, %249 : !x86.avx512reg<zmm14>, %250 : !x86.avx512reg<zmm15>, %252 : !x86.avx512reg<zmm16>, %253 : !x86.avx512reg<zmm17>, %255 : !x86.avx512reg<zmm18>, %256 : !x86.avx512reg<zmm19>, %258 : !x86.avx512reg<zmm20>, %259 : !x86.avx512reg<zmm21>, %261 : !x86.avx512reg<zmm22>, %262 : !x86.avx512reg<zmm23>, %264 : !x86.avx512reg<zmm24>, %265 : !x86.avx512reg<zmm25>, %267 : !x86.avx512reg<zmm26>, %268 : !x86.avx512reg<zmm27>, %270 : !x86.avx512reg<zmm28>, %271 : !x86.avx512reg<zmm29>, %275 : !x86.avx512reg<zmm30>, %276 : !x86.avx512reg<zmm31>, %92 : !x86.reg64<r12>), ^bb3(%274 : !x86.reg64<rdi>, %273 : !x86.reg64<rsi>, %58 : !x86.reg64<rdx>, %59 : !x86.reg64<rbp>, %60 : !x86.reg64<rsp>, %61 : !x86.reg64<r11>, %62 : !x86.reg64<r10>, %234 : !x86.avx512reg<zmm4>, %235 : !x86.avx512reg<zmm5>, %237 : !x86.avx512reg<zmm6>, %238 : !x86.avx512reg<zmm7>, %240 : !x86.avx512reg<zmm8>, %241 : !x86.avx512reg<zmm9>, %243 : !x86.avx512reg<zmm10>, %244 : !x86.avx512reg<zmm11>, %246 : !x86.avx512reg<zmm12>, %247 : !x86.avx512reg<zmm13>, %249 : !x86.avx512reg<zmm14>, %250 : !x86.avx512reg<zmm15>, %252 : !x86.avx512reg<zmm16>, %253 : !x86.avx512reg<zmm17>, %255 : !x86.avx512reg<zmm18>, %256 : !x86.avx512reg<zmm19>, %258 : !x86.avx512reg<zmm20>, %259 : !x86.avx512reg<zmm21>, %261 : !x86.avx512reg<zmm22>, %262 : !x86.avx512reg<zmm23>, %264 : !x86.avx512reg<zmm24>, %265 : !x86.avx512reg<zmm25>, %267 : !x86.avx512reg<zmm26>, %268 : !x86.avx512reg<zmm27>, %270 : !x86.avx512reg<zmm28>, %271 : !x86.avx512reg<zmm29>, %275 : !x86.avx512reg<zmm30>, %276 : !x86.avx512reg<zmm31>, %92 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%278: !x86.reg64<rdi>, %279: !x86.reg64<rsi>, %280: !x86.reg64<rdx>, %281: !x86.reg64<rbp>, %282: !x86.reg64<rsp>, %283: !x86.reg64<r11>, %284: !x86.reg64<r10>, %285: !x86.avx512reg<zmm4>, %286: !x86.avx512reg<zmm5>, %287: !x86.avx512reg<zmm6>, %288: !x86.avx512reg<zmm7>, %289: !x86.avx512reg<zmm8>, %290: !x86.avx512reg<zmm9>, %291: !x86.avx512reg<zmm10>, %292: !x86.avx512reg<zmm11>, %293: !x86.avx512reg<zmm12>, %294: !x86.avx512reg<zmm13>, %295: !x86.avx512reg<zmm14>, %296: !x86.avx512reg<zmm15>, %297: !x86.avx512reg<zmm16>, %298: !x86.avx512reg<zmm17>, %299: !x86.avx512reg<zmm18>, %300: !x86.avx512reg<zmm19>, %301: !x86.avx512reg<zmm20>, %302: !x86.avx512reg<zmm21>, %303: !x86.avx512reg<zmm22>, %304: !x86.avx512reg<zmm23>, %305: !x86.avx512reg<zmm24>, %306: !x86.avx512reg<zmm25>, %307: !x86.avx512reg<zmm26>, %308: !x86.avx512reg<zmm27>, %309: !x86.avx512reg<zmm28>, %310: !x86.avx512reg<zmm29>, %311: !x86.avx512reg<zmm30>, %312: !x86.avx512reg<zmm31>, %313: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.ri.sub %279, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280], %285 : (!x86.reg64<rdx>, !x86.avx512reg<zmm4>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 64], %286 : (!x86.reg64<rdx>, !x86.avx512reg<zmm5>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 128], %287 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 192], %288 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 256], %289 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 320], %290 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 384], %291 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 448], %292 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 512], %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 576], %294 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 640], %295 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 704], %296 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 768], %297 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 832], %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 896], %299 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 960], %300 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1024], %301 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1088], %302 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1152], %303 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1216], %304 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1280], %305 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1344], %306 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1408], %307 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1472], %308 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1536], %309 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1600], %310 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1664], %311 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%280 + 1728], %312 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.ri.add %280, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.ri.sub %278, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.si.cmp %284, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %317 : !x86.rflags<rflags>, ^bb1(%316 : !x86.reg64<rdi>, %314 : !x86.reg64<rsi>, %315 : !x86.reg64<rdx>, %281 : !x86.reg64<rbp>, %282 : !x86.reg64<rsp>, %283 : !x86.reg64<r11>, %284 : !x86.reg64<r10>), ^bb4(%316 : !x86.reg64<rdi>, %314 : !x86.reg64<rsi>, %315 : !x86.reg64<rdx>, %281 : !x86.reg64<rbp>, %282 : !x86.reg64<rsp>, %283 : !x86.reg64<r11>, %284 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb4(%318: !x86.reg64<rdi>, %319: !x86.reg64<rsi>, %320: !x86.reg64<rdx>, %321: !x86.reg64<rbp>, %322: !x86.reg64<rsp>, %323: !x86.reg64<r11>, %324: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %325 = x86.ri.add %320, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %326 = x86.ri.add %319, 2688 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %327 = x86.ri.sub %318, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %328 = x86.si.cmp %323, 14 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %328 : !x86.rflags<rflags>, ^bb0(%327 : !x86.reg64<rdi>, %326 : !x86.reg64<rsi>, %325 : !x86.reg64<rdx>, %321 : !x86.reg64<rbp>, %322 : !x86.reg64<rsp>, %323 : !x86.reg64<r11>), ^bb5(%327 : !x86.reg64<rdi>, %326 : !x86.reg64<rsi>, %325 : !x86.reg64<rdx>, %321 : !x86.reg64<rbp>, %322 : !x86.reg64<rsp>, %323 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb5(%329: !x86.reg64<rdi>, %330: !x86.reg64<rsi>, %331: !x86.reg64<rdx>, %332: !x86.reg64<rbp>, %333: !x86.reg64<rsp>, %334: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %335 = x86.di.mov 14 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb6(%329 : !x86.reg64<rdi>, %330 : !x86.reg64<rsi>, %331 : !x86.reg64<rdx>, %332 : !x86.reg64<rbp>, %333 : !x86.reg64<rsp>, %335 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb6(%336: !x86.reg64<rdi>, %337: !x86.reg64<rsi>, %338: !x86.reg64<rdx>, %339: !x86.reg64<rbp>, %340: !x86.reg64<rsp>, %341: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l36"
// CHECK-IR-LIBXSMM-NEXT:      %342 = x86.ri.add %341, 13 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %343 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb7(%336 : !x86.reg64<rdi>, %337 : !x86.reg64<rsi>, %338 : !x86.reg64<rdx>, %339 : !x86.reg64<rbp>, %340 : !x86.reg64<rsp>, %342 : !x86.reg64<r11>, %343 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb7(%344: !x86.reg64<rdi>, %345: !x86.reg64<rsi>, %346: !x86.reg64<rdx>, %347: !x86.reg64<rbp>, %348: !x86.reg64<rsp>, %349: !x86.reg64<r11>, %350: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l37"
// CHECK-IR-LIBXSMM-NEXT:      %351 = x86.ri.add %350, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %352 = x86.dm.vmovapd [%346] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %353 = x86.dm.vmovapd [%346 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %354 = x86.dm.vmovapd [%346 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %355 = x86.dm.vmovapd [%346 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %356 = x86.dm.vmovapd [%346 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %357 = x86.dm.vmovapd [%346 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %358 = x86.dm.vmovapd [%346 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %359 = x86.dm.vmovapd [%346 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %360 = x86.dm.vmovapd [%346 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %361 = x86.dm.vmovapd [%346 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %362 = x86.dm.vmovapd [%346 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %363 = x86.dm.vmovapd [%346 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %364 = x86.dm.vmovapd [%346 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %365 = x86.dm.vmovapd [%346 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %366 = x86.dm.vmovapd [%346 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.dm.vmovapd [%346 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %368 = x86.dm.vmovapd [%346 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %369 = x86.dm.vmovapd [%346 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %370 = x86.dm.vmovapd [%346 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %371 = x86.dm.vmovapd [%346 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %372 = x86.dm.vmovapd [%346 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %373 = x86.dm.vmovapd [%346 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %374 = x86.dm.vmovapd [%346 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %375 = x86.dm.vmovapd [%346 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %376 = x86.dm.vmovapd [%346 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %377 = x86.dm.vmovapd [%346 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %378 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb8(%344 : !x86.reg64<rdi>, %345 : !x86.reg64<rsi>, %346 : !x86.reg64<rdx>, %347 : !x86.reg64<rbp>, %348 : !x86.reg64<rsp>, %349 : !x86.reg64<r11>, %351 : !x86.reg64<r10>, %352 : !x86.avx512reg<zmm6>, %353 : !x86.avx512reg<zmm7>, %354 : !x86.avx512reg<zmm8>, %355 : !x86.avx512reg<zmm9>, %356 : !x86.avx512reg<zmm10>, %357 : !x86.avx512reg<zmm11>, %358 : !x86.avx512reg<zmm12>, %359 : !x86.avx512reg<zmm13>, %360 : !x86.avx512reg<zmm14>, %361 : !x86.avx512reg<zmm15>, %362 : !x86.avx512reg<zmm16>, %363 : !x86.avx512reg<zmm17>, %364 : !x86.avx512reg<zmm18>, %365 : !x86.avx512reg<zmm19>, %366 : !x86.avx512reg<zmm20>, %367 : !x86.avx512reg<zmm21>, %368 : !x86.avx512reg<zmm22>, %369 : !x86.avx512reg<zmm23>, %370 : !x86.avx512reg<zmm24>, %371 : !x86.avx512reg<zmm25>, %372 : !x86.avx512reg<zmm26>, %373 : !x86.avx512reg<zmm27>, %374 : !x86.avx512reg<zmm28>, %375 : !x86.avx512reg<zmm29>, %376 : !x86.avx512reg<zmm30>, %377 : !x86.avx512reg<zmm31>, %378 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb8(%379: !x86.reg64<rdi>, %380: !x86.reg64<rsi>, %381: !x86.reg64<rdx>, %382: !x86.reg64<rbp>, %383: !x86.reg64<rsp>, %384: !x86.reg64<r11>, %385: !x86.reg64<r10>, %386: !x86.avx512reg<zmm6>, %387: !x86.avx512reg<zmm7>, %388: !x86.avx512reg<zmm8>, %389: !x86.avx512reg<zmm9>, %390: !x86.avx512reg<zmm10>, %391: !x86.avx512reg<zmm11>, %392: !x86.avx512reg<zmm12>, %393: !x86.avx512reg<zmm13>, %394: !x86.avx512reg<zmm14>, %395: !x86.avx512reg<zmm15>, %396: !x86.avx512reg<zmm16>, %397: !x86.avx512reg<zmm17>, %398: !x86.avx512reg<zmm18>, %399: !x86.avx512reg<zmm19>, %400: !x86.avx512reg<zmm20>, %401: !x86.avx512reg<zmm21>, %402: !x86.avx512reg<zmm22>, %403: !x86.avx512reg<zmm23>, %404: !x86.avx512reg<zmm24>, %405: !x86.avx512reg<zmm25>, %406: !x86.avx512reg<zmm26>, %407: !x86.avx512reg<zmm27>, %408: !x86.avx512reg<zmm28>, %409: !x86.avx512reg<zmm29>, %410: !x86.avx512reg<zmm30>, %411: !x86.avx512reg<zmm31>, %412: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l38"
// CHECK-IR-LIBXSMM-NEXT:      %413 = x86.ri.add %412, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %414 = x86.dm.vmovapd [%379] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %415 = x86.dm.vmovapd [%379 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %416 = x86.dm.vbroadcastsd [%380] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %417 = x86.rss.vfmadd231pd %386, %414, %416 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %418 = x86.rss.vfmadd231pd %387, %415, %416 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %419 = x86.dm.vbroadcastsd [%380 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %420 = x86.rss.vfmadd231pd %388, %414, %419 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %421 = x86.rss.vfmadd231pd %389, %415, %419 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %422 = x86.dm.vbroadcastsd [%380 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %423 = x86.rss.vfmadd231pd %390, %414, %422 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %424 = x86.rss.vfmadd231pd %391, %415, %422 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %425 = x86.dm.vbroadcastsd [%380 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %426 = x86.rss.vfmadd231pd %392, %414, %425 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %427 = x86.rss.vfmadd231pd %393, %415, %425 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %428 = x86.dm.vbroadcastsd [%380 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %429 = x86.rss.vfmadd231pd %394, %414, %428 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %430 = x86.rss.vfmadd231pd %395, %415, %428 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %431 = x86.dm.vbroadcastsd [%380 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %432 = x86.rss.vfmadd231pd %396, %414, %431 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %433 = x86.rss.vfmadd231pd %397, %415, %431 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %434 = x86.dm.vbroadcastsd [%380 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %435 = x86.rss.vfmadd231pd %398, %414, %434 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %436 = x86.rss.vfmadd231pd %399, %415, %434 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %437 = x86.dm.vbroadcastsd [%380 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %438 = x86.rss.vfmadd231pd %400, %414, %437 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %439 = x86.rss.vfmadd231pd %401, %415, %437 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %440 = x86.dm.vbroadcastsd [%380 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %441 = x86.rss.vfmadd231pd %402, %414, %440 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %442 = x86.rss.vfmadd231pd %403, %415, %440 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %443 = x86.dm.vbroadcastsd [%380 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %444 = x86.rss.vfmadd231pd %404, %414, %443 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %445 = x86.rss.vfmadd231pd %405, %415, %443 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %446 = x86.dm.vbroadcastsd [%380 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %447 = x86.rss.vfmadd231pd %406, %414, %446 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %448 = x86.rss.vfmadd231pd %407, %415, %446 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %449 = x86.dm.vbroadcastsd [%380 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %450 = x86.rss.vfmadd231pd %408, %414, %449 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %451 = x86.rss.vfmadd231pd %409, %415, %449 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %452 = x86.dm.vbroadcastsd [%380 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %453 = x86.ri.add %380, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %454 = x86.ri.add %379, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %455 = x86.rss.vfmadd231pd %410, %414, %452 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %456 = x86.rss.vfmadd231pd %411, %415, %452 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %457 = x86.dm.vmovapd [%454] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %458 = x86.dm.vmovapd [%454 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %459 = x86.dm.vbroadcastsd [%453] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %460 = x86.rss.vfmadd231pd %417, %457, %459 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %461 = x86.rss.vfmadd231pd %418, %458, %459 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %462 = x86.dm.vbroadcastsd [%453 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %463 = x86.rss.vfmadd231pd %420, %457, %462 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %464 = x86.rss.vfmadd231pd %421, %458, %462 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %465 = x86.dm.vbroadcastsd [%453 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %466 = x86.rss.vfmadd231pd %423, %457, %465 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %467 = x86.rss.vfmadd231pd %424, %458, %465 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %468 = x86.dm.vbroadcastsd [%453 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %469 = x86.rss.vfmadd231pd %426, %457, %468 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %470 = x86.rss.vfmadd231pd %427, %458, %468 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %471 = x86.dm.vbroadcastsd [%453 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %472 = x86.rss.vfmadd231pd %429, %457, %471 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %473 = x86.rss.vfmadd231pd %430, %458, %471 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %474 = x86.dm.vbroadcastsd [%453 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %475 = x86.rss.vfmadd231pd %432, %457, %474 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %476 = x86.rss.vfmadd231pd %433, %458, %474 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %477 = x86.dm.vbroadcastsd [%453 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %478 = x86.rss.vfmadd231pd %435, %457, %477 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %479 = x86.rss.vfmadd231pd %436, %458, %477 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %480 = x86.dm.vbroadcastsd [%453 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %481 = x86.rss.vfmadd231pd %438, %457, %480 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %482 = x86.rss.vfmadd231pd %439, %458, %480 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %483 = x86.dm.vbroadcastsd [%453 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %484 = x86.rss.vfmadd231pd %441, %457, %483 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %485 = x86.rss.vfmadd231pd %442, %458, %483 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %486 = x86.dm.vbroadcastsd [%453 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %487 = x86.rss.vfmadd231pd %444, %457, %486 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %488 = x86.rss.vfmadd231pd %445, %458, %486 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %489 = x86.dm.vbroadcastsd [%453 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %490 = x86.rss.vfmadd231pd %447, %457, %489 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %491 = x86.rss.vfmadd231pd %448, %458, %489 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %492 = x86.dm.vbroadcastsd [%453 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %493 = x86.rss.vfmadd231pd %450, %457, %492 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %494 = x86.rss.vfmadd231pd %451, %458, %492 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %495 = x86.dm.vbroadcastsd [%453 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %496 = x86.ri.add %453, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %497 = x86.ri.add %454, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %498 = x86.rss.vfmadd231pd %455, %457, %495 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %499 = x86.rss.vfmadd231pd %456, %458, %495 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %500 = x86.dm.vmovapd [%497] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %501 = x86.dm.vmovapd [%497 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %502 = x86.dm.vbroadcastsd [%496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %503 = x86.rss.vfmadd231pd %460, %500, %502 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %504 = x86.rss.vfmadd231pd %461, %501, %502 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %505 = x86.dm.vbroadcastsd [%496 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %506 = x86.rss.vfmadd231pd %463, %500, %505 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %507 = x86.rss.vfmadd231pd %464, %501, %505 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %508 = x86.dm.vbroadcastsd [%496 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %509 = x86.rss.vfmadd231pd %466, %500, %508 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %510 = x86.rss.vfmadd231pd %467, %501, %508 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %511 = x86.dm.vbroadcastsd [%496 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %512 = x86.rss.vfmadd231pd %469, %500, %511 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %513 = x86.rss.vfmadd231pd %470, %501, %511 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %514 = x86.dm.vbroadcastsd [%496 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %515 = x86.rss.vfmadd231pd %472, %500, %514 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %516 = x86.rss.vfmadd231pd %473, %501, %514 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %517 = x86.dm.vbroadcastsd [%496 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %518 = x86.rss.vfmadd231pd %475, %500, %517 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %519 = x86.rss.vfmadd231pd %476, %501, %517 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %520 = x86.dm.vbroadcastsd [%496 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %521 = x86.rss.vfmadd231pd %478, %500, %520 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %522 = x86.rss.vfmadd231pd %479, %501, %520 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %523 = x86.dm.vbroadcastsd [%496 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %524 = x86.rss.vfmadd231pd %481, %500, %523 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %525 = x86.rss.vfmadd231pd %482, %501, %523 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %526 = x86.dm.vbroadcastsd [%496 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %527 = x86.rss.vfmadd231pd %484, %500, %526 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %528 = x86.rss.vfmadd231pd %485, %501, %526 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %529 = x86.dm.vbroadcastsd [%496 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %530 = x86.rss.vfmadd231pd %487, %500, %529 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %531 = x86.rss.vfmadd231pd %488, %501, %529 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %532 = x86.dm.vbroadcastsd [%496 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %533 = x86.rss.vfmadd231pd %490, %500, %532 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %534 = x86.rss.vfmadd231pd %491, %501, %532 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %535 = x86.dm.vbroadcastsd [%496 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %536 = x86.rss.vfmadd231pd %493, %500, %535 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %537 = x86.rss.vfmadd231pd %494, %501, %535 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %538 = x86.dm.vbroadcastsd [%496 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %539 = x86.ri.add %496, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %540 = x86.ri.add %497, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %541 = x86.rss.vfmadd231pd %498, %500, %538 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %542 = x86.rss.vfmadd231pd %499, %501, %538 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %543 = x86.dm.vmovapd [%540] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %544 = x86.dm.vmovapd [%540 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %545 = x86.dm.vbroadcastsd [%539] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %546 = x86.rss.vfmadd231pd %503, %543, %545 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-IR-LIBXSMM-NEXT:      %547 = x86.rss.vfmadd231pd %504, %544, %545 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-IR-LIBXSMM-NEXT:      %548 = x86.dm.vbroadcastsd [%539 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %549 = x86.rss.vfmadd231pd %506, %543, %548 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %550 = x86.rss.vfmadd231pd %507, %544, %548 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %551 = x86.dm.vbroadcastsd [%539 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %552 = x86.rss.vfmadd231pd %509, %543, %551 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %553 = x86.rss.vfmadd231pd %510, %544, %551 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %554 = x86.dm.vbroadcastsd [%539 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %555 = x86.rss.vfmadd231pd %512, %543, %554 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %556 = x86.rss.vfmadd231pd %513, %544, %554 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %557 = x86.dm.vbroadcastsd [%539 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %558 = x86.rss.vfmadd231pd %515, %543, %557 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %559 = x86.rss.vfmadd231pd %516, %544, %557 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %560 = x86.dm.vbroadcastsd [%539 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %561 = x86.rss.vfmadd231pd %518, %543, %560 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %562 = x86.rss.vfmadd231pd %519, %544, %560 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %563 = x86.dm.vbroadcastsd [%539 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %564 = x86.rss.vfmadd231pd %521, %543, %563 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %565 = x86.rss.vfmadd231pd %522, %544, %563 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %566 = x86.dm.vbroadcastsd [%539 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %567 = x86.rss.vfmadd231pd %524, %543, %566 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %568 = x86.rss.vfmadd231pd %525, %544, %566 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %569 = x86.dm.vbroadcastsd [%539 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %570 = x86.rss.vfmadd231pd %527, %543, %569 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %571 = x86.rss.vfmadd231pd %528, %544, %569 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %572 = x86.dm.vbroadcastsd [%539 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %573 = x86.rss.vfmadd231pd %530, %543, %572 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %574 = x86.rss.vfmadd231pd %531, %544, %572 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %575 = x86.dm.vbroadcastsd [%539 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %576 = x86.rss.vfmadd231pd %533, %543, %575 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %577 = x86.rss.vfmadd231pd %534, %544, %575 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %578 = x86.dm.vbroadcastsd [%539 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %579 = x86.rss.vfmadd231pd %536, %543, %578 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %580 = x86.rss.vfmadd231pd %537, %544, %578 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %581 = x86.dm.vbroadcastsd [%539 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %582 = x86.ri.add %539, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %583 = x86.ri.add %540, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %584 = x86.rss.vfmadd231pd %541, %543, %581 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %585 = x86.rss.vfmadd231pd %542, %544, %581 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %586 = x86.si.cmp %413, 24 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %586 : !x86.rflags<rflags>, ^bb8(%583 : !x86.reg64<rdi>, %582 : !x86.reg64<rsi>, %381 : !x86.reg64<rdx>, %382 : !x86.reg64<rbp>, %383 : !x86.reg64<rsp>, %384 : !x86.reg64<r11>, %385 : !x86.reg64<r10>, %546 : !x86.avx512reg<zmm6>, %547 : !x86.avx512reg<zmm7>, %549 : !x86.avx512reg<zmm8>, %550 : !x86.avx512reg<zmm9>, %552 : !x86.avx512reg<zmm10>, %553 : !x86.avx512reg<zmm11>, %555 : !x86.avx512reg<zmm12>, %556 : !x86.avx512reg<zmm13>, %558 : !x86.avx512reg<zmm14>, %559 : !x86.avx512reg<zmm15>, %561 : !x86.avx512reg<zmm16>, %562 : !x86.avx512reg<zmm17>, %564 : !x86.avx512reg<zmm18>, %565 : !x86.avx512reg<zmm19>, %567 : !x86.avx512reg<zmm20>, %568 : !x86.avx512reg<zmm21>, %570 : !x86.avx512reg<zmm22>, %571 : !x86.avx512reg<zmm23>, %573 : !x86.avx512reg<zmm24>, %574 : !x86.avx512reg<zmm25>, %576 : !x86.avx512reg<zmm26>, %577 : !x86.avx512reg<zmm27>, %579 : !x86.avx512reg<zmm28>, %580 : !x86.avx512reg<zmm29>, %584 : !x86.avx512reg<zmm30>, %585 : !x86.avx512reg<zmm31>, %413 : !x86.reg64<r12>), ^bb9(%583 : !x86.reg64<rdi>, %582 : !x86.reg64<rsi>, %381 : !x86.reg64<rdx>, %382 : !x86.reg64<rbp>, %383 : !x86.reg64<rsp>, %384 : !x86.reg64<r11>, %385 : !x86.reg64<r10>, %546 : !x86.avx512reg<zmm6>, %547 : !x86.avx512reg<zmm7>, %549 : !x86.avx512reg<zmm8>, %550 : !x86.avx512reg<zmm9>, %552 : !x86.avx512reg<zmm10>, %553 : !x86.avx512reg<zmm11>, %555 : !x86.avx512reg<zmm12>, %556 : !x86.avx512reg<zmm13>, %558 : !x86.avx512reg<zmm14>, %559 : !x86.avx512reg<zmm15>, %561 : !x86.avx512reg<zmm16>, %562 : !x86.avx512reg<zmm17>, %564 : !x86.avx512reg<zmm18>, %565 : !x86.avx512reg<zmm19>, %567 : !x86.avx512reg<zmm20>, %568 : !x86.avx512reg<zmm21>, %570 : !x86.avx512reg<zmm22>, %571 : !x86.avx512reg<zmm23>, %573 : !x86.avx512reg<zmm24>, %574 : !x86.avx512reg<zmm25>, %576 : !x86.avx512reg<zmm26>, %577 : !x86.avx512reg<zmm27>, %579 : !x86.avx512reg<zmm28>, %580 : !x86.avx512reg<zmm29>, %584 : !x86.avx512reg<zmm30>, %585 : !x86.avx512reg<zmm31>, %413 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb9(%587: !x86.reg64<rdi>, %588: !x86.reg64<rsi>, %589: !x86.reg64<rdx>, %590: !x86.reg64<rbp>, %591: !x86.reg64<rsp>, %592: !x86.reg64<r11>, %593: !x86.reg64<r10>, %594: !x86.avx512reg<zmm6>, %595: !x86.avx512reg<zmm7>, %596: !x86.avx512reg<zmm8>, %597: !x86.avx512reg<zmm9>, %598: !x86.avx512reg<zmm10>, %599: !x86.avx512reg<zmm11>, %600: !x86.avx512reg<zmm12>, %601: !x86.avx512reg<zmm13>, %602: !x86.avx512reg<zmm14>, %603: !x86.avx512reg<zmm15>, %604: !x86.avx512reg<zmm16>, %605: !x86.avx512reg<zmm17>, %606: !x86.avx512reg<zmm18>, %607: !x86.avx512reg<zmm19>, %608: !x86.avx512reg<zmm20>, %609: !x86.avx512reg<zmm21>, %610: !x86.avx512reg<zmm22>, %611: !x86.avx512reg<zmm23>, %612: !x86.avx512reg<zmm24>, %613: !x86.avx512reg<zmm25>, %614: !x86.avx512reg<zmm26>, %615: !x86.avx512reg<zmm27>, %616: !x86.avx512reg<zmm28>, %617: !x86.avx512reg<zmm29>, %618: !x86.avx512reg<zmm30>, %619: !x86.avx512reg<zmm31>, %620: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %621 = x86.ri.sub %588, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589], %594 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 64], %595 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 128], %596 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 192], %597 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 256], %598 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 320], %599 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 384], %600 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 448], %601 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 512], %602 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 576], %603 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 640], %604 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 704], %605 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 768], %606 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 832], %607 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 896], %608 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 960], %609 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1024], %610 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1088], %611 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1152], %612 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1216], %613 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1280], %614 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1344], %615 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1408], %616 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1472], %617 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1536], %618 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%589 + 1600], %619 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %622 = x86.ri.add %589, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %623 = x86.ri.sub %587, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %624 = x86.si.cmp %593, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %624 : !x86.rflags<rflags>, ^bb7(%623 : !x86.reg64<rdi>, %621 : !x86.reg64<rsi>, %622 : !x86.reg64<rdx>, %590 : !x86.reg64<rbp>, %591 : !x86.reg64<rsp>, %592 : !x86.reg64<r11>, %593 : !x86.reg64<r10>), ^bb10(%623 : !x86.reg64<rdi>, %621 : !x86.reg64<rsi>, %622 : !x86.reg64<rdx>, %590 : !x86.reg64<rbp>, %591 : !x86.reg64<rsp>, %592 : !x86.reg64<r11>, %593 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb10(%625: !x86.reg64<rdi>, %626: !x86.reg64<rsi>, %627: !x86.reg64<rdx>, %628: !x86.reg64<rbp>, %629: !x86.reg64<rsp>, %630: !x86.reg64<r11>, %631: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %632 = x86.ri.add %627, 1536 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %633 = x86.ri.add %626, 2496 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %634 = x86.ri.sub %625, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %635 = x86.si.cmp %630, 66 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %635 : !x86.rflags<rflags>, ^bb6(%634 : !x86.reg64<rdi>, %633 : !x86.reg64<rsi>, %632 : !x86.reg64<rdx>, %628 : !x86.reg64<rbp>, %629 : !x86.reg64<rsp>, %630 : !x86.reg64<r11>), ^bb11(%634 : !x86.reg64<rdi>, %633 : !x86.reg64<rsi>, %632 : !x86.reg64<rdx>, %628 : !x86.reg64<rbp>, %629 : !x86.reg64<rsp>, %630 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb11(%636: !x86.reg64<rdi>, %637: !x86.reg64<rsi>, %638: !x86.reg64<rdx>, %639: !x86.reg64<rbp>, %640: !x86.reg64<rsp>, %641: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %642 = x86.ds.mov %639 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %643, %644 = x86.d.pop %642 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }
