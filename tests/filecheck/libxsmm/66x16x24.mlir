// RUN: libxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

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
