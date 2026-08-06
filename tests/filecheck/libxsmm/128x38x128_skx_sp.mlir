// RUN: libxsmm-gemm dense %t matmul_bac 128 38 128  128 128 128  1 1  1 1  skx  nopf  SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 128 38 128  128 128 128  1 1  1 1  skx  nopf  SP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128  128 128 128  1 1  1 1  skx  nopf  SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

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
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 6
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 64
// CHECK-NEXT:      vmovaps zmm8, [rdx]
// CHECK-NEXT:      vmovaps zmm9, [rdx+64]
// CHECK-NEXT:      vmovaps zmm10, [rdx+128]
// CHECK-NEXT:      vmovaps zmm11, [rdx+192]
// CHECK-NEXT:      vmovaps zmm12, [rdx+512]
// CHECK-NEXT:      vmovaps zmm13, [rdx+576]
// CHECK-NEXT:      vmovaps zmm14, [rdx+640]
// CHECK-NEXT:      vmovaps zmm15, [rdx+704]
// CHECK-NEXT:      vmovaps zmm16, [rdx+1024]
// CHECK-NEXT:      vmovaps zmm17, [rdx+1088]
// CHECK-NEXT:      vmovaps zmm18, [rdx+1152]
// CHECK-NEXT:      vmovaps zmm19, [rdx+1216]
// CHECK-NEXT:      vmovaps zmm20, [rdx+1536]
// CHECK-NEXT:      vmovaps zmm21, [rdx+1600]
// CHECK-NEXT:      vmovaps zmm22, [rdx+1664]
// CHECK-NEXT:      vmovaps zmm23, [rdx+1728]
// CHECK-NEXT:      vmovaps zmm24, [rdx+2048]
// CHECK-NEXT:      vmovaps zmm25, [rdx+2112]
// CHECK-NEXT:      vmovaps zmm26, [rdx+2176]
// CHECK-NEXT:      vmovaps zmm27, [rdx+2240]
// CHECK-NEXT:      vmovaps zmm28, [rdx+2560]
// CHECK-NEXT:      vmovaps zmm29, [rdx+2624]
// CHECK-NEXT:      vmovaps zmm30, [rdx+2688]
// CHECK-NEXT:      vmovaps zmm31, [rdx+2752]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovaps [rdx], zmm8
// CHECK-NEXT:      vmovaps [rdx+64], zmm9
// CHECK-NEXT:      vmovaps [rdx+128], zmm10
// CHECK-NEXT:      vmovaps [rdx+192], zmm11
// CHECK-NEXT:      vmovaps [rdx+512], zmm12
// CHECK-NEXT:      vmovaps [rdx+576], zmm13
// CHECK-NEXT:      vmovaps [rdx+640], zmm14
// CHECK-NEXT:      vmovaps [rdx+704], zmm15
// CHECK-NEXT:      vmovaps [rdx+1024], zmm16
// CHECK-NEXT:      vmovaps [rdx+1088], zmm17
// CHECK-NEXT:      vmovaps [rdx+1152], zmm18
// CHECK-NEXT:      vmovaps [rdx+1216], zmm19
// CHECK-NEXT:      vmovaps [rdx+1536], zmm20
// CHECK-NEXT:      vmovaps [rdx+1600], zmm21
// CHECK-NEXT:      vmovaps [rdx+1664], zmm22
// CHECK-NEXT:      vmovaps [rdx+1728], zmm23
// CHECK-NEXT:      vmovaps [rdx+2048], zmm24
// CHECK-NEXT:      vmovaps [rdx+2112], zmm25
// CHECK-NEXT:      vmovaps [rdx+2176], zmm26
// CHECK-NEXT:      vmovaps [rdx+2240], zmm27
// CHECK-NEXT:      vmovaps [rdx+2560], zmm28
// CHECK-NEXT:      vmovaps [rdx+2624], zmm29
// CHECK-NEXT:      vmovaps [rdx+2688], zmm30
// CHECK-NEXT:      vmovaps [rdx+2752], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 65280
// CHECK-NEXT:      cmp r10, 128
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      add rdx, 2560
// CHECK-NEXT:      add rsi, 3072
// CHECK-NEXT:      sub rdi, 512
// CHECK-NEXT:      cmp r11, 18
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov r11, 18
// CHECK-NEXT:  [[SCF_N2_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M2_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 64
// CHECK-NEXT:      vmovaps zmm12, [rdx]
// CHECK-NEXT:      vmovaps zmm13, [rdx+64]
// CHECK-NEXT:      vmovaps zmm14, [rdx+128]
// CHECK-NEXT:      vmovaps zmm15, [rdx+192]
// CHECK-NEXT:      vmovaps zmm16, [rdx+512]
// CHECK-NEXT:      vmovaps zmm17, [rdx+576]
// CHECK-NEXT:      vmovaps zmm18, [rdx+640]
// CHECK-NEXT:      vmovaps zmm19, [rdx+704]
// CHECK-NEXT:      vmovaps zmm20, [rdx+1024]
// CHECK-NEXT:      vmovaps zmm21, [rdx+1088]
// CHECK-NEXT:      vmovaps zmm22, [rdx+1152]
// CHECK-NEXT:      vmovaps zmm23, [rdx+1216]
// CHECK-NEXT:      vmovaps zmm24, [rdx+1536]
// CHECK-NEXT:      vmovaps zmm25, [rdx+1600]
// CHECK-NEXT:      vmovaps zmm26, [rdx+1664]
// CHECK-NEXT:      vmovaps zmm27, [rdx+1728]
// CHECK-NEXT:      vmovaps zmm28, [rdx+2048]
// CHECK-NEXT:      vmovaps zmm29, [rdx+2112]
// CHECK-NEXT:      vmovaps zmm30, [rdx+2176]
// CHECK-NEXT:      vmovaps zmm31, [rdx+2240]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K2_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovaps zmm1, [rdi]
// CHECK-NEXT:      vmovaps zmm2, [rdi+64]
// CHECK-NEXT:      vmovaps zmm3, [rdi+128]
// CHECK-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K2_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovaps [rdx], zmm12
// CHECK-NEXT:      vmovaps [rdx+64], zmm13
// CHECK-NEXT:      vmovaps [rdx+128], zmm14
// CHECK-NEXT:      vmovaps [rdx+192], zmm15
// CHECK-NEXT:      vmovaps [rdx+512], zmm16
// CHECK-NEXT:      vmovaps [rdx+576], zmm17
// CHECK-NEXT:      vmovaps [rdx+640], zmm18
// CHECK-NEXT:      vmovaps [rdx+704], zmm19
// CHECK-NEXT:      vmovaps [rdx+1024], zmm20
// CHECK-NEXT:      vmovaps [rdx+1088], zmm21
// CHECK-NEXT:      vmovaps [rdx+1152], zmm22
// CHECK-NEXT:      vmovaps [rdx+1216], zmm23
// CHECK-NEXT:      vmovaps [rdx+1536], zmm24
// CHECK-NEXT:      vmovaps [rdx+1600], zmm25
// CHECK-NEXT:      vmovaps [rdx+1664], zmm26
// CHECK-NEXT:      vmovaps [rdx+1728], zmm27
// CHECK-NEXT:      vmovaps [rdx+2048], zmm28
// CHECK-NEXT:      vmovaps [rdx+2112], zmm29
// CHECK-NEXT:      vmovaps [rdx+2176], zmm30
// CHECK-NEXT:      vmovaps [rdx+2240], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 65280
// CHECK-NEXT:      cmp r10, 128
// CHECK-NEXT:      jl [[SCF_M2_BODY]]
// CHECK-NEXT:      add rdx, 2048
// CHECK-NEXT:      add rsi, 2560
// CHECK-NEXT:      sub rdi, 512
// CHECK-NEXT:      cmp r11, 38
// CHECK-NEXT:      jl [[SCF_N2_BODY]]
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
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l33"
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 6 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %26 = x86.ri.add %25, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %27 = x86.dm.vmovaps [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %28 = x86.dm.vmovaps [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %29 = x86.dm.vmovaps [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.dm.vmovaps [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vmovaps [%21 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dm.vmovaps [%21 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dm.vmovaps [%21 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dm.vmovaps [%21 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.dm.vmovaps [%21 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dm.vmovaps [%21 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.dm.vmovaps [%21 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dm.vmovaps [%21 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.dm.vmovaps [%21 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dm.vmovaps [%21 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.dm.vmovaps [%21 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dm.vmovaps [%21 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.dm.vmovaps [%21 + 2048] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.dm.vmovaps [%21 + 2112] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.dm.vmovaps [%21 + 2176] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dm.vmovaps [%21 + 2240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.dm.vmovaps [%21 + 2560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.dm.vmovaps [%21 + 2624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.dm.vmovaps [%21 + 2688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.dm.vmovaps [%21 + 2752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb3(%19 : !x86.reg64<rdi>, %20 : !x86.reg64<rsi>, %21 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>, %27 : !x86.avx512reg<zmm8>, %28 : !x86.avx512reg<zmm9>, %29 : !x86.avx512reg<zmm10>, %30 : !x86.avx512reg<zmm11>, %31 : !x86.avx512reg<zmm12>, %32 : !x86.avx512reg<zmm13>, %33 : !x86.avx512reg<zmm14>, %34 : !x86.avx512reg<zmm15>, %35 : !x86.avx512reg<zmm16>, %36 : !x86.avx512reg<zmm17>, %37 : !x86.avx512reg<zmm18>, %38 : !x86.avx512reg<zmm19>, %39 : !x86.avx512reg<zmm20>, %40 : !x86.avx512reg<zmm21>, %41 : !x86.avx512reg<zmm22>, %42 : !x86.avx512reg<zmm23>, %43 : !x86.avx512reg<zmm24>, %44 : !x86.avx512reg<zmm25>, %45 : !x86.avx512reg<zmm26>, %46 : !x86.avx512reg<zmm27>, %47 : !x86.avx512reg<zmm28>, %48 : !x86.avx512reg<zmm29>, %49 : !x86.avx512reg<zmm30>, %50 : !x86.avx512reg<zmm31>, %51 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%52: !x86.reg64<rdi>, %53: !x86.reg64<rsi>, %54: !x86.reg64<rdx>, %55: !x86.reg64<rbp>, %56: !x86.reg64<rsp>, %57: !x86.reg64<r11>, %58: !x86.reg64<r10>, %59: !x86.avx512reg<zmm8>, %60: !x86.avx512reg<zmm9>, %61: !x86.avx512reg<zmm10>, %62: !x86.avx512reg<zmm11>, %63: !x86.avx512reg<zmm12>, %64: !x86.avx512reg<zmm13>, %65: !x86.avx512reg<zmm14>, %66: !x86.avx512reg<zmm15>, %67: !x86.avx512reg<zmm16>, %68: !x86.avx512reg<zmm17>, %69: !x86.avx512reg<zmm18>, %70: !x86.avx512reg<zmm19>, %71: !x86.avx512reg<zmm20>, %72: !x86.avx512reg<zmm21>, %73: !x86.avx512reg<zmm22>, %74: !x86.avx512reg<zmm23>, %75: !x86.avx512reg<zmm24>, %76: !x86.avx512reg<zmm25>, %77: !x86.avx512reg<zmm26>, %78: !x86.avx512reg<zmm27>, %79: !x86.avx512reg<zmm28>, %80: !x86.avx512reg<zmm29>, %81: !x86.avx512reg<zmm30>, %82: !x86.avx512reg<zmm31>, %83: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l35"
// CHECK-IR-LIBXSMM-NEXT:      %84 = x86.ri.add %83, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %85 = x86.dm.vmovaps [%52] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %86 = x86.dm.vmovaps [%52 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %87 = x86.dm.vmovaps [%52 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %88 = x86.dm.vmovaps [%52 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %89 = x86.dm.vbroadcastss [%53] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %90 = x86.rss.vfmadd231ps %59, %85, %89 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %91 = x86.rss.vfmadd231ps %60, %86, %89 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %92 = x86.rss.vfmadd231ps %61, %87, %89 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %93 = x86.rss.vfmadd231ps %62, %88, %89 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %94 = x86.dm.vbroadcastss [%53 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %95 = x86.rss.vfmadd231ps %63, %85, %94 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %96 = x86.rss.vfmadd231ps %64, %86, %94 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %97 = x86.rss.vfmadd231ps %65, %87, %94 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %98 = x86.rss.vfmadd231ps %66, %88, %94 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %99 = x86.dm.vbroadcastss [%53 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %100 = x86.rss.vfmadd231ps %67, %85, %99 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %101 = x86.rss.vfmadd231ps %68, %86, %99 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %102 = x86.rss.vfmadd231ps %69, %87, %99 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %103 = x86.rss.vfmadd231ps %70, %88, %99 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %104 = x86.dm.vbroadcastss [%53 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %105 = x86.rss.vfmadd231ps %71, %85, %104 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %106 = x86.rss.vfmadd231ps %72, %86, %104 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %107 = x86.rss.vfmadd231ps %73, %87, %104 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %108 = x86.rss.vfmadd231ps %74, %88, %104 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %109 = x86.dm.vbroadcastss [%53 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %110 = x86.rss.vfmadd231ps %75, %85, %109 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %111 = x86.rss.vfmadd231ps %76, %86, %109 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %112 = x86.rss.vfmadd231ps %77, %87, %109 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %113 = x86.rss.vfmadd231ps %78, %88, %109 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %114 = x86.dm.vbroadcastss [%53 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %115 = x86.ri.add %53, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.ri.add %52, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.rss.vfmadd231ps %79, %85, %114 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.rss.vfmadd231ps %80, %86, %114 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.rss.vfmadd231ps %81, %87, %114 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.rss.vfmadd231ps %82, %88, %114 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.dm.vmovaps [%116] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.dm.vmovaps [%116 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.dm.vmovaps [%116 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.dm.vmovaps [%116 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %125 = x86.dm.vbroadcastss [%115] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %126 = x86.rss.vfmadd231ps %90, %121, %125 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %127 = x86.rss.vfmadd231ps %91, %122, %125 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %128 = x86.rss.vfmadd231ps %92, %123, %125 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %129 = x86.rss.vfmadd231ps %93, %124, %125 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %130 = x86.dm.vbroadcastss [%115 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %131 = x86.rss.vfmadd231ps %95, %121, %130 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %132 = x86.rss.vfmadd231ps %96, %122, %130 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %133 = x86.rss.vfmadd231ps %97, %123, %130 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %134 = x86.rss.vfmadd231ps %98, %124, %130 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %135 = x86.dm.vbroadcastss [%115 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %136 = x86.rss.vfmadd231ps %100, %121, %135 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %137 = x86.rss.vfmadd231ps %101, %122, %135 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %138 = x86.rss.vfmadd231ps %102, %123, %135 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %139 = x86.rss.vfmadd231ps %103, %124, %135 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %140 = x86.dm.vbroadcastss [%115 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %141 = x86.rss.vfmadd231ps %105, %121, %140 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %142 = x86.rss.vfmadd231ps %106, %122, %140 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %143 = x86.rss.vfmadd231ps %107, %123, %140 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %144 = x86.rss.vfmadd231ps %108, %124, %140 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %145 = x86.dm.vbroadcastss [%115 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %146 = x86.rss.vfmadd231ps %110, %121, %145 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %147 = x86.rss.vfmadd231ps %111, %122, %145 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %148 = x86.rss.vfmadd231ps %112, %123, %145 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %149 = x86.rss.vfmadd231ps %113, %124, %145 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %150 = x86.dm.vbroadcastss [%115 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %151 = x86.ri.add %115, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.ri.add %116, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.rss.vfmadd231ps %117, %121, %150 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.rss.vfmadd231ps %118, %122, %150 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.rss.vfmadd231ps %119, %123, %150 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.rss.vfmadd231ps %120, %124, %150 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.dm.vmovaps [%152] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.dm.vmovaps [%152 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.dm.vmovaps [%152 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.dm.vmovaps [%152 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %161 = x86.dm.vbroadcastss [%151] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %162 = x86.rss.vfmadd231ps %126, %157, %161 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %163 = x86.rss.vfmadd231ps %127, %158, %161 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %164 = x86.rss.vfmadd231ps %128, %159, %161 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %165 = x86.rss.vfmadd231ps %129, %160, %161 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %166 = x86.dm.vbroadcastss [%151 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %167 = x86.rss.vfmadd231ps %131, %157, %166 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %168 = x86.rss.vfmadd231ps %132, %158, %166 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %169 = x86.rss.vfmadd231ps %133, %159, %166 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %170 = x86.rss.vfmadd231ps %134, %160, %166 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %171 = x86.dm.vbroadcastss [%151 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %172 = x86.rss.vfmadd231ps %136, %157, %171 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %173 = x86.rss.vfmadd231ps %137, %158, %171 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %174 = x86.rss.vfmadd231ps %138, %159, %171 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %175 = x86.rss.vfmadd231ps %139, %160, %171 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %176 = x86.dm.vbroadcastss [%151 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %177 = x86.rss.vfmadd231ps %141, %157, %176 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %178 = x86.rss.vfmadd231ps %142, %158, %176 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %179 = x86.rss.vfmadd231ps %143, %159, %176 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %180 = x86.rss.vfmadd231ps %144, %160, %176 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %181 = x86.dm.vbroadcastss [%151 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %182 = x86.rss.vfmadd231ps %146, %157, %181 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %183 = x86.rss.vfmadd231ps %147, %158, %181 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %184 = x86.rss.vfmadd231ps %148, %159, %181 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %185 = x86.rss.vfmadd231ps %149, %160, %181 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %186 = x86.dm.vbroadcastss [%151 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %187 = x86.ri.add %151, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %188 = x86.ri.add %152, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.rss.vfmadd231ps %153, %157, %186 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.rss.vfmadd231ps %154, %158, %186 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.rss.vfmadd231ps %155, %159, %186 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.rss.vfmadd231ps %156, %160, %186 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %193 = x86.dm.vmovaps [%188] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %194 = x86.dm.vmovaps [%188 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %195 = x86.dm.vmovaps [%188 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %196 = x86.dm.vmovaps [%188 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %197 = x86.dm.vbroadcastss [%187] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %198 = x86.rss.vfmadd231ps %162, %193, %197 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %199 = x86.rss.vfmadd231ps %163, %194, %197 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %200 = x86.rss.vfmadd231ps %164, %195, %197 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %201 = x86.rss.vfmadd231ps %165, %196, %197 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %202 = x86.dm.vbroadcastss [%187 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %203 = x86.rss.vfmadd231ps %167, %193, %202 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %204 = x86.rss.vfmadd231ps %168, %194, %202 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %205 = x86.rss.vfmadd231ps %169, %195, %202 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %206 = x86.rss.vfmadd231ps %170, %196, %202 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %207 = x86.dm.vbroadcastss [%187 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %208 = x86.rss.vfmadd231ps %172, %193, %207 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %209 = x86.rss.vfmadd231ps %173, %194, %207 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %210 = x86.rss.vfmadd231ps %174, %195, %207 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %211 = x86.rss.vfmadd231ps %175, %196, %207 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %212 = x86.dm.vbroadcastss [%187 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %213 = x86.rss.vfmadd231ps %177, %193, %212 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %214 = x86.rss.vfmadd231ps %178, %194, %212 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %215 = x86.rss.vfmadd231ps %179, %195, %212 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %216 = x86.rss.vfmadd231ps %180, %196, %212 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %217 = x86.dm.vbroadcastss [%187 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %218 = x86.rss.vfmadd231ps %182, %193, %217 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %219 = x86.rss.vfmadd231ps %183, %194, %217 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %220 = x86.rss.vfmadd231ps %184, %195, %217 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %221 = x86.rss.vfmadd231ps %185, %196, %217 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %222 = x86.dm.vbroadcastss [%187 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %223 = x86.ri.add %187, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %224 = x86.ri.add %188, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %225 = x86.rss.vfmadd231ps %189, %193, %222 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %226 = x86.rss.vfmadd231ps %190, %194, %222 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %227 = x86.rss.vfmadd231ps %191, %195, %222 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %228 = x86.rss.vfmadd231ps %192, %196, %222 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %229 = x86.si.cmp %84, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %229 : !x86.rflags<rflags>, ^bb3(%224 : !x86.reg64<rdi>, %223 : !x86.reg64<rsi>, %54 : !x86.reg64<rdx>, %55 : !x86.reg64<rbp>, %56 : !x86.reg64<rsp>, %57 : !x86.reg64<r11>, %58 : !x86.reg64<r10>, %198 : !x86.avx512reg<zmm8>, %199 : !x86.avx512reg<zmm9>, %200 : !x86.avx512reg<zmm10>, %201 : !x86.avx512reg<zmm11>, %203 : !x86.avx512reg<zmm12>, %204 : !x86.avx512reg<zmm13>, %205 : !x86.avx512reg<zmm14>, %206 : !x86.avx512reg<zmm15>, %208 : !x86.avx512reg<zmm16>, %209 : !x86.avx512reg<zmm17>, %210 : !x86.avx512reg<zmm18>, %211 : !x86.avx512reg<zmm19>, %213 : !x86.avx512reg<zmm20>, %214 : !x86.avx512reg<zmm21>, %215 : !x86.avx512reg<zmm22>, %216 : !x86.avx512reg<zmm23>, %218 : !x86.avx512reg<zmm24>, %219 : !x86.avx512reg<zmm25>, %220 : !x86.avx512reg<zmm26>, %221 : !x86.avx512reg<zmm27>, %225 : !x86.avx512reg<zmm28>, %226 : !x86.avx512reg<zmm29>, %227 : !x86.avx512reg<zmm30>, %228 : !x86.avx512reg<zmm31>, %84 : !x86.reg64<r12>), ^bb4(%224 : !x86.reg64<rdi>, %223 : !x86.reg64<rsi>, %54 : !x86.reg64<rdx>, %55 : !x86.reg64<rbp>, %56 : !x86.reg64<rsp>, %57 : !x86.reg64<r11>, %58 : !x86.reg64<r10>, %198 : !x86.avx512reg<zmm8>, %199 : !x86.avx512reg<zmm9>, %200 : !x86.avx512reg<zmm10>, %201 : !x86.avx512reg<zmm11>, %203 : !x86.avx512reg<zmm12>, %204 : !x86.avx512reg<zmm13>, %205 : !x86.avx512reg<zmm14>, %206 : !x86.avx512reg<zmm15>, %208 : !x86.avx512reg<zmm16>, %209 : !x86.avx512reg<zmm17>, %210 : !x86.avx512reg<zmm18>, %211 : !x86.avx512reg<zmm19>, %213 : !x86.avx512reg<zmm20>, %214 : !x86.avx512reg<zmm21>, %215 : !x86.avx512reg<zmm22>, %216 : !x86.avx512reg<zmm23>, %218 : !x86.avx512reg<zmm24>, %219 : !x86.avx512reg<zmm25>, %220 : !x86.avx512reg<zmm26>, %221 : !x86.avx512reg<zmm27>, %225 : !x86.avx512reg<zmm28>, %226 : !x86.avx512reg<zmm29>, %227 : !x86.avx512reg<zmm30>, %228 : !x86.avx512reg<zmm31>, %84 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb4(%230: !x86.reg64<rdi>, %231: !x86.reg64<rsi>, %232: !x86.reg64<rdx>, %233: !x86.reg64<rbp>, %234: !x86.reg64<rsp>, %235: !x86.reg64<r11>, %236: !x86.reg64<r10>, %237: !x86.avx512reg<zmm8>, %238: !x86.avx512reg<zmm9>, %239: !x86.avx512reg<zmm10>, %240: !x86.avx512reg<zmm11>, %241: !x86.avx512reg<zmm12>, %242: !x86.avx512reg<zmm13>, %243: !x86.avx512reg<zmm14>, %244: !x86.avx512reg<zmm15>, %245: !x86.avx512reg<zmm16>, %246: !x86.avx512reg<zmm17>, %247: !x86.avx512reg<zmm18>, %248: !x86.avx512reg<zmm19>, %249: !x86.avx512reg<zmm20>, %250: !x86.avx512reg<zmm21>, %251: !x86.avx512reg<zmm22>, %252: !x86.avx512reg<zmm23>, %253: !x86.avx512reg<zmm24>, %254: !x86.avx512reg<zmm25>, %255: !x86.avx512reg<zmm26>, %256: !x86.avx512reg<zmm27>, %257: !x86.avx512reg<zmm28>, %258: !x86.avx512reg<zmm29>, %259: !x86.avx512reg<zmm30>, %260: !x86.avx512reg<zmm31>, %261: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %262 = x86.ri.sub %231, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232], %237 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 64], %238 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 128], %239 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 192], %240 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 512], %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 576], %242 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 640], %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 704], %244 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1024], %245 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1088], %246 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1152], %247 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1216], %248 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1536], %249 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1600], %250 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1664], %251 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 1728], %252 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2048], %253 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2112], %254 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2176], %255 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2240], %256 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2560], %257 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2624], %258 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2688], %259 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%232 + 2752], %260 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %263 = x86.ri.add %232, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %264 = x86.ri.sub %230, 65280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %265 = x86.si.cmp %236, 128 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %265 : !x86.rflags<rflags>, ^bb2(%264 : !x86.reg64<rdi>, %262 : !x86.reg64<rsi>, %263 : !x86.reg64<rdx>, %233 : !x86.reg64<rbp>, %234 : !x86.reg64<rsp>, %235 : !x86.reg64<r11>, %236 : !x86.reg64<r10>), ^bb5(%264 : !x86.reg64<rdi>, %262 : !x86.reg64<rsi>, %263 : !x86.reg64<rdx>, %233 : !x86.reg64<rbp>, %234 : !x86.reg64<rsp>, %235 : !x86.reg64<r11>, %236 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb5(%266: !x86.reg64<rdi>, %267: !x86.reg64<rsi>, %268: !x86.reg64<rdx>, %269: !x86.reg64<rbp>, %270: !x86.reg64<rsp>, %271: !x86.reg64<r11>, %272: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %273 = x86.ri.add %268, 2560 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %274 = x86.ri.add %267, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %275 = x86.ri.sub %266, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %276 = x86.si.cmp %271, 18 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %276 : !x86.rflags<rflags>, ^bb1(%275 : !x86.reg64<rdi>, %274 : !x86.reg64<rsi>, %273 : !x86.reg64<rdx>, %269 : !x86.reg64<rbp>, %270 : !x86.reg64<rsp>, %271 : !x86.reg64<r11>), ^bb6(%275 : !x86.reg64<rdi>, %274 : !x86.reg64<rsi>, %273 : !x86.reg64<rdx>, %269 : !x86.reg64<rbp>, %270 : !x86.reg64<rsp>, %271 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb6(%277: !x86.reg64<rdi>, %278: !x86.reg64<rsi>, %279: !x86.reg64<rdx>, %280: !x86.reg64<rbp>, %281: !x86.reg64<rsp>, %282: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %283 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb7(%277 : !x86.reg64<rdi>, %278 : !x86.reg64<rsi>, %279 : !x86.reg64<rdx>, %280 : !x86.reg64<rbp>, %281 : !x86.reg64<rsp>, %283 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb7(%284: !x86.reg64<rdi>, %285: !x86.reg64<rsi>, %286: !x86.reg64<rdx>, %287: !x86.reg64<rbp>, %288: !x86.reg64<rsp>, %289: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l36"
// CHECK-IR-LIBXSMM-NEXT:      %290 = x86.ri.add %289, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %291 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb8(%284 : !x86.reg64<rdi>, %285 : !x86.reg64<rsi>, %286 : !x86.reg64<rdx>, %287 : !x86.reg64<rbp>, %288 : !x86.reg64<rsp>, %290 : !x86.reg64<r11>, %291 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb8(%292: !x86.reg64<rdi>, %293: !x86.reg64<rsi>, %294: !x86.reg64<rdx>, %295: !x86.reg64<rbp>, %296: !x86.reg64<rsp>, %297: !x86.reg64<r11>, %298: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l37"
// CHECK-IR-LIBXSMM-NEXT:      %299 = x86.ri.add %298, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %300 = x86.dm.vmovaps [%294] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %301 = x86.dm.vmovaps [%294 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %302 = x86.dm.vmovaps [%294 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %303 = x86.dm.vmovaps [%294 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %304 = x86.dm.vmovaps [%294 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %305 = x86.dm.vmovaps [%294 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %306 = x86.dm.vmovaps [%294 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %307 = x86.dm.vmovaps [%294 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %308 = x86.dm.vmovaps [%294 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %309 = x86.dm.vmovaps [%294 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %310 = x86.dm.vmovaps [%294 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %311 = x86.dm.vmovaps [%294 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %312 = x86.dm.vmovaps [%294 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %313 = x86.dm.vmovaps [%294 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.dm.vmovaps [%294 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.dm.vmovaps [%294 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.dm.vmovaps [%294 + 2048] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.dm.vmovaps [%294 + 2112] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %318 = x86.dm.vmovaps [%294 + 2176] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %319 = x86.dm.vmovaps [%294 + 2240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %320 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb9(%292 : !x86.reg64<rdi>, %293 : !x86.reg64<rsi>, %294 : !x86.reg64<rdx>, %295 : !x86.reg64<rbp>, %296 : !x86.reg64<rsp>, %297 : !x86.reg64<r11>, %299 : !x86.reg64<r10>, %300 : !x86.avx512reg<zmm12>, %301 : !x86.avx512reg<zmm13>, %302 : !x86.avx512reg<zmm14>, %303 : !x86.avx512reg<zmm15>, %304 : !x86.avx512reg<zmm16>, %305 : !x86.avx512reg<zmm17>, %306 : !x86.avx512reg<zmm18>, %307 : !x86.avx512reg<zmm19>, %308 : !x86.avx512reg<zmm20>, %309 : !x86.avx512reg<zmm21>, %310 : !x86.avx512reg<zmm22>, %311 : !x86.avx512reg<zmm23>, %312 : !x86.avx512reg<zmm24>, %313 : !x86.avx512reg<zmm25>, %314 : !x86.avx512reg<zmm26>, %315 : !x86.avx512reg<zmm27>, %316 : !x86.avx512reg<zmm28>, %317 : !x86.avx512reg<zmm29>, %318 : !x86.avx512reg<zmm30>, %319 : !x86.avx512reg<zmm31>, %320 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb9(%321: !x86.reg64<rdi>, %322: !x86.reg64<rsi>, %323: !x86.reg64<rdx>, %324: !x86.reg64<rbp>, %325: !x86.reg64<rsp>, %326: !x86.reg64<r11>, %327: !x86.reg64<r10>, %328: !x86.avx512reg<zmm12>, %329: !x86.avx512reg<zmm13>, %330: !x86.avx512reg<zmm14>, %331: !x86.avx512reg<zmm15>, %332: !x86.avx512reg<zmm16>, %333: !x86.avx512reg<zmm17>, %334: !x86.avx512reg<zmm18>, %335: !x86.avx512reg<zmm19>, %336: !x86.avx512reg<zmm20>, %337: !x86.avx512reg<zmm21>, %338: !x86.avx512reg<zmm22>, %339: !x86.avx512reg<zmm23>, %340: !x86.avx512reg<zmm24>, %341: !x86.avx512reg<zmm25>, %342: !x86.avx512reg<zmm26>, %343: !x86.avx512reg<zmm27>, %344: !x86.avx512reg<zmm28>, %345: !x86.avx512reg<zmm29>, %346: !x86.avx512reg<zmm30>, %347: !x86.avx512reg<zmm31>, %348: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l38"
// CHECK-IR-LIBXSMM-NEXT:      %349 = x86.ri.add %348, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %350 = x86.dm.vmovaps [%321] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %351 = x86.dm.vmovaps [%321 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %352 = x86.dm.vmovaps [%321 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %353 = x86.dm.vmovaps [%321 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %354 = x86.dm.vbroadcastss [%322] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %355 = x86.rss.vfmadd231ps %328, %350, %354 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %356 = x86.rss.vfmadd231ps %329, %351, %354 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %357 = x86.rss.vfmadd231ps %330, %352, %354 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %358 = x86.rss.vfmadd231ps %331, %353, %354 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %359 = x86.dm.vbroadcastss [%322 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %360 = x86.rss.vfmadd231ps %332, %350, %359 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %361 = x86.rss.vfmadd231ps %333, %351, %359 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %362 = x86.rss.vfmadd231ps %334, %352, %359 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %363 = x86.rss.vfmadd231ps %335, %353, %359 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %364 = x86.dm.vbroadcastss [%322 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %365 = x86.rss.vfmadd231ps %336, %350, %364 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %366 = x86.rss.vfmadd231ps %337, %351, %364 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.rss.vfmadd231ps %338, %352, %364 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %368 = x86.rss.vfmadd231ps %339, %353, %364 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %369 = x86.dm.vbroadcastss [%322 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %370 = x86.rss.vfmadd231ps %340, %350, %369 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %371 = x86.rss.vfmadd231ps %341, %351, %369 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %372 = x86.rss.vfmadd231ps %342, %352, %369 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %373 = x86.rss.vfmadd231ps %343, %353, %369 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %374 = x86.dm.vbroadcastss [%322 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %375 = x86.ri.add %322, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %376 = x86.ri.add %321, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %377 = x86.rss.vfmadd231ps %344, %350, %374 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %378 = x86.rss.vfmadd231ps %345, %351, %374 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %379 = x86.rss.vfmadd231ps %346, %352, %374 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %380 = x86.rss.vfmadd231ps %347, %353, %374 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %381 = x86.dm.vmovaps [%376] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %382 = x86.dm.vmovaps [%376 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %383 = x86.dm.vmovaps [%376 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %384 = x86.dm.vmovaps [%376 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %385 = x86.dm.vbroadcastss [%375] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %386 = x86.rss.vfmadd231ps %355, %381, %385 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %387 = x86.rss.vfmadd231ps %356, %382, %385 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %388 = x86.rss.vfmadd231ps %357, %383, %385 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %389 = x86.rss.vfmadd231ps %358, %384, %385 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %390 = x86.dm.vbroadcastss [%375 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %391 = x86.rss.vfmadd231ps %360, %381, %390 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %392 = x86.rss.vfmadd231ps %361, %382, %390 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %393 = x86.rss.vfmadd231ps %362, %383, %390 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %394 = x86.rss.vfmadd231ps %363, %384, %390 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %395 = x86.dm.vbroadcastss [%375 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %396 = x86.rss.vfmadd231ps %365, %381, %395 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %397 = x86.rss.vfmadd231ps %366, %382, %395 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %398 = x86.rss.vfmadd231ps %367, %383, %395 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %399 = x86.rss.vfmadd231ps %368, %384, %395 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %400 = x86.dm.vbroadcastss [%375 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %401 = x86.rss.vfmadd231ps %370, %381, %400 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %402 = x86.rss.vfmadd231ps %371, %382, %400 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %403 = x86.rss.vfmadd231ps %372, %383, %400 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %404 = x86.rss.vfmadd231ps %373, %384, %400 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %405 = x86.dm.vbroadcastss [%375 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %406 = x86.ri.add %375, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %407 = x86.ri.add %376, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %408 = x86.rss.vfmadd231ps %377, %381, %405 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %409 = x86.rss.vfmadd231ps %378, %382, %405 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %410 = x86.rss.vfmadd231ps %379, %383, %405 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %411 = x86.rss.vfmadd231ps %380, %384, %405 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %412 = x86.dm.vmovaps [%407] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %413 = x86.dm.vmovaps [%407 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %414 = x86.dm.vmovaps [%407 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %415 = x86.dm.vmovaps [%407 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %416 = x86.dm.vbroadcastss [%406] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %417 = x86.rss.vfmadd231ps %386, %412, %416 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %418 = x86.rss.vfmadd231ps %387, %413, %416 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %419 = x86.rss.vfmadd231ps %388, %414, %416 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %420 = x86.rss.vfmadd231ps %389, %415, %416 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %421 = x86.dm.vbroadcastss [%406 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %422 = x86.rss.vfmadd231ps %391, %412, %421 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %423 = x86.rss.vfmadd231ps %392, %413, %421 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %424 = x86.rss.vfmadd231ps %393, %414, %421 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %425 = x86.rss.vfmadd231ps %394, %415, %421 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %426 = x86.dm.vbroadcastss [%406 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %427 = x86.rss.vfmadd231ps %396, %412, %426 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %428 = x86.rss.vfmadd231ps %397, %413, %426 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %429 = x86.rss.vfmadd231ps %398, %414, %426 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %430 = x86.rss.vfmadd231ps %399, %415, %426 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %431 = x86.dm.vbroadcastss [%406 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %432 = x86.rss.vfmadd231ps %401, %412, %431 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %433 = x86.rss.vfmadd231ps %402, %413, %431 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %434 = x86.rss.vfmadd231ps %403, %414, %431 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %435 = x86.rss.vfmadd231ps %404, %415, %431 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %436 = x86.dm.vbroadcastss [%406 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %437 = x86.ri.add %406, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %438 = x86.ri.add %407, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %439 = x86.rss.vfmadd231ps %408, %412, %436 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %440 = x86.rss.vfmadd231ps %409, %413, %436 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %441 = x86.rss.vfmadd231ps %410, %414, %436 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %442 = x86.rss.vfmadd231ps %411, %415, %436 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %443 = x86.dm.vmovaps [%438] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %444 = x86.dm.vmovaps [%438 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %445 = x86.dm.vmovaps [%438 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %446 = x86.dm.vmovaps [%438 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %447 = x86.dm.vbroadcastss [%437] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %448 = x86.rss.vfmadd231ps %417, %443, %447 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %449 = x86.rss.vfmadd231ps %418, %444, %447 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %450 = x86.rss.vfmadd231ps %419, %445, %447 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %451 = x86.rss.vfmadd231ps %420, %446, %447 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %452 = x86.dm.vbroadcastss [%437 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %453 = x86.rss.vfmadd231ps %422, %443, %452 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %454 = x86.rss.vfmadd231ps %423, %444, %452 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %455 = x86.rss.vfmadd231ps %424, %445, %452 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %456 = x86.rss.vfmadd231ps %425, %446, %452 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %457 = x86.dm.vbroadcastss [%437 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %458 = x86.rss.vfmadd231ps %427, %443, %457 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %459 = x86.rss.vfmadd231ps %428, %444, %457 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %460 = x86.rss.vfmadd231ps %429, %445, %457 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %461 = x86.rss.vfmadd231ps %430, %446, %457 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %462 = x86.dm.vbroadcastss [%437 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %463 = x86.rss.vfmadd231ps %432, %443, %462 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %464 = x86.rss.vfmadd231ps %433, %444, %462 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %465 = x86.rss.vfmadd231ps %434, %445, %462 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %466 = x86.rss.vfmadd231ps %435, %446, %462 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %467 = x86.dm.vbroadcastss [%437 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %468 = x86.ri.add %437, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %469 = x86.ri.add %438, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %470 = x86.rss.vfmadd231ps %439, %443, %467 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %471 = x86.rss.vfmadd231ps %440, %444, %467 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %472 = x86.rss.vfmadd231ps %441, %445, %467 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %473 = x86.rss.vfmadd231ps %442, %446, %467 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %474 = x86.si.cmp %349, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %474 : !x86.rflags<rflags>, ^bb9(%469 : !x86.reg64<rdi>, %468 : !x86.reg64<rsi>, %323 : !x86.reg64<rdx>, %324 : !x86.reg64<rbp>, %325 : !x86.reg64<rsp>, %326 : !x86.reg64<r11>, %327 : !x86.reg64<r10>, %448 : !x86.avx512reg<zmm12>, %449 : !x86.avx512reg<zmm13>, %450 : !x86.avx512reg<zmm14>, %451 : !x86.avx512reg<zmm15>, %453 : !x86.avx512reg<zmm16>, %454 : !x86.avx512reg<zmm17>, %455 : !x86.avx512reg<zmm18>, %456 : !x86.avx512reg<zmm19>, %458 : !x86.avx512reg<zmm20>, %459 : !x86.avx512reg<zmm21>, %460 : !x86.avx512reg<zmm22>, %461 : !x86.avx512reg<zmm23>, %463 : !x86.avx512reg<zmm24>, %464 : !x86.avx512reg<zmm25>, %465 : !x86.avx512reg<zmm26>, %466 : !x86.avx512reg<zmm27>, %470 : !x86.avx512reg<zmm28>, %471 : !x86.avx512reg<zmm29>, %472 : !x86.avx512reg<zmm30>, %473 : !x86.avx512reg<zmm31>, %349 : !x86.reg64<r12>), ^bb10(%469 : !x86.reg64<rdi>, %468 : !x86.reg64<rsi>, %323 : !x86.reg64<rdx>, %324 : !x86.reg64<rbp>, %325 : !x86.reg64<rsp>, %326 : !x86.reg64<r11>, %327 : !x86.reg64<r10>, %448 : !x86.avx512reg<zmm12>, %449 : !x86.avx512reg<zmm13>, %450 : !x86.avx512reg<zmm14>, %451 : !x86.avx512reg<zmm15>, %453 : !x86.avx512reg<zmm16>, %454 : !x86.avx512reg<zmm17>, %455 : !x86.avx512reg<zmm18>, %456 : !x86.avx512reg<zmm19>, %458 : !x86.avx512reg<zmm20>, %459 : !x86.avx512reg<zmm21>, %460 : !x86.avx512reg<zmm22>, %461 : !x86.avx512reg<zmm23>, %463 : !x86.avx512reg<zmm24>, %464 : !x86.avx512reg<zmm25>, %465 : !x86.avx512reg<zmm26>, %466 : !x86.avx512reg<zmm27>, %470 : !x86.avx512reg<zmm28>, %471 : !x86.avx512reg<zmm29>, %472 : !x86.avx512reg<zmm30>, %473 : !x86.avx512reg<zmm31>, %349 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb10(%475: !x86.reg64<rdi>, %476: !x86.reg64<rsi>, %477: !x86.reg64<rdx>, %478: !x86.reg64<rbp>, %479: !x86.reg64<rsp>, %480: !x86.reg64<r11>, %481: !x86.reg64<r10>, %482: !x86.avx512reg<zmm12>, %483: !x86.avx512reg<zmm13>, %484: !x86.avx512reg<zmm14>, %485: !x86.avx512reg<zmm15>, %486: !x86.avx512reg<zmm16>, %487: !x86.avx512reg<zmm17>, %488: !x86.avx512reg<zmm18>, %489: !x86.avx512reg<zmm19>, %490: !x86.avx512reg<zmm20>, %491: !x86.avx512reg<zmm21>, %492: !x86.avx512reg<zmm22>, %493: !x86.avx512reg<zmm23>, %494: !x86.avx512reg<zmm24>, %495: !x86.avx512reg<zmm25>, %496: !x86.avx512reg<zmm26>, %497: !x86.avx512reg<zmm27>, %498: !x86.avx512reg<zmm28>, %499: !x86.avx512reg<zmm29>, %500: !x86.avx512reg<zmm30>, %501: !x86.avx512reg<zmm31>, %502: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %503 = x86.ri.sub %476, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477], %482 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 64], %483 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 128], %484 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 192], %485 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 512], %486 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 576], %487 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 640], %488 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 704], %489 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1024], %490 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1088], %491 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1152], %492 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1216], %493 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1536], %494 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1600], %495 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1664], %496 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 1728], %497 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 2048], %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 2112], %499 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 2176], %500 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovaps [%477 + 2240], %501 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %504 = x86.ri.add %477, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %505 = x86.ri.sub %475, 65280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %506 = x86.si.cmp %481, 128 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %506 : !x86.rflags<rflags>, ^bb8(%505 : !x86.reg64<rdi>, %503 : !x86.reg64<rsi>, %504 : !x86.reg64<rdx>, %478 : !x86.reg64<rbp>, %479 : !x86.reg64<rsp>, %480 : !x86.reg64<r11>, %481 : !x86.reg64<r10>), ^bb11(%505 : !x86.reg64<rdi>, %503 : !x86.reg64<rsi>, %504 : !x86.reg64<rdx>, %478 : !x86.reg64<rbp>, %479 : !x86.reg64<rsp>, %480 : !x86.reg64<r11>, %481 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb11(%507: !x86.reg64<rdi>, %508: !x86.reg64<rsi>, %509: !x86.reg64<rdx>, %510: !x86.reg64<rbp>, %511: !x86.reg64<rsp>, %512: !x86.reg64<r11>, %513: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %514 = x86.ri.add %509, 2048 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %515 = x86.ri.add %508, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %516 = x86.ri.sub %507, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %517 = x86.si.cmp %512, 38 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %517 : !x86.rflags<rflags>, ^bb7(%516 : !x86.reg64<rdi>, %515 : !x86.reg64<rsi>, %514 : !x86.reg64<rdx>, %510 : !x86.reg64<rbp>, %511 : !x86.reg64<rsp>, %512 : !x86.reg64<r11>), ^bb12(%516 : !x86.reg64<rdi>, %515 : !x86.reg64<rsi>, %514 : !x86.reg64<rdx>, %510 : !x86.reg64<rbp>, %511 : !x86.reg64<rsp>, %512 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb12(%518: !x86.reg64<rdi>, %519: !x86.reg64<rsi>, %520: !x86.reg64<rdx>, %521: !x86.reg64<rbp>, %522: !x86.reg64<rsp>, %523: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %524 = x86.ds.mov %521 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %525, %526 = x86.d.pop %524 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }
