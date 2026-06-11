// RUN: libxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push r12
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 6
// CHECK-NEXT:      mov r15, 3
// CHECK-NEXT:      kmovw k1, r15d
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 50
// CHECK-NEXT:      vmovups zmm8, [rdx]
// CHECK-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-NEXT:      vmovups zmm11 {k1}{z}, [rdx+192]
// CHECK-NEXT:      vmovups zmm12, [rdx+200]
// CHECK-NEXT:      vmovups zmm13, [rdx+264]
// CHECK-NEXT:      vmovups zmm14, [rdx+328]
// CHECK-NEXT:      vmovups zmm15 {k1}{z}, [rdx+392]
// CHECK-NEXT:      vmovups zmm16, [rdx+400]
// CHECK-NEXT:      vmovups zmm17, [rdx+464]
// CHECK-NEXT:      vmovups zmm18, [rdx+528]
// CHECK-NEXT:      vmovups zmm19 {k1}{z}, [rdx+592]
// CHECK-NEXT:      vmovups zmm20, [rdx+600]
// CHECK-NEXT:      vmovups zmm21, [rdx+664]
// CHECK-NEXT:      vmovups zmm22, [rdx+728]
// CHECK-NEXT:      vmovups zmm23 {k1}{z}, [rdx+792]
// CHECK-NEXT:      vmovups zmm24, [rdx+800]
// CHECK-NEXT:      vmovups zmm25, [rdx+864]
// CHECK-NEXT:      vmovups zmm26, [rdx+928]
// CHECK-NEXT:      vmovups zmm27 {k1}{z}, [rdx+992]
// CHECK-NEXT:      vmovups zmm28, [rdx+1000]
// CHECK-NEXT:      vmovups zmm29, [rdx+1064]
// CHECK-NEXT:      vmovups zmm30, [rdx+1128]
// CHECK-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1192]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovups [rdx], zmm8
// CHECK-NEXT:      vmovups [rdx+64], zmm9
// CHECK-NEXT:      vmovups [rdx+128], zmm10
// CHECK-NEXT:      vmovups [rdx+192] {k1}, zmm11
// CHECK-NEXT:      vmovups [rdx+200], zmm12
// CHECK-NEXT:      vmovups [rdx+264], zmm13
// CHECK-NEXT:      vmovups [rdx+328], zmm14
// CHECK-NEXT:      vmovups [rdx+392] {k1}, zmm15
// CHECK-NEXT:      vmovups [rdx+400], zmm16
// CHECK-NEXT:      vmovups [rdx+464], zmm17
// CHECK-NEXT:      vmovups [rdx+528], zmm18
// CHECK-NEXT:      vmovups [rdx+592] {k1}, zmm19
// CHECK-NEXT:      vmovups [rdx+600], zmm20
// CHECK-NEXT:      vmovups [rdx+664], zmm21
// CHECK-NEXT:      vmovups [rdx+728], zmm22
// CHECK-NEXT:      vmovups [rdx+792] {k1}, zmm23
// CHECK-NEXT:      vmovups [rdx+800], zmm24
// CHECK-NEXT:      vmovups [rdx+864], zmm25
// CHECK-NEXT:      vmovups [rdx+928], zmm26
// CHECK-NEXT:      vmovups [rdx+992] {k1}, zmm27
// CHECK-NEXT:      vmovups [rdx+1000], zmm28
// CHECK-NEXT:      vmovups [rdx+1064], zmm29
// CHECK-NEXT:      vmovups [rdx+1128], zmm30
// CHECK-NEXT:      vmovups [rdx+1192] {k1}, zmm31
// CHECK-NEXT:      add rdx, 200
// CHECK-NEXT:      sub rdi, 25400
// CHECK-NEXT:      cmp r10, 50
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      add rdx, 1000
// CHECK-NEXT:      add rsi, 3072
// CHECK-NEXT:      sub rdi, 200
// CHECK-NEXT:      cmp r11, 18
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov r11, 18
// CHECK-NEXT:  [[SCF_N2_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r15, 3
// CHECK-NEXT:      kmovw k1, r15d
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M2_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 50
// CHECK-NEXT:      vmovups zmm12, [rdx]
// CHECK-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-NEXT:      vmovups zmm15 {k1}{z}, [rdx+192]
// CHECK-NEXT:      vmovups zmm16, [rdx+200]
// CHECK-NEXT:      vmovups zmm17, [rdx+264]
// CHECK-NEXT:      vmovups zmm18, [rdx+328]
// CHECK-NEXT:      vmovups zmm19 {k1}{z}, [rdx+392]
// CHECK-NEXT:      vmovups zmm20, [rdx+400]
// CHECK-NEXT:      vmovups zmm21, [rdx+464]
// CHECK-NEXT:      vmovups zmm22, [rdx+528]
// CHECK-NEXT:      vmovups zmm23 {k1}{z}, [rdx+592]
// CHECK-NEXT:      vmovups zmm24, [rdx+600]
// CHECK-NEXT:      vmovups zmm25, [rdx+664]
// CHECK-NEXT:      vmovups zmm26, [rdx+728]
// CHECK-NEXT:      vmovups zmm27 {k1}{z}, [rdx+792]
// CHECK-NEXT:      vmovups zmm28, [rdx+800]
// CHECK-NEXT:      vmovups zmm29, [rdx+864]
// CHECK-NEXT:      vmovups zmm30, [rdx+928]
// CHECK-NEXT:      vmovups zmm31 {k1}{z}, [rdx+992]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K2_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-NEXT:      add rdi, 200
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K2_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovups [rdx], zmm12
// CHECK-NEXT:      vmovups [rdx+64], zmm13
// CHECK-NEXT:      vmovups [rdx+128], zmm14
// CHECK-NEXT:      vmovups [rdx+192] {k1}, zmm15
// CHECK-NEXT:      vmovups [rdx+200], zmm16
// CHECK-NEXT:      vmovups [rdx+264], zmm17
// CHECK-NEXT:      vmovups [rdx+328], zmm18
// CHECK-NEXT:      vmovups [rdx+392] {k1}, zmm19
// CHECK-NEXT:      vmovups [rdx+400], zmm20
// CHECK-NEXT:      vmovups [rdx+464], zmm21
// CHECK-NEXT:      vmovups [rdx+528], zmm22
// CHECK-NEXT:      vmovups [rdx+592] {k1}, zmm23
// CHECK-NEXT:      vmovups [rdx+600], zmm24
// CHECK-NEXT:      vmovups [rdx+664], zmm25
// CHECK-NEXT:      vmovups [rdx+728], zmm26
// CHECK-NEXT:      vmovups [rdx+792] {k1}, zmm27
// CHECK-NEXT:      vmovups [rdx+800], zmm28
// CHECK-NEXT:      vmovups [rdx+864], zmm29
// CHECK-NEXT:      vmovups [rdx+928], zmm30
// CHECK-NEXT:      vmovups [rdx+992] {k1}, zmm31
// CHECK-NEXT:      add rdx, 200
// CHECK-NEXT:      sub rdi, 25400
// CHECK-NEXT:      cmp r10, 50
// CHECK-NEXT:      jl [[SCF_M2_BODY]]
// CHECK-NEXT:      add rdx, 800
// CHECK-NEXT:      add rsi, 2560
// CHECK-NEXT:      sub rdi, 200
// CHECK-NEXT:      cmp r11, 38
// CHECK-NEXT:      jl [[SCF_N2_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r12
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret

// CHECK-IR-LIBXSMM:       builtin.module {
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
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 6 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %19 = x86.ks.kmovw %18 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %20 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r15>, %19 : !x86.avx512maskreg<k1>, %20 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%21: !x86.reg64<rdi>, %22: !x86.reg64<rsi>, %23: !x86.reg64<rdx>, %24: !x86.reg64<rbp>, %25: !x86.reg64<rsp>, %26: !x86.reg64<r11>, %27: !x86.reg64<r15>, %28: !x86.avx512maskreg<k1>, %29: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.ri.add %29, 50 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vmovups [%23] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dm.vmovups [%23 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dm.vmovups [%23 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dmk.vmovups[%23 + 192], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.dm.vmovups [%23 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dm.vmovups [%23 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.dm.vmovups [%23 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dmk.vmovups[%23 + 392], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.dm.vmovups [%23 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dm.vmovups [%23 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.dm.vmovups [%23 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dmk.vmovups[%23 + 592], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.dm.vmovups [%23 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.dm.vmovups [%23 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.dm.vmovups [%23 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dmk.vmovups[%23 + 792], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.dm.vmovups [%23 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.dm.vmovups [%23 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.dm.vmovups [%23 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.dmk.vmovups[%23 + 992], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.dm.vmovups [%23 + 1000] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %52 = x86.dm.vmovups [%23 + 1064] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %53 = x86.dm.vmovups [%23 + 1128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %54 = x86.dmk.vmovups[%23 + 1192], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %55 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb2(%21 : !x86.reg64<rdi>, %22 : !x86.reg64<rsi>, %23 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %27 : !x86.reg64<r15>, %28 : !x86.avx512maskreg<k1>, %30 : !x86.reg64<r10>, %31 : !x86.avx512reg<zmm8>, %32 : !x86.avx512reg<zmm9>, %33 : !x86.avx512reg<zmm10>, %34 : !x86.avx512reg<zmm11>, %35 : !x86.avx512reg<zmm12>, %36 : !x86.avx512reg<zmm13>, %37 : !x86.avx512reg<zmm14>, %38 : !x86.avx512reg<zmm15>, %39 : !x86.avx512reg<zmm16>, %40 : !x86.avx512reg<zmm17>, %41 : !x86.avx512reg<zmm18>, %42 : !x86.avx512reg<zmm19>, %43 : !x86.avx512reg<zmm20>, %44 : !x86.avx512reg<zmm21>, %45 : !x86.avx512reg<zmm22>, %46 : !x86.avx512reg<zmm23>, %47 : !x86.avx512reg<zmm24>, %48 : !x86.avx512reg<zmm25>, %49 : !x86.avx512reg<zmm26>, %50 : !x86.avx512reg<zmm27>, %51 : !x86.avx512reg<zmm28>, %52 : !x86.avx512reg<zmm29>, %53 : !x86.avx512reg<zmm30>, %54 : !x86.avx512reg<zmm31>, %55 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%56: !x86.reg64<rdi>, %57: !x86.reg64<rsi>, %58: !x86.reg64<rdx>, %59: !x86.reg64<rbp>, %60: !x86.reg64<rsp>, %61: !x86.reg64<r11>, %62: !x86.reg64<r15>, %63: !x86.avx512maskreg<k1>, %64: !x86.reg64<r10>, %65: !x86.avx512reg<zmm8>, %66: !x86.avx512reg<zmm9>, %67: !x86.avx512reg<zmm10>, %68: !x86.avx512reg<zmm11>, %69: !x86.avx512reg<zmm12>, %70: !x86.avx512reg<zmm13>, %71: !x86.avx512reg<zmm14>, %72: !x86.avx512reg<zmm15>, %73: !x86.avx512reg<zmm16>, %74: !x86.avx512reg<zmm17>, %75: !x86.avx512reg<zmm18>, %76: !x86.avx512reg<zmm19>, %77: !x86.avx512reg<zmm20>, %78: !x86.avx512reg<zmm21>, %79: !x86.avx512reg<zmm22>, %80: !x86.avx512reg<zmm23>, %81: !x86.avx512reg<zmm24>, %82: !x86.avx512reg<zmm25>, %83: !x86.avx512reg<zmm26>, %84: !x86.avx512reg<zmm27>, %85: !x86.avx512reg<zmm28>, %86: !x86.avx512reg<zmm29>, %87: !x86.avx512reg<zmm30>, %88: !x86.avx512reg<zmm31>, %89: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l35"
// CHECK-IR-LIBXSMM-NEXT:      %90 = x86.ri.add %89, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %91 = x86.dm.vmovups [%56] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %92 = x86.dm.vmovups [%56 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %93 = x86.dm.vmovups [%56 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %94 = x86.dmk.vmovups[%56 + 192], %63 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %95 = x86.dm.vbroadcastss [%57] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %96 = x86.rss.vfmadd231ps %65, %91, %95 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %97 = x86.rss.vfmadd231ps %66, %92, %95 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %98 = x86.rss.vfmadd231ps %67, %93, %95 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %99 = x86.rss.vfmadd231ps %68, %94, %95 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %100 = x86.dm.vbroadcastss [%57 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %101 = x86.rss.vfmadd231ps %69, %91, %100 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %102 = x86.rss.vfmadd231ps %70, %92, %100 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %103 = x86.rss.vfmadd231ps %71, %93, %100 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %104 = x86.rss.vfmadd231ps %72, %94, %100 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %105 = x86.dm.vbroadcastss [%57 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %106 = x86.rss.vfmadd231ps %73, %91, %105 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %107 = x86.rss.vfmadd231ps %74, %92, %105 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %108 = x86.rss.vfmadd231ps %75, %93, %105 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %109 = x86.rss.vfmadd231ps %76, %94, %105 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %110 = x86.dm.vbroadcastss [%57 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %111 = x86.rss.vfmadd231ps %77, %91, %110 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %112 = x86.rss.vfmadd231ps %78, %92, %110 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %113 = x86.rss.vfmadd231ps %79, %93, %110 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %114 = x86.rss.vfmadd231ps %80, %94, %110 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %115 = x86.dm.vbroadcastss [%57 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.rss.vfmadd231ps %81, %91, %115 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.rss.vfmadd231ps %82, %92, %115 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.rss.vfmadd231ps %83, %93, %115 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.rss.vfmadd231ps %84, %94, %115 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.dm.vbroadcastss [%57 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.ri.add %57, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.ri.add %56, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.rss.vfmadd231ps %85, %91, %120 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.rss.vfmadd231ps %86, %92, %120 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %125 = x86.rss.vfmadd231ps %87, %93, %120 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %126 = x86.rss.vfmadd231ps %88, %94, %120 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %127 = x86.dm.vmovups [%122] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %128 = x86.dm.vmovups [%122 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %129 = x86.dm.vmovups [%122 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %130 = x86.dmk.vmovups[%122 + 192], %63 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %131 = x86.dm.vbroadcastss [%121] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %132 = x86.rss.vfmadd231ps %96, %127, %131 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %133 = x86.rss.vfmadd231ps %97, %128, %131 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %134 = x86.rss.vfmadd231ps %98, %129, %131 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %135 = x86.rss.vfmadd231ps %99, %130, %131 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %136 = x86.dm.vbroadcastss [%121 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %137 = x86.rss.vfmadd231ps %101, %127, %136 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %138 = x86.rss.vfmadd231ps %102, %128, %136 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %139 = x86.rss.vfmadd231ps %103, %129, %136 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %140 = x86.rss.vfmadd231ps %104, %130, %136 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %141 = x86.dm.vbroadcastss [%121 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %142 = x86.rss.vfmadd231ps %106, %127, %141 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %143 = x86.rss.vfmadd231ps %107, %128, %141 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %144 = x86.rss.vfmadd231ps %108, %129, %141 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %145 = x86.rss.vfmadd231ps %109, %130, %141 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %146 = x86.dm.vbroadcastss [%121 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %147 = x86.rss.vfmadd231ps %111, %127, %146 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %148 = x86.rss.vfmadd231ps %112, %128, %146 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %149 = x86.rss.vfmadd231ps %113, %129, %146 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %150 = x86.rss.vfmadd231ps %114, %130, %146 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %151 = x86.dm.vbroadcastss [%121 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.rss.vfmadd231ps %116, %127, %151 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.rss.vfmadd231ps %117, %128, %151 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.rss.vfmadd231ps %118, %129, %151 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.rss.vfmadd231ps %119, %130, %151 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.dm.vbroadcastss [%121 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.ri.add %121, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.ri.add %122, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.rss.vfmadd231ps %123, %127, %156 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.rss.vfmadd231ps %124, %128, %156 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %161 = x86.rss.vfmadd231ps %125, %129, %156 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %162 = x86.rss.vfmadd231ps %126, %130, %156 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %163 = x86.dm.vmovups [%158] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %164 = x86.dm.vmovups [%158 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %165 = x86.dm.vmovups [%158 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %166 = x86.dmk.vmovups[%158 + 192], %63 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %167 = x86.dm.vbroadcastss [%157] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %168 = x86.rss.vfmadd231ps %132, %163, %167 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %169 = x86.rss.vfmadd231ps %133, %164, %167 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %170 = x86.rss.vfmadd231ps %134, %165, %167 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %171 = x86.rss.vfmadd231ps %135, %166, %167 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %172 = x86.dm.vbroadcastss [%157 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %173 = x86.rss.vfmadd231ps %137, %163, %172 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %174 = x86.rss.vfmadd231ps %138, %164, %172 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %175 = x86.rss.vfmadd231ps %139, %165, %172 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %176 = x86.rss.vfmadd231ps %140, %166, %172 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %177 = x86.dm.vbroadcastss [%157 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %178 = x86.rss.vfmadd231ps %142, %163, %177 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %179 = x86.rss.vfmadd231ps %143, %164, %177 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %180 = x86.rss.vfmadd231ps %144, %165, %177 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %181 = x86.rss.vfmadd231ps %145, %166, %177 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %182 = x86.dm.vbroadcastss [%157 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %183 = x86.rss.vfmadd231ps %147, %163, %182 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %184 = x86.rss.vfmadd231ps %148, %164, %182 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %185 = x86.rss.vfmadd231ps %149, %165, %182 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %186 = x86.rss.vfmadd231ps %150, %166, %182 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %187 = x86.dm.vbroadcastss [%157 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %188 = x86.rss.vfmadd231ps %152, %163, %187 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.rss.vfmadd231ps %153, %164, %187 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.rss.vfmadd231ps %154, %165, %187 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.rss.vfmadd231ps %155, %166, %187 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.dm.vbroadcastss [%157 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %193 = x86.ri.add %157, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %194 = x86.ri.add %158, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %195 = x86.rss.vfmadd231ps %159, %163, %192 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %196 = x86.rss.vfmadd231ps %160, %164, %192 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %197 = x86.rss.vfmadd231ps %161, %165, %192 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %198 = x86.rss.vfmadd231ps %162, %166, %192 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %199 = x86.dm.vmovups [%194] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %200 = x86.dm.vmovups [%194 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %201 = x86.dm.vmovups [%194 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %202 = x86.dmk.vmovups[%194 + 192], %63 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %203 = x86.dm.vbroadcastss [%193] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %204 = x86.rss.vfmadd231ps %168, %199, %203 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %205 = x86.rss.vfmadd231ps %169, %200, %203 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %206 = x86.rss.vfmadd231ps %170, %201, %203 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %207 = x86.rss.vfmadd231ps %171, %202, %203 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %208 = x86.dm.vbroadcastss [%193 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %209 = x86.rss.vfmadd231ps %173, %199, %208 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %210 = x86.rss.vfmadd231ps %174, %200, %208 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %211 = x86.rss.vfmadd231ps %175, %201, %208 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %212 = x86.rss.vfmadd231ps %176, %202, %208 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %213 = x86.dm.vbroadcastss [%193 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %214 = x86.rss.vfmadd231ps %178, %199, %213 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %215 = x86.rss.vfmadd231ps %179, %200, %213 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %216 = x86.rss.vfmadd231ps %180, %201, %213 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %217 = x86.rss.vfmadd231ps %181, %202, %213 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %218 = x86.dm.vbroadcastss [%193 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %219 = x86.rss.vfmadd231ps %183, %199, %218 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %220 = x86.rss.vfmadd231ps %184, %200, %218 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %221 = x86.rss.vfmadd231ps %185, %201, %218 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %222 = x86.rss.vfmadd231ps %186, %202, %218 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %223 = x86.dm.vbroadcastss [%193 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %224 = x86.rss.vfmadd231ps %188, %199, %223 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %225 = x86.rss.vfmadd231ps %189, %200, %223 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %226 = x86.rss.vfmadd231ps %190, %201, %223 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %227 = x86.rss.vfmadd231ps %191, %202, %223 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %228 = x86.dm.vbroadcastss [%193 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %229 = x86.ri.add %193, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %230 = x86.ri.add %194, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %231 = x86.rss.vfmadd231ps %195, %199, %228 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %232 = x86.rss.vfmadd231ps %196, %200, %228 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %233 = x86.rss.vfmadd231ps %197, %201, %228 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %234 = x86.rss.vfmadd231ps %198, %202, %228 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %235 = x86.si.cmp %90, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %235 : !x86.rflags<rflags>, ^bb2(%230 : !x86.reg64<rdi>, %229 : !x86.reg64<rsi>, %58 : !x86.reg64<rdx>, %59 : !x86.reg64<rbp>, %60 : !x86.reg64<rsp>, %61 : !x86.reg64<r11>, %62 : !x86.reg64<r15>, %63 : !x86.avx512maskreg<k1>, %64 : !x86.reg64<r10>, %204 : !x86.avx512reg<zmm8>, %205 : !x86.avx512reg<zmm9>, %206 : !x86.avx512reg<zmm10>, %207 : !x86.avx512reg<zmm11>, %209 : !x86.avx512reg<zmm12>, %210 : !x86.avx512reg<zmm13>, %211 : !x86.avx512reg<zmm14>, %212 : !x86.avx512reg<zmm15>, %214 : !x86.avx512reg<zmm16>, %215 : !x86.avx512reg<zmm17>, %216 : !x86.avx512reg<zmm18>, %217 : !x86.avx512reg<zmm19>, %219 : !x86.avx512reg<zmm20>, %220 : !x86.avx512reg<zmm21>, %221 : !x86.avx512reg<zmm22>, %222 : !x86.avx512reg<zmm23>, %224 : !x86.avx512reg<zmm24>, %225 : !x86.avx512reg<zmm25>, %226 : !x86.avx512reg<zmm26>, %227 : !x86.avx512reg<zmm27>, %231 : !x86.avx512reg<zmm28>, %232 : !x86.avx512reg<zmm29>, %233 : !x86.avx512reg<zmm30>, %234 : !x86.avx512reg<zmm31>, %90 : !x86.reg64<r12>), ^bb3(%230 : !x86.reg64<rdi>, %229 : !x86.reg64<rsi>, %58 : !x86.reg64<rdx>, %59 : !x86.reg64<rbp>, %60 : !x86.reg64<rsp>, %61 : !x86.reg64<r11>, %62 : !x86.reg64<r15>, %63 : !x86.avx512maskreg<k1>, %64 : !x86.reg64<r10>, %204 : !x86.avx512reg<zmm8>, %205 : !x86.avx512reg<zmm9>, %206 : !x86.avx512reg<zmm10>, %207 : !x86.avx512reg<zmm11>, %209 : !x86.avx512reg<zmm12>, %210 : !x86.avx512reg<zmm13>, %211 : !x86.avx512reg<zmm14>, %212 : !x86.avx512reg<zmm15>, %214 : !x86.avx512reg<zmm16>, %215 : !x86.avx512reg<zmm17>, %216 : !x86.avx512reg<zmm18>, %217 : !x86.avx512reg<zmm19>, %219 : !x86.avx512reg<zmm20>, %220 : !x86.avx512reg<zmm21>, %221 : !x86.avx512reg<zmm22>, %222 : !x86.avx512reg<zmm23>, %224 : !x86.avx512reg<zmm24>, %225 : !x86.avx512reg<zmm25>, %226 : !x86.avx512reg<zmm26>, %227 : !x86.avx512reg<zmm27>, %231 : !x86.avx512reg<zmm28>, %232 : !x86.avx512reg<zmm29>, %233 : !x86.avx512reg<zmm30>, %234 : !x86.avx512reg<zmm31>, %90 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%236: !x86.reg64<rdi>, %237: !x86.reg64<rsi>, %238: !x86.reg64<rdx>, %239: !x86.reg64<rbp>, %240: !x86.reg64<rsp>, %241: !x86.reg64<r11>, %242: !x86.reg64<r15>, %243: !x86.avx512maskreg<k1>, %244: !x86.reg64<r10>, %245: !x86.avx512reg<zmm8>, %246: !x86.avx512reg<zmm9>, %247: !x86.avx512reg<zmm10>, %248: !x86.avx512reg<zmm11>, %249: !x86.avx512reg<zmm12>, %250: !x86.avx512reg<zmm13>, %251: !x86.avx512reg<zmm14>, %252: !x86.avx512reg<zmm15>, %253: !x86.avx512reg<zmm16>, %254: !x86.avx512reg<zmm17>, %255: !x86.avx512reg<zmm18>, %256: !x86.avx512reg<zmm19>, %257: !x86.avx512reg<zmm20>, %258: !x86.avx512reg<zmm21>, %259: !x86.avx512reg<zmm22>, %260: !x86.avx512reg<zmm23>, %261: !x86.avx512reg<zmm24>, %262: !x86.avx512reg<zmm25>, %263: !x86.avx512reg<zmm26>, %264: !x86.avx512reg<zmm27>, %265: !x86.avx512reg<zmm28>, %266: !x86.avx512reg<zmm29>, %267: !x86.avx512reg<zmm30>, %268: !x86.avx512reg<zmm31>, %269: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %270 = x86.ri.sub %237, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238], %245 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 64], %246 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 128], %247 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%238 + 192], %248, %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 200], %249 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 264], %250 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 328], %251 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%238 + 392], %252, %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 400], %253 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 464], %254 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 528], %255 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%238 + 592], %256, %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 600], %257 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 664], %258 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 728], %259 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%238 + 792], %260, %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 800], %261 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 864], %262 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 928], %263 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%238 + 992], %264, %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 1000], %265 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 1064], %266 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%238 + 1128], %267 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%238 + 1192], %268, %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %271 = x86.ri.add %238, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %272 = x86.ri.sub %236, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %273 = x86.si.cmp %244, 50 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %273 : !x86.rflags<rflags>, ^bb1(%272 : !x86.reg64<rdi>, %270 : !x86.reg64<rsi>, %271 : !x86.reg64<rdx>, %239 : !x86.reg64<rbp>, %240 : !x86.reg64<rsp>, %241 : !x86.reg64<r11>, %242 : !x86.reg64<r15>, %243 : !x86.avx512maskreg<k1>, %244 : !x86.reg64<r10>), ^bb4(%272 : !x86.reg64<rdi>, %270 : !x86.reg64<rsi>, %271 : !x86.reg64<rdx>, %239 : !x86.reg64<rbp>, %240 : !x86.reg64<rsp>, %241 : !x86.reg64<r11>, %242 : !x86.reg64<r15>, %243 : !x86.avx512maskreg<k1>, %244 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb4(%274: !x86.reg64<rdi>, %275: !x86.reg64<rsi>, %276: !x86.reg64<rdx>, %277: !x86.reg64<rbp>, %278: !x86.reg64<rsp>, %279: !x86.reg64<r11>, %280: !x86.reg64<r15>, %281: !x86.avx512maskreg<k1>, %282: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %283 = x86.ri.add %276, 1000 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %284 = x86.ri.add %275, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %285 = x86.ri.sub %274, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %286 = x86.si.cmp %279, 18 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %286 : !x86.rflags<rflags>, ^bb0(%285 : !x86.reg64<rdi>, %284 : !x86.reg64<rsi>, %283 : !x86.reg64<rdx>, %277 : !x86.reg64<rbp>, %278 : !x86.reg64<rsp>, %279 : !x86.reg64<r11>), ^bb5(%285 : !x86.reg64<rdi>, %284 : !x86.reg64<rsi>, %283 : !x86.reg64<rdx>, %277 : !x86.reg64<rbp>, %278 : !x86.reg64<rsp>, %279 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb5(%287: !x86.reg64<rdi>, %288: !x86.reg64<rsi>, %289: !x86.reg64<rdx>, %290: !x86.reg64<rbp>, %291: !x86.reg64<rsp>, %292: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %293 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb6(%287 : !x86.reg64<rdi>, %288 : !x86.reg64<rsi>, %289 : !x86.reg64<rdx>, %290 : !x86.reg64<rbp>, %291 : !x86.reg64<rsp>, %293 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb6(%294: !x86.reg64<rdi>, %295: !x86.reg64<rsi>, %296: !x86.reg64<rdx>, %297: !x86.reg64<rbp>, %298: !x86.reg64<rsp>, %299: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l36"
// CHECK-IR-LIBXSMM-NEXT:      %300 = x86.ri.add %299, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %301 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %302 = x86.ks.kmovw %301 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %303 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb7(%294 : !x86.reg64<rdi>, %295 : !x86.reg64<rsi>, %296 : !x86.reg64<rdx>, %297 : !x86.reg64<rbp>, %298 : !x86.reg64<rsp>, %300 : !x86.reg64<r11>, %301 : !x86.reg64<r15>, %302 : !x86.avx512maskreg<k1>, %303 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb7(%304: !x86.reg64<rdi>, %305: !x86.reg64<rsi>, %306: !x86.reg64<rdx>, %307: !x86.reg64<rbp>, %308: !x86.reg64<rsp>, %309: !x86.reg64<r11>, %310: !x86.reg64<r15>, %311: !x86.avx512maskreg<k1>, %312: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l37"
// CHECK-IR-LIBXSMM-NEXT:      %313 = x86.ri.add %312, 50 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.dm.vmovups [%306] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.dm.vmovups [%306 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.dm.vmovups [%306 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.dmk.vmovups[%306 + 192], %311 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %318 = x86.dm.vmovups [%306 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %319 = x86.dm.vmovups [%306 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %320 = x86.dm.vmovups [%306 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %321 = x86.dmk.vmovups[%306 + 392], %311 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %322 = x86.dm.vmovups [%306 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %323 = x86.dm.vmovups [%306 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %324 = x86.dm.vmovups [%306 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %325 = x86.dmk.vmovups[%306 + 592], %311 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %326 = x86.dm.vmovups [%306 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %327 = x86.dm.vmovups [%306 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %328 = x86.dm.vmovups [%306 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %329 = x86.dmk.vmovups[%306 + 792], %311 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %330 = x86.dm.vmovups [%306 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %331 = x86.dm.vmovups [%306 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %332 = x86.dm.vmovups [%306 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %333 = x86.dmk.vmovups[%306 + 992], %311 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %334 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb8(%304 : !x86.reg64<rdi>, %305 : !x86.reg64<rsi>, %306 : !x86.reg64<rdx>, %307 : !x86.reg64<rbp>, %308 : !x86.reg64<rsp>, %309 : !x86.reg64<r11>, %310 : !x86.reg64<r15>, %311 : !x86.avx512maskreg<k1>, %313 : !x86.reg64<r10>, %314 : !x86.avx512reg<zmm12>, %315 : !x86.avx512reg<zmm13>, %316 : !x86.avx512reg<zmm14>, %317 : !x86.avx512reg<zmm15>, %318 : !x86.avx512reg<zmm16>, %319 : !x86.avx512reg<zmm17>, %320 : !x86.avx512reg<zmm18>, %321 : !x86.avx512reg<zmm19>, %322 : !x86.avx512reg<zmm20>, %323 : !x86.avx512reg<zmm21>, %324 : !x86.avx512reg<zmm22>, %325 : !x86.avx512reg<zmm23>, %326 : !x86.avx512reg<zmm24>, %327 : !x86.avx512reg<zmm25>, %328 : !x86.avx512reg<zmm26>, %329 : !x86.avx512reg<zmm27>, %330 : !x86.avx512reg<zmm28>, %331 : !x86.avx512reg<zmm29>, %332 : !x86.avx512reg<zmm30>, %333 : !x86.avx512reg<zmm31>, %334 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb8(%335: !x86.reg64<rdi>, %336: !x86.reg64<rsi>, %337: !x86.reg64<rdx>, %338: !x86.reg64<rbp>, %339: !x86.reg64<rsp>, %340: !x86.reg64<r11>, %341: !x86.reg64<r15>, %342: !x86.avx512maskreg<k1>, %343: !x86.reg64<r10>, %344: !x86.avx512reg<zmm12>, %345: !x86.avx512reg<zmm13>, %346: !x86.avx512reg<zmm14>, %347: !x86.avx512reg<zmm15>, %348: !x86.avx512reg<zmm16>, %349: !x86.avx512reg<zmm17>, %350: !x86.avx512reg<zmm18>, %351: !x86.avx512reg<zmm19>, %352: !x86.avx512reg<zmm20>, %353: !x86.avx512reg<zmm21>, %354: !x86.avx512reg<zmm22>, %355: !x86.avx512reg<zmm23>, %356: !x86.avx512reg<zmm24>, %357: !x86.avx512reg<zmm25>, %358: !x86.avx512reg<zmm26>, %359: !x86.avx512reg<zmm27>, %360: !x86.avx512reg<zmm28>, %361: !x86.avx512reg<zmm29>, %362: !x86.avx512reg<zmm30>, %363: !x86.avx512reg<zmm31>, %364: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l38"
// CHECK-IR-LIBXSMM-NEXT:      %365 = x86.ri.add %364, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %366 = x86.dm.vmovups [%335] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.dm.vmovups [%335 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %368 = x86.dm.vmovups [%335 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %369 = x86.dmk.vmovups[%335 + 192], %342 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %370 = x86.dm.vbroadcastss [%336] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %371 = x86.rss.vfmadd231ps %344, %366, %370 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %372 = x86.rss.vfmadd231ps %345, %367, %370 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %373 = x86.rss.vfmadd231ps %346, %368, %370 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %374 = x86.rss.vfmadd231ps %347, %369, %370 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %375 = x86.dm.vbroadcastss [%336 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %376 = x86.rss.vfmadd231ps %348, %366, %375 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %377 = x86.rss.vfmadd231ps %349, %367, %375 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %378 = x86.rss.vfmadd231ps %350, %368, %375 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %379 = x86.rss.vfmadd231ps %351, %369, %375 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %380 = x86.dm.vbroadcastss [%336 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %381 = x86.rss.vfmadd231ps %352, %366, %380 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %382 = x86.rss.vfmadd231ps %353, %367, %380 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %383 = x86.rss.vfmadd231ps %354, %368, %380 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %384 = x86.rss.vfmadd231ps %355, %369, %380 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %385 = x86.dm.vbroadcastss [%336 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %386 = x86.rss.vfmadd231ps %356, %366, %385 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %387 = x86.rss.vfmadd231ps %357, %367, %385 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %388 = x86.rss.vfmadd231ps %358, %368, %385 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %389 = x86.rss.vfmadd231ps %359, %369, %385 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %390 = x86.dm.vbroadcastss [%336 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %391 = x86.ri.add %336, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %392 = x86.ri.add %335, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %393 = x86.rss.vfmadd231ps %360, %366, %390 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %394 = x86.rss.vfmadd231ps %361, %367, %390 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %395 = x86.rss.vfmadd231ps %362, %368, %390 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %396 = x86.rss.vfmadd231ps %363, %369, %390 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %397 = x86.dm.vmovups [%392] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %398 = x86.dm.vmovups [%392 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %399 = x86.dm.vmovups [%392 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %400 = x86.dmk.vmovups[%392 + 192], %342 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %401 = x86.dm.vbroadcastss [%391] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %402 = x86.rss.vfmadd231ps %371, %397, %401 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %403 = x86.rss.vfmadd231ps %372, %398, %401 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %404 = x86.rss.vfmadd231ps %373, %399, %401 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %405 = x86.rss.vfmadd231ps %374, %400, %401 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %406 = x86.dm.vbroadcastss [%391 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %407 = x86.rss.vfmadd231ps %376, %397, %406 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %408 = x86.rss.vfmadd231ps %377, %398, %406 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %409 = x86.rss.vfmadd231ps %378, %399, %406 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %410 = x86.rss.vfmadd231ps %379, %400, %406 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %411 = x86.dm.vbroadcastss [%391 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %412 = x86.rss.vfmadd231ps %381, %397, %411 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %413 = x86.rss.vfmadd231ps %382, %398, %411 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %414 = x86.rss.vfmadd231ps %383, %399, %411 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %415 = x86.rss.vfmadd231ps %384, %400, %411 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %416 = x86.dm.vbroadcastss [%391 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %417 = x86.rss.vfmadd231ps %386, %397, %416 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %418 = x86.rss.vfmadd231ps %387, %398, %416 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %419 = x86.rss.vfmadd231ps %388, %399, %416 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %420 = x86.rss.vfmadd231ps %389, %400, %416 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %421 = x86.dm.vbroadcastss [%391 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %422 = x86.ri.add %391, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %423 = x86.ri.add %392, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %424 = x86.rss.vfmadd231ps %393, %397, %421 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %425 = x86.rss.vfmadd231ps %394, %398, %421 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %426 = x86.rss.vfmadd231ps %395, %399, %421 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %427 = x86.rss.vfmadd231ps %396, %400, %421 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %428 = x86.dm.vmovups [%423] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %429 = x86.dm.vmovups [%423 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %430 = x86.dm.vmovups [%423 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %431 = x86.dmk.vmovups[%423 + 192], %342 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %432 = x86.dm.vbroadcastss [%422] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %433 = x86.rss.vfmadd231ps %402, %428, %432 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %434 = x86.rss.vfmadd231ps %403, %429, %432 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %435 = x86.rss.vfmadd231ps %404, %430, %432 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %436 = x86.rss.vfmadd231ps %405, %431, %432 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %437 = x86.dm.vbroadcastss [%422 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %438 = x86.rss.vfmadd231ps %407, %428, %437 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %439 = x86.rss.vfmadd231ps %408, %429, %437 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %440 = x86.rss.vfmadd231ps %409, %430, %437 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %441 = x86.rss.vfmadd231ps %410, %431, %437 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %442 = x86.dm.vbroadcastss [%422 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %443 = x86.rss.vfmadd231ps %412, %428, %442 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %444 = x86.rss.vfmadd231ps %413, %429, %442 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %445 = x86.rss.vfmadd231ps %414, %430, %442 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %446 = x86.rss.vfmadd231ps %415, %431, %442 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %447 = x86.dm.vbroadcastss [%422 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %448 = x86.rss.vfmadd231ps %417, %428, %447 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %449 = x86.rss.vfmadd231ps %418, %429, %447 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %450 = x86.rss.vfmadd231ps %419, %430, %447 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %451 = x86.rss.vfmadd231ps %420, %431, %447 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %452 = x86.dm.vbroadcastss [%422 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %453 = x86.ri.add %422, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %454 = x86.ri.add %423, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %455 = x86.rss.vfmadd231ps %424, %428, %452 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %456 = x86.rss.vfmadd231ps %425, %429, %452 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %457 = x86.rss.vfmadd231ps %426, %430, %452 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %458 = x86.rss.vfmadd231ps %427, %431, %452 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %459 = x86.dm.vmovups [%454] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %460 = x86.dm.vmovups [%454 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %461 = x86.dm.vmovups [%454 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %462 = x86.dmk.vmovups[%454 + 192], %342 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %463 = x86.dm.vbroadcastss [%453] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %464 = x86.rss.vfmadd231ps %433, %459, %463 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %465 = x86.rss.vfmadd231ps %434, %460, %463 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %466 = x86.rss.vfmadd231ps %435, %461, %463 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %467 = x86.rss.vfmadd231ps %436, %462, %463 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %468 = x86.dm.vbroadcastss [%453 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %469 = x86.rss.vfmadd231ps %438, %459, %468 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %470 = x86.rss.vfmadd231ps %439, %460, %468 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %471 = x86.rss.vfmadd231ps %440, %461, %468 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %472 = x86.rss.vfmadd231ps %441, %462, %468 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %473 = x86.dm.vbroadcastss [%453 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %474 = x86.rss.vfmadd231ps %443, %459, %473 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %475 = x86.rss.vfmadd231ps %444, %460, %473 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %476 = x86.rss.vfmadd231ps %445, %461, %473 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %477 = x86.rss.vfmadd231ps %446, %462, %473 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %478 = x86.dm.vbroadcastss [%453 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %479 = x86.rss.vfmadd231ps %448, %459, %478 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %480 = x86.rss.vfmadd231ps %449, %460, %478 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %481 = x86.rss.vfmadd231ps %450, %461, %478 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %482 = x86.rss.vfmadd231ps %451, %462, %478 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %483 = x86.dm.vbroadcastss [%453 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %484 = x86.ri.add %453, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %485 = x86.ri.add %454, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %486 = x86.rss.vfmadd231ps %455, %459, %483 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %487 = x86.rss.vfmadd231ps %456, %460, %483 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %488 = x86.rss.vfmadd231ps %457, %461, %483 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %489 = x86.rss.vfmadd231ps %458, %462, %483 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %490 = x86.si.cmp %365, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %490 : !x86.rflags<rflags>, ^bb8(%485 : !x86.reg64<rdi>, %484 : !x86.reg64<rsi>, %337 : !x86.reg64<rdx>, %338 : !x86.reg64<rbp>, %339 : !x86.reg64<rsp>, %340 : !x86.reg64<r11>, %341 : !x86.reg64<r15>, %342 : !x86.avx512maskreg<k1>, %343 : !x86.reg64<r10>, %464 : !x86.avx512reg<zmm12>, %465 : !x86.avx512reg<zmm13>, %466 : !x86.avx512reg<zmm14>, %467 : !x86.avx512reg<zmm15>, %469 : !x86.avx512reg<zmm16>, %470 : !x86.avx512reg<zmm17>, %471 : !x86.avx512reg<zmm18>, %472 : !x86.avx512reg<zmm19>, %474 : !x86.avx512reg<zmm20>, %475 : !x86.avx512reg<zmm21>, %476 : !x86.avx512reg<zmm22>, %477 : !x86.avx512reg<zmm23>, %479 : !x86.avx512reg<zmm24>, %480 : !x86.avx512reg<zmm25>, %481 : !x86.avx512reg<zmm26>, %482 : !x86.avx512reg<zmm27>, %486 : !x86.avx512reg<zmm28>, %487 : !x86.avx512reg<zmm29>, %488 : !x86.avx512reg<zmm30>, %489 : !x86.avx512reg<zmm31>, %365 : !x86.reg64<r12>), ^bb9(%485 : !x86.reg64<rdi>, %484 : !x86.reg64<rsi>, %337 : !x86.reg64<rdx>, %338 : !x86.reg64<rbp>, %339 : !x86.reg64<rsp>, %340 : !x86.reg64<r11>, %341 : !x86.reg64<r15>, %342 : !x86.avx512maskreg<k1>, %343 : !x86.reg64<r10>, %464 : !x86.avx512reg<zmm12>, %465 : !x86.avx512reg<zmm13>, %466 : !x86.avx512reg<zmm14>, %467 : !x86.avx512reg<zmm15>, %469 : !x86.avx512reg<zmm16>, %470 : !x86.avx512reg<zmm17>, %471 : !x86.avx512reg<zmm18>, %472 : !x86.avx512reg<zmm19>, %474 : !x86.avx512reg<zmm20>, %475 : !x86.avx512reg<zmm21>, %476 : !x86.avx512reg<zmm22>, %477 : !x86.avx512reg<zmm23>, %479 : !x86.avx512reg<zmm24>, %480 : !x86.avx512reg<zmm25>, %481 : !x86.avx512reg<zmm26>, %482 : !x86.avx512reg<zmm27>, %486 : !x86.avx512reg<zmm28>, %487 : !x86.avx512reg<zmm29>, %488 : !x86.avx512reg<zmm30>, %489 : !x86.avx512reg<zmm31>, %365 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb9(%491: !x86.reg64<rdi>, %492: !x86.reg64<rsi>, %493: !x86.reg64<rdx>, %494: !x86.reg64<rbp>, %495: !x86.reg64<rsp>, %496: !x86.reg64<r11>, %497: !x86.reg64<r15>, %498: !x86.avx512maskreg<k1>, %499: !x86.reg64<r10>, %500: !x86.avx512reg<zmm12>, %501: !x86.avx512reg<zmm13>, %502: !x86.avx512reg<zmm14>, %503: !x86.avx512reg<zmm15>, %504: !x86.avx512reg<zmm16>, %505: !x86.avx512reg<zmm17>, %506: !x86.avx512reg<zmm18>, %507: !x86.avx512reg<zmm19>, %508: !x86.avx512reg<zmm20>, %509: !x86.avx512reg<zmm21>, %510: !x86.avx512reg<zmm22>, %511: !x86.avx512reg<zmm23>, %512: !x86.avx512reg<zmm24>, %513: !x86.avx512reg<zmm25>, %514: !x86.avx512reg<zmm26>, %515: !x86.avx512reg<zmm27>, %516: !x86.avx512reg<zmm28>, %517: !x86.avx512reg<zmm29>, %518: !x86.avx512reg<zmm30>, %519: !x86.avx512reg<zmm31>, %520: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %521 = x86.ri.sub %492, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493], %500 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 64], %501 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 128], %502 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%493 + 192], %503, %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 200], %504 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 264], %505 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 328], %506 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%493 + 392], %507, %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 400], %508 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 464], %509 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 528], %510 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%493 + 592], %511, %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 600], %512 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 664], %513 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 728], %514 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%493 + 792], %515, %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 800], %516 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 864], %517 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%493 + 928], %518 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%493 + 992], %519, %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %522 = x86.ri.add %493, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %523 = x86.ri.sub %491, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %524 = x86.si.cmp %499, 50 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %524 : !x86.rflags<rflags>, ^bb7(%523 : !x86.reg64<rdi>, %521 : !x86.reg64<rsi>, %522 : !x86.reg64<rdx>, %494 : !x86.reg64<rbp>, %495 : !x86.reg64<rsp>, %496 : !x86.reg64<r11>, %497 : !x86.reg64<r15>, %498 : !x86.avx512maskreg<k1>, %499 : !x86.reg64<r10>), ^bb10(%523 : !x86.reg64<rdi>, %521 : !x86.reg64<rsi>, %522 : !x86.reg64<rdx>, %494 : !x86.reg64<rbp>, %495 : !x86.reg64<rsp>, %496 : !x86.reg64<r11>, %497 : !x86.reg64<r15>, %498 : !x86.avx512maskreg<k1>, %499 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb10(%525: !x86.reg64<rdi>, %526: !x86.reg64<rsi>, %527: !x86.reg64<rdx>, %528: !x86.reg64<rbp>, %529: !x86.reg64<rsp>, %530: !x86.reg64<r11>, %531: !x86.reg64<r15>, %532: !x86.avx512maskreg<k1>, %533: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %534 = x86.ri.add %527, 800 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %535 = x86.ri.add %526, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %536 = x86.ri.sub %525, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %537 = x86.si.cmp %530, 38 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %537 : !x86.rflags<rflags>, ^bb6(%536 : !x86.reg64<rdi>, %535 : !x86.reg64<rsi>, %534 : !x86.reg64<rdx>, %528 : !x86.reg64<rbp>, %529 : !x86.reg64<rsp>, %530 : !x86.reg64<r11>), ^bb11(%536 : !x86.reg64<rdi>, %535 : !x86.reg64<rsi>, %534 : !x86.reg64<rdx>, %528 : !x86.reg64<rbp>, %529 : !x86.reg64<rsp>, %530 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb11(%538: !x86.reg64<rdi>, %539: !x86.reg64<rsi>, %540: !x86.reg64<rdx>, %541: !x86.reg64<rbp>, %542: !x86.reg64<rsp>, %543: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %544 = x86.ds.mov %541 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %545, %546 = x86.d.pop %544 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }

