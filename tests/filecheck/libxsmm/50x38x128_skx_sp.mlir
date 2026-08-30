// RUN: libxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push r15
// CHECK-MANUAL-NEXT:      push r12
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      mov rbp, rsp
// CHECK-MANUAL-NEXT:      sub rsp, 192
// CHECK-MANUAL-NEXT:      mov r10, -64
// CHECK-MANUAL-NEXT:      and rsp, r10
// CHECK-MANUAL-NEXT:      mov r11, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_0:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 6
// CHECK-MANUAL-NEXT:      mov r15, 3
// CHECK-MANUAL-NEXT:      kmovw k1, r15d
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_1:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 50
// CHECK-MANUAL-NEXT:      vmovups zmm8, [rdx]
// CHECK-MANUAL-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovups zmm11 {k1}{z}, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovups zmm12, [rdx+200]
// CHECK-MANUAL-NEXT:      vmovups zmm13, [rdx+264]
// CHECK-MANUAL-NEXT:      vmovups zmm14, [rdx+328]
// CHECK-MANUAL-NEXT:      vmovups zmm15 {k1}{z}, [rdx+392]
// CHECK-MANUAL-NEXT:      vmovups zmm16, [rdx+400]
// CHECK-MANUAL-NEXT:      vmovups zmm17, [rdx+464]
// CHECK-MANUAL-NEXT:      vmovups zmm18, [rdx+528]
// CHECK-MANUAL-NEXT:      vmovups zmm19 {k1}{z}, [rdx+592]
// CHECK-MANUAL-NEXT:      vmovups zmm20, [rdx+600]
// CHECK-MANUAL-NEXT:      vmovups zmm21, [rdx+664]
// CHECK-MANUAL-NEXT:      vmovups zmm22, [rdx+728]
// CHECK-MANUAL-NEXT:      vmovups zmm23 {k1}{z}, [rdx+792]
// CHECK-MANUAL-NEXT:      vmovups zmm24, [rdx+800]
// CHECK-MANUAL-NEXT:      vmovups zmm25, [rdx+864]
// CHECK-MANUAL-NEXT:      vmovups zmm26, [rdx+928]
// CHECK-MANUAL-NEXT:      vmovups zmm27 {k1}{z}, [rdx+992]
// CHECK-MANUAL-NEXT:      vmovups zmm28, [rdx+1000]
// CHECK-MANUAL-NEXT:      vmovups zmm29, [rdx+1064]
// CHECK-MANUAL-NEXT:      vmovups zmm30, [rdx+1128]
// CHECK-MANUAL-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1192]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_2:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm8, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm9, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm10, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm11, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      cmp r12, 128
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_2]]
// CHECK-COMPXSMM-NEXT:    sub rdi, 25400
// CHECK-MANUAL-NEXT:      sub rsi, 512
// CHECK-MANUAL-NEXT:      vmovups [rdx], zmm8
// CHECK-MANUAL-NEXT:      vmovups [rdx+64], zmm9
// CHECK-MANUAL-NEXT:      vmovups [rdx+128], zmm10
// CHECK-MANUAL-NEXT:      vmovups [rdx+192] {k1}, zmm11
// CHECK-MANUAL-NEXT:      vmovups [rdx+200], zmm12
// CHECK-MANUAL-NEXT:      vmovups [rdx+264], zmm13
// CHECK-MANUAL-NEXT:      vmovups [rdx+328], zmm14
// CHECK-MANUAL-NEXT:      vmovups [rdx+392] {k1}, zmm15
// CHECK-MANUAL-NEXT:      vmovups [rdx+400], zmm16
// CHECK-MANUAL-NEXT:      vmovups [rdx+464], zmm17
// CHECK-MANUAL-NEXT:      vmovups [rdx+528], zmm18
// CHECK-MANUAL-NEXT:      vmovups [rdx+592] {k1}, zmm19
// CHECK-MANUAL-NEXT:      vmovups [rdx+600], zmm20
// CHECK-MANUAL-NEXT:      vmovups [rdx+664], zmm21
// CHECK-MANUAL-NEXT:      vmovups [rdx+728], zmm22
// CHECK-MANUAL-NEXT:      vmovups [rdx+792] {k1}, zmm23
// CHECK-MANUAL-NEXT:      vmovups [rdx+800], zmm24
// CHECK-MANUAL-NEXT:      vmovups [rdx+864], zmm25
// CHECK-MANUAL-NEXT:      vmovups [rdx+928], zmm26
// CHECK-MANUAL-NEXT:      vmovups [rdx+992] {k1}, zmm27
// CHECK-MANUAL-NEXT:      vmovups [rdx+1000], zmm28
// CHECK-MANUAL-NEXT:      vmovups [rdx+1064], zmm29
// CHECK-MANUAL-NEXT:      vmovups [rdx+1128], zmm30
// CHECK-MANUAL-NEXT:      vmovups [rdx+1192] {k1}, zmm31
// CHECK-MANUAL-NEXT:      add rdx, 200
// CHECK-LIBXSMM-NEXT:     sub rdi, 25400
// CHECK-MANUAL-NEXT:      cmp r10, 50
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_1]]
// CHECK-LIBXSMM-NEXT:     add rdx, 1000
// CHECK-COMPXSMM-NEXT:    sub rdi, 200
// CHECK-MANUAL-NEXT:      add rsi, 3072
// CHECK-LIBXSMM-NEXT:     sub rdi, 200
// CHECK-COMPXSMM-NEXT:    add rdx, 1000
// CHECK-MANUAL-NEXT:      cmp r11, 18
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_0]]
// CHECK-MANUAL-NEXT:      mov r11, 18
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_3:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 5
// CHECK-MANUAL-NEXT:      mov r15, 3
// CHECK-MANUAL-NEXT:      kmovw k1, r15d
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_4:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 50
// CHECK-MANUAL-NEXT:      vmovups zmm12, [rdx]
// CHECK-MANUAL-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovups zmm15 {k1}{z}, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovups zmm16, [rdx+200]
// CHECK-MANUAL-NEXT:      vmovups zmm17, [rdx+264]
// CHECK-MANUAL-NEXT:      vmovups zmm18, [rdx+328]
// CHECK-MANUAL-NEXT:      vmovups zmm19 {k1}{z}, [rdx+392]
// CHECK-MANUAL-NEXT:      vmovups zmm20, [rdx+400]
// CHECK-MANUAL-NEXT:      vmovups zmm21, [rdx+464]
// CHECK-MANUAL-NEXT:      vmovups zmm22, [rdx+528]
// CHECK-MANUAL-NEXT:      vmovups zmm23 {k1}{z}, [rdx+592]
// CHECK-MANUAL-NEXT:      vmovups zmm24, [rdx+600]
// CHECK-MANUAL-NEXT:      vmovups zmm25, [rdx+664]
// CHECK-MANUAL-NEXT:      vmovups zmm26, [rdx+728]
// CHECK-MANUAL-NEXT:      vmovups zmm27 {k1}{z}, [rdx+792]
// CHECK-MANUAL-NEXT:      vmovups zmm28, [rdx+800]
// CHECK-MANUAL-NEXT:      vmovups zmm29, [rdx+864]
// CHECK-MANUAL-NEXT:      vmovups zmm30, [rdx+928]
// CHECK-MANUAL-NEXT:      vmovups zmm31 {k1}{z}, [rdx+992]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[ASM_LABEL_5:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-MANUAL-NEXT:      add rsi, 4
// CHECK-MANUAL-NEXT:      add rdi, 200
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      cmp r12, 128
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_5]]
// CHECK-COMPXSMM-NEXT:    sub rdi, 25400
// CHECK-MANUAL-NEXT:      sub rsi, 512
// CHECK-MANUAL-NEXT:      vmovups [rdx], zmm12
// CHECK-MANUAL-NEXT:      vmovups [rdx+64], zmm13
// CHECK-MANUAL-NEXT:      vmovups [rdx+128], zmm14
// CHECK-MANUAL-NEXT:      vmovups [rdx+192] {k1}, zmm15
// CHECK-MANUAL-NEXT:      vmovups [rdx+200], zmm16
// CHECK-MANUAL-NEXT:      vmovups [rdx+264], zmm17
// CHECK-MANUAL-NEXT:      vmovups [rdx+328], zmm18
// CHECK-MANUAL-NEXT:      vmovups [rdx+392] {k1}, zmm19
// CHECK-MANUAL-NEXT:      vmovups [rdx+400], zmm20
// CHECK-MANUAL-NEXT:      vmovups [rdx+464], zmm21
// CHECK-MANUAL-NEXT:      vmovups [rdx+528], zmm22
// CHECK-MANUAL-NEXT:      vmovups [rdx+592] {k1}, zmm23
// CHECK-MANUAL-NEXT:      vmovups [rdx+600], zmm24
// CHECK-MANUAL-NEXT:      vmovups [rdx+664], zmm25
// CHECK-MANUAL-NEXT:      vmovups [rdx+728], zmm26
// CHECK-MANUAL-NEXT:      vmovups [rdx+792] {k1}, zmm27
// CHECK-MANUAL-NEXT:      vmovups [rdx+800], zmm28
// CHECK-MANUAL-NEXT:      vmovups [rdx+864], zmm29
// CHECK-MANUAL-NEXT:      vmovups [rdx+928], zmm30
// CHECK-MANUAL-NEXT:      vmovups [rdx+992] {k1}, zmm31
// CHECK-MANUAL-NEXT:      add rdx, 200
// CHECK-LIBXSMM-NEXT:     sub rdi, 25400
// CHECK-MANUAL-NEXT:      cmp r10, 50
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_4]]
// CHECK-LIBXSMM-NEXT:     add rdx, 800
// CHECK-COMPXSMM-NEXT:    sub rdi, 200
// CHECK-MANUAL-NEXT:      add rsi, 2560
// CHECK-LIBXSMM-NEXT:     sub rdi, 200
// CHECK-COMPXSMM-NEXT:    add rdx, 800
// CHECK-MANUAL-NEXT:      cmp r11, 38
// CHECK-MANUAL-NEXT:      jl [[ASM_LABEL_3]]
// CHECK-MANUAL-NEXT:      mov rsp, rbp
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      pop r12
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
// CHECK-NEXT:      %17 = x86.ri.add %16, 6 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %18 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %19 = x86.ks.kmovw %18 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %20 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %20 : !x86.reg64<r10>, %19 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb2(%21: !x86.reg64<rdi>, %22: !x86.reg64<rsi>, %23: !x86.reg64<rdx>, %24: !x86.reg64<rbp>, %25: !x86.reg64<rsp>, %26: !x86.reg64<r11>, %27: !x86.reg64<r10>, %28: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %29 = x86.ri.add %27, 50 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %30 = x86.dm.vmovups [%23] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %31 = x86.dm.vmovups [%23 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %32 = x86.dm.vmovups [%23 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %33 = x86.dmk.vmovups[%23 + 192], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %34 = x86.dm.vmovups [%23 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %35 = x86.dm.vmovups [%23 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %36 = x86.dm.vmovups [%23 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %37 = x86.dmk.vmovups[%23 + 392], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %38 = x86.dm.vmovups [%23 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %39 = x86.dm.vmovups [%23 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %40 = x86.dm.vmovups [%23 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %41 = x86.dmk.vmovups[%23 + 592], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %42 = x86.dm.vmovups [%23 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %43 = x86.dm.vmovups [%23 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %44 = x86.dm.vmovups [%23 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %45 = x86.dmk.vmovups[%23 + 792], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %46 = x86.dm.vmovups [%23 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %47 = x86.dm.vmovups [%23 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %48 = x86.dm.vmovups [%23 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %49 = x86.dmk.vmovups[%23 + 992], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %50 = x86.dm.vmovups [%23 + 1000] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %51 = x86.dm.vmovups [%23 + 1064] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %52 = x86.dm.vmovups [%23 + 1128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %53 = x86.dmk.vmovups[%23 + 1192], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %54 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb3(%21 : !x86.reg64<rdi>, %22 : !x86.reg64<rsi>, %23 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %29 : !x86.reg64<r10>, %28 : !x86.avx512maskreg<k1>, %30 : !x86.avx512reg<zmm8>, %31 : !x86.avx512reg<zmm9>, %32 : !x86.avx512reg<zmm10>, %33 : !x86.avx512reg<zmm11>, %34 : !x86.avx512reg<zmm12>, %35 : !x86.avx512reg<zmm13>, %36 : !x86.avx512reg<zmm14>, %37 : !x86.avx512reg<zmm15>, %38 : !x86.avx512reg<zmm16>, %39 : !x86.avx512reg<zmm17>, %40 : !x86.avx512reg<zmm18>, %41 : !x86.avx512reg<zmm19>, %42 : !x86.avx512reg<zmm20>, %43 : !x86.avx512reg<zmm21>, %44 : !x86.avx512reg<zmm22>, %45 : !x86.avx512reg<zmm23>, %46 : !x86.avx512reg<zmm24>, %47 : !x86.avx512reg<zmm25>, %48 : !x86.avx512reg<zmm26>, %49 : !x86.avx512reg<zmm27>, %50 : !x86.avx512reg<zmm28>, %51 : !x86.avx512reg<zmm29>, %52 : !x86.avx512reg<zmm30>, %53 : !x86.avx512reg<zmm31>, %54 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb3(%55: !x86.reg64<rdi>, %56: !x86.reg64<rsi>, %57: !x86.reg64<rdx>, %58: !x86.reg64<rbp>, %59: !x86.reg64<rsp>, %60: !x86.reg64<r11>, %61: !x86.reg64<r10>, %62: !x86.avx512maskreg<k1>, %63: !x86.avx512reg<zmm8>, %64: !x86.avx512reg<zmm9>, %65: !x86.avx512reg<zmm10>, %66: !x86.avx512reg<zmm11>, %67: !x86.avx512reg<zmm12>, %68: !x86.avx512reg<zmm13>, %69: !x86.avx512reg<zmm14>, %70: !x86.avx512reg<zmm15>, %71: !x86.avx512reg<zmm16>, %72: !x86.avx512reg<zmm17>, %73: !x86.avx512reg<zmm18>, %74: !x86.avx512reg<zmm19>, %75: !x86.avx512reg<zmm20>, %76: !x86.avx512reg<zmm21>, %77: !x86.avx512reg<zmm22>, %78: !x86.avx512reg<zmm23>, %79: !x86.avx512reg<zmm24>, %80: !x86.avx512reg<zmm25>, %81: !x86.avx512reg<zmm26>, %82: !x86.avx512reg<zmm27>, %83: !x86.avx512reg<zmm28>, %84: !x86.avx512reg<zmm29>, %85: !x86.avx512reg<zmm30>, %86: !x86.avx512reg<zmm31>, %87: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l35"
// CHECK-NEXT:      %88 = x86.ri.add %87, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %89 = x86.dm.vmovups [%55] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %90 = x86.dm.vmovups [%55 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %91 = x86.dm.vmovups [%55 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %92 = x86.dmk.vmovups[%55 + 192], %62 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %93 = x86.dm.vbroadcastss [%56] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %94 = x86.rss.vfmadd231ps %63, %89, %93 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %95 = x86.rss.vfmadd231ps %64, %90, %93 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %96 = x86.rss.vfmadd231ps %65, %91, %93 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %97 = x86.rss.vfmadd231ps %66, %92, %93 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %98 = x86.dm.vbroadcastss [%56 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %99 = x86.rss.vfmadd231ps %67, %89, %98 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %100 = x86.rss.vfmadd231ps %68, %90, %98 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %101 = x86.rss.vfmadd231ps %69, %91, %98 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %102 = x86.rss.vfmadd231ps %70, %92, %98 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %103 = x86.dm.vbroadcastss [%56 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %104 = x86.rss.vfmadd231ps %71, %89, %103 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %105 = x86.rss.vfmadd231ps %72, %90, %103 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %106 = x86.rss.vfmadd231ps %73, %91, %103 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %107 = x86.rss.vfmadd231ps %74, %92, %103 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %108 = x86.dm.vbroadcastss [%56 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %109 = x86.rss.vfmadd231ps %75, %89, %108 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %110 = x86.rss.vfmadd231ps %76, %90, %108 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %111 = x86.rss.vfmadd231ps %77, %91, %108 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %112 = x86.rss.vfmadd231ps %78, %92, %108 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %113 = x86.dm.vbroadcastss [%56 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %114 = x86.rss.vfmadd231ps %79, %89, %113 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %115 = x86.rss.vfmadd231ps %80, %90, %113 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %116 = x86.rss.vfmadd231ps %81, %91, %113 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %117 = x86.rss.vfmadd231ps %82, %92, %113 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %118 = x86.dm.vbroadcastss [%56 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %119 = x86.ri.add %56, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %120 = x86.ri.add %55, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %121 = x86.rss.vfmadd231ps %83, %89, %118 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %122 = x86.rss.vfmadd231ps %84, %90, %118 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %123 = x86.rss.vfmadd231ps %85, %91, %118 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %124 = x86.rss.vfmadd231ps %86, %92, %118 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %125 = x86.dm.vmovups [%120] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %126 = x86.dm.vmovups [%120 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %127 = x86.dm.vmovups [%120 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %128 = x86.dmk.vmovups[%120 + 192], %62 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %129 = x86.dm.vbroadcastss [%119] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %130 = x86.rss.vfmadd231ps %94, %125, %129 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %131 = x86.rss.vfmadd231ps %95, %126, %129 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %132 = x86.rss.vfmadd231ps %96, %127, %129 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %133 = x86.rss.vfmadd231ps %97, %128, %129 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %134 = x86.dm.vbroadcastss [%119 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %135 = x86.rss.vfmadd231ps %99, %125, %134 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %136 = x86.rss.vfmadd231ps %100, %126, %134 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %137 = x86.rss.vfmadd231ps %101, %127, %134 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %138 = x86.rss.vfmadd231ps %102, %128, %134 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %139 = x86.dm.vbroadcastss [%119 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %140 = x86.rss.vfmadd231ps %104, %125, %139 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %141 = x86.rss.vfmadd231ps %105, %126, %139 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %142 = x86.rss.vfmadd231ps %106, %127, %139 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %143 = x86.rss.vfmadd231ps %107, %128, %139 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %144 = x86.dm.vbroadcastss [%119 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %145 = x86.rss.vfmadd231ps %109, %125, %144 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %146 = x86.rss.vfmadd231ps %110, %126, %144 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %147 = x86.rss.vfmadd231ps %111, %127, %144 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %148 = x86.rss.vfmadd231ps %112, %128, %144 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %149 = x86.dm.vbroadcastss [%119 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %150 = x86.rss.vfmadd231ps %114, %125, %149 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %151 = x86.rss.vfmadd231ps %115, %126, %149 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %152 = x86.rss.vfmadd231ps %116, %127, %149 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %153 = x86.rss.vfmadd231ps %117, %128, %149 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %154 = x86.dm.vbroadcastss [%119 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %155 = x86.ri.add %119, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %156 = x86.ri.add %120, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %157 = x86.rss.vfmadd231ps %121, %125, %154 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %158 = x86.rss.vfmadd231ps %122, %126, %154 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %159 = x86.rss.vfmadd231ps %123, %127, %154 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %160 = x86.rss.vfmadd231ps %124, %128, %154 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %161 = x86.dm.vmovups [%156] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %162 = x86.dm.vmovups [%156 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %163 = x86.dm.vmovups [%156 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %164 = x86.dmk.vmovups[%156 + 192], %62 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %165 = x86.dm.vbroadcastss [%155] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %166 = x86.rss.vfmadd231ps %130, %161, %165 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %167 = x86.rss.vfmadd231ps %131, %162, %165 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %168 = x86.rss.vfmadd231ps %132, %163, %165 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %169 = x86.rss.vfmadd231ps %133, %164, %165 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %170 = x86.dm.vbroadcastss [%155 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %171 = x86.rss.vfmadd231ps %135, %161, %170 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %172 = x86.rss.vfmadd231ps %136, %162, %170 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %173 = x86.rss.vfmadd231ps %137, %163, %170 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %174 = x86.rss.vfmadd231ps %138, %164, %170 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %175 = x86.dm.vbroadcastss [%155 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %176 = x86.rss.vfmadd231ps %140, %161, %175 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %177 = x86.rss.vfmadd231ps %141, %162, %175 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %178 = x86.rss.vfmadd231ps %142, %163, %175 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %179 = x86.rss.vfmadd231ps %143, %164, %175 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %180 = x86.dm.vbroadcastss [%155 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %181 = x86.rss.vfmadd231ps %145, %161, %180 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %182 = x86.rss.vfmadd231ps %146, %162, %180 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %183 = x86.rss.vfmadd231ps %147, %163, %180 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %184 = x86.rss.vfmadd231ps %148, %164, %180 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %185 = x86.dm.vbroadcastss [%155 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %186 = x86.rss.vfmadd231ps %150, %161, %185 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %187 = x86.rss.vfmadd231ps %151, %162, %185 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %188 = x86.rss.vfmadd231ps %152, %163, %185 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %189 = x86.rss.vfmadd231ps %153, %164, %185 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %190 = x86.dm.vbroadcastss [%155 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %191 = x86.ri.add %155, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %192 = x86.ri.add %156, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %193 = x86.rss.vfmadd231ps %157, %161, %190 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %194 = x86.rss.vfmadd231ps %158, %162, %190 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %195 = x86.rss.vfmadd231ps %159, %163, %190 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %196 = x86.rss.vfmadd231ps %160, %164, %190 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %197 = x86.dm.vmovups [%192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %198 = x86.dm.vmovups [%192 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %199 = x86.dm.vmovups [%192 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %200 = x86.dmk.vmovups[%192 + 192], %62 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %201 = x86.dm.vbroadcastss [%191] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %202 = x86.rss.vfmadd231ps %166, %197, %201 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %203 = x86.rss.vfmadd231ps %167, %198, %201 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %204 = x86.rss.vfmadd231ps %168, %199, %201 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %205 = x86.rss.vfmadd231ps %169, %200, %201 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %206 = x86.dm.vbroadcastss [%191 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %207 = x86.rss.vfmadd231ps %171, %197, %206 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %208 = x86.rss.vfmadd231ps %172, %198, %206 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %209 = x86.rss.vfmadd231ps %173, %199, %206 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %210 = x86.rss.vfmadd231ps %174, %200, %206 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %211 = x86.dm.vbroadcastss [%191 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %212 = x86.rss.vfmadd231ps %176, %197, %211 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %213 = x86.rss.vfmadd231ps %177, %198, %211 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %214 = x86.rss.vfmadd231ps %178, %199, %211 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %215 = x86.rss.vfmadd231ps %179, %200, %211 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %216 = x86.dm.vbroadcastss [%191 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %217 = x86.rss.vfmadd231ps %181, %197, %216 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %218 = x86.rss.vfmadd231ps %182, %198, %216 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %219 = x86.rss.vfmadd231ps %183, %199, %216 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %220 = x86.rss.vfmadd231ps %184, %200, %216 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %221 = x86.dm.vbroadcastss [%191 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %222 = x86.rss.vfmadd231ps %186, %197, %221 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %223 = x86.rss.vfmadd231ps %187, %198, %221 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %224 = x86.rss.vfmadd231ps %188, %199, %221 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %225 = x86.rss.vfmadd231ps %189, %200, %221 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %226 = x86.dm.vbroadcastss [%191 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %227 = x86.ri.add %191, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %228 = x86.ri.add %192, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %229 = x86.rss.vfmadd231ps %193, %197, %226 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %230 = x86.rss.vfmadd231ps %194, %198, %226 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %231 = x86.rss.vfmadd231ps %195, %199, %226 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %232 = x86.rss.vfmadd231ps %196, %200, %226 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %233 = x86.si.cmp %88, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %233 : !x86.rflags<rflags>, ^bb3(%228 : !x86.reg64<rdi>, %227 : !x86.reg64<rsi>, %57 : !x86.reg64<rdx>, %58 : !x86.reg64<rbp>, %59 : !x86.reg64<rsp>, %60 : !x86.reg64<r11>, %61 : !x86.reg64<r10>, %62 : !x86.avx512maskreg<k1>, %202 : !x86.avx512reg<zmm8>, %203 : !x86.avx512reg<zmm9>, %204 : !x86.avx512reg<zmm10>, %205 : !x86.avx512reg<zmm11>, %207 : !x86.avx512reg<zmm12>, %208 : !x86.avx512reg<zmm13>, %209 : !x86.avx512reg<zmm14>, %210 : !x86.avx512reg<zmm15>, %212 : !x86.avx512reg<zmm16>, %213 : !x86.avx512reg<zmm17>, %214 : !x86.avx512reg<zmm18>, %215 : !x86.avx512reg<zmm19>, %217 : !x86.avx512reg<zmm20>, %218 : !x86.avx512reg<zmm21>, %219 : !x86.avx512reg<zmm22>, %220 : !x86.avx512reg<zmm23>, %222 : !x86.avx512reg<zmm24>, %223 : !x86.avx512reg<zmm25>, %224 : !x86.avx512reg<zmm26>, %225 : !x86.avx512reg<zmm27>, %229 : !x86.avx512reg<zmm28>, %230 : !x86.avx512reg<zmm29>, %231 : !x86.avx512reg<zmm30>, %232 : !x86.avx512reg<zmm31>, %88 : !x86.reg64<r12>), ^bb4(%228 : !x86.reg64<rdi>, %227 : !x86.reg64<rsi>, %57 : !x86.reg64<rdx>, %58 : !x86.reg64<rbp>, %59 : !x86.reg64<rsp>, %60 : !x86.reg64<r11>, %61 : !x86.reg64<r10>, %62 : !x86.avx512maskreg<k1>, %202 : !x86.avx512reg<zmm8>, %203 : !x86.avx512reg<zmm9>, %204 : !x86.avx512reg<zmm10>, %205 : !x86.avx512reg<zmm11>, %207 : !x86.avx512reg<zmm12>, %208 : !x86.avx512reg<zmm13>, %209 : !x86.avx512reg<zmm14>, %210 : !x86.avx512reg<zmm15>, %212 : !x86.avx512reg<zmm16>, %213 : !x86.avx512reg<zmm17>, %214 : !x86.avx512reg<zmm18>, %215 : !x86.avx512reg<zmm19>, %217 : !x86.avx512reg<zmm20>, %218 : !x86.avx512reg<zmm21>, %219 : !x86.avx512reg<zmm22>, %220 : !x86.avx512reg<zmm23>, %222 : !x86.avx512reg<zmm24>, %223 : !x86.avx512reg<zmm25>, %224 : !x86.avx512reg<zmm26>, %225 : !x86.avx512reg<zmm27>, %229 : !x86.avx512reg<zmm28>, %230 : !x86.avx512reg<zmm29>, %231 : !x86.avx512reg<zmm30>, %232 : !x86.avx512reg<zmm31>, %88 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb4(%234: !x86.reg64<rdi>, %235: !x86.reg64<rsi>, %236: !x86.reg64<rdx>, %237: !x86.reg64<rbp>, %238: !x86.reg64<rsp>, %239: !x86.reg64<r11>, %240: !x86.reg64<r10>, %241: !x86.avx512maskreg<k1>, %242: !x86.avx512reg<zmm8>, %243: !x86.avx512reg<zmm9>, %244: !x86.avx512reg<zmm10>, %245: !x86.avx512reg<zmm11>, %246: !x86.avx512reg<zmm12>, %247: !x86.avx512reg<zmm13>, %248: !x86.avx512reg<zmm14>, %249: !x86.avx512reg<zmm15>, %250: !x86.avx512reg<zmm16>, %251: !x86.avx512reg<zmm17>, %252: !x86.avx512reg<zmm18>, %253: !x86.avx512reg<zmm19>, %254: !x86.avx512reg<zmm20>, %255: !x86.avx512reg<zmm21>, %256: !x86.avx512reg<zmm22>, %257: !x86.avx512reg<zmm23>, %258: !x86.avx512reg<zmm24>, %259: !x86.avx512reg<zmm25>, %260: !x86.avx512reg<zmm26>, %261: !x86.avx512reg<zmm27>, %262: !x86.avx512reg<zmm28>, %263: !x86.avx512reg<zmm29>, %264: !x86.avx512reg<zmm30>, %265: !x86.avx512reg<zmm31>, %266: !x86.reg64<r12>):
// CHECK-NEXT:      %267 = x86.ri.sub %235, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%236], %242 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 64], %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 128], %244 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%236 + 192], %245, %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 200], %246 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 264], %247 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 328], %248 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%236 + 392], %249, %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 400], %250 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 464], %251 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 528], %252 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%236 + 592], %253, %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 600], %254 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 664], %255 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 728], %256 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%236 + 792], %257, %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 800], %258 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 864], %259 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 928], %260 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%236 + 992], %261, %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 1000], %262 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 1064], %263 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%236 + 1128], %264 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%236 + 1192], %265, %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %268 = x86.ri.add %236, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %269 = x86.ri.sub %234, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %270 = x86.si.cmp %240, 50 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %270 : !x86.rflags<rflags>, ^bb2(%269 : !x86.reg64<rdi>, %267 : !x86.reg64<rsi>, %268 : !x86.reg64<rdx>, %237 : !x86.reg64<rbp>, %238 : !x86.reg64<rsp>, %239 : !x86.reg64<r11>, %240 : !x86.reg64<r10>, %241 : !x86.avx512maskreg<k1>), ^bb5(%269 : !x86.reg64<rdi>, %267 : !x86.reg64<rsi>, %268 : !x86.reg64<rdx>, %237 : !x86.reg64<rbp>, %238 : !x86.reg64<rsp>, %239 : !x86.reg64<r11>, %240 : !x86.reg64<r10>, %241 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb5(%271: !x86.reg64<rdi>, %272: !x86.reg64<rsi>, %273: !x86.reg64<rdx>, %274: !x86.reg64<rbp>, %275: !x86.reg64<rsp>, %276: !x86.reg64<r11>, %277: !x86.reg64<r10>, %278: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      %279 = x86.ri.add %273, 1000 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %280 = x86.ri.add %272, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %281 = x86.ri.sub %271, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %282 = x86.si.cmp %276, 18 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %282 : !x86.rflags<rflags>, ^bb1(%281 : !x86.reg64<rdi>, %280 : !x86.reg64<rsi>, %279 : !x86.reg64<rdx>, %274 : !x86.reg64<rbp>, %275 : !x86.reg64<rsp>, %276 : !x86.reg64<r11>), ^bb6(%281 : !x86.reg64<rdi>, %280 : !x86.reg64<rsi>, %279 : !x86.reg64<rdx>, %274 : !x86.reg64<rbp>, %275 : !x86.reg64<rsp>, %276 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb6(%283: !x86.reg64<rdi>, %284: !x86.reg64<rsi>, %285: !x86.reg64<rdx>, %286: !x86.reg64<rbp>, %287: !x86.reg64<rsp>, %288: !x86.reg64<r11>):
// CHECK-NEXT:      %289 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:      x86.fallthrough ^bb7(%283 : !x86.reg64<rdi>, %284 : !x86.reg64<rsi>, %285 : !x86.reg64<rdx>, %286 : !x86.reg64<rbp>, %287 : !x86.reg64<rsp>, %289 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb7(%290: !x86.reg64<rdi>, %291: !x86.reg64<rsi>, %292: !x86.reg64<rdx>, %293: !x86.reg64<rbp>, %294: !x86.reg64<rsp>, %295: !x86.reg64<r11>):
// CHECK-NEXT:      x86.label "l36"
// CHECK-NEXT:      %296 = x86.ri.add %295, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %297 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %298 = x86.ks.kmovw %297 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %299 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb8(%290 : !x86.reg64<rdi>, %291 : !x86.reg64<rsi>, %292 : !x86.reg64<rdx>, %293 : !x86.reg64<rbp>, %294 : !x86.reg64<rsp>, %296 : !x86.reg64<r11>, %299 : !x86.reg64<r10>, %298 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb8(%300: !x86.reg64<rdi>, %301: !x86.reg64<rsi>, %302: !x86.reg64<rdx>, %303: !x86.reg64<rbp>, %304: !x86.reg64<rsp>, %305: !x86.reg64<r11>, %306: !x86.reg64<r10>, %307: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      x86.label "l37"
// CHECK-NEXT:      %308 = x86.ri.add %306, 50 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %309 = x86.dm.vmovups [%302] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %310 = x86.dm.vmovups [%302 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %311 = x86.dm.vmovups [%302 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %312 = x86.dmk.vmovups[%302 + 192], %307 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %313 = x86.dm.vmovups [%302 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %314 = x86.dm.vmovups [%302 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %315 = x86.dm.vmovups [%302 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %316 = x86.dmk.vmovups[%302 + 392], %307 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %317 = x86.dm.vmovups [%302 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %318 = x86.dm.vmovups [%302 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %319 = x86.dm.vmovups [%302 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %320 = x86.dmk.vmovups[%302 + 592], %307 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %321 = x86.dm.vmovups [%302 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %322 = x86.dm.vmovups [%302 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %323 = x86.dm.vmovups [%302 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %324 = x86.dmk.vmovups[%302 + 792], %307 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %325 = x86.dm.vmovups [%302 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %326 = x86.dm.vmovups [%302 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %327 = x86.dm.vmovups [%302 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %328 = x86.dmk.vmovups[%302 + 992], %307 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %329 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb9(%300 : !x86.reg64<rdi>, %301 : !x86.reg64<rsi>, %302 : !x86.reg64<rdx>, %303 : !x86.reg64<rbp>, %304 : !x86.reg64<rsp>, %305 : !x86.reg64<r11>, %308 : !x86.reg64<r10>, %307 : !x86.avx512maskreg<k1>, %309 : !x86.avx512reg<zmm12>, %310 : !x86.avx512reg<zmm13>, %311 : !x86.avx512reg<zmm14>, %312 : !x86.avx512reg<zmm15>, %313 : !x86.avx512reg<zmm16>, %314 : !x86.avx512reg<zmm17>, %315 : !x86.avx512reg<zmm18>, %316 : !x86.avx512reg<zmm19>, %317 : !x86.avx512reg<zmm20>, %318 : !x86.avx512reg<zmm21>, %319 : !x86.avx512reg<zmm22>, %320 : !x86.avx512reg<zmm23>, %321 : !x86.avx512reg<zmm24>, %322 : !x86.avx512reg<zmm25>, %323 : !x86.avx512reg<zmm26>, %324 : !x86.avx512reg<zmm27>, %325 : !x86.avx512reg<zmm28>, %326 : !x86.avx512reg<zmm29>, %327 : !x86.avx512reg<zmm30>, %328 : !x86.avx512reg<zmm31>, %329 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb9(%330: !x86.reg64<rdi>, %331: !x86.reg64<rsi>, %332: !x86.reg64<rdx>, %333: !x86.reg64<rbp>, %334: !x86.reg64<rsp>, %335: !x86.reg64<r11>, %336: !x86.reg64<r10>, %337: !x86.avx512maskreg<k1>, %338: !x86.avx512reg<zmm12>, %339: !x86.avx512reg<zmm13>, %340: !x86.avx512reg<zmm14>, %341: !x86.avx512reg<zmm15>, %342: !x86.avx512reg<zmm16>, %343: !x86.avx512reg<zmm17>, %344: !x86.avx512reg<zmm18>, %345: !x86.avx512reg<zmm19>, %346: !x86.avx512reg<zmm20>, %347: !x86.avx512reg<zmm21>, %348: !x86.avx512reg<zmm22>, %349: !x86.avx512reg<zmm23>, %350: !x86.avx512reg<zmm24>, %351: !x86.avx512reg<zmm25>, %352: !x86.avx512reg<zmm26>, %353: !x86.avx512reg<zmm27>, %354: !x86.avx512reg<zmm28>, %355: !x86.avx512reg<zmm29>, %356: !x86.avx512reg<zmm30>, %357: !x86.avx512reg<zmm31>, %358: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l38"
// CHECK-NEXT:      %359 = x86.ri.add %358, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %360 = x86.dm.vmovups [%330] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %361 = x86.dm.vmovups [%330 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %362 = x86.dm.vmovups [%330 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %363 = x86.dmk.vmovups[%330 + 192], %337 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %364 = x86.dm.vbroadcastss [%331] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %365 = x86.rss.vfmadd231ps %338, %360, %364 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %366 = x86.rss.vfmadd231ps %339, %361, %364 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %367 = x86.rss.vfmadd231ps %340, %362, %364 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %368 = x86.rss.vfmadd231ps %341, %363, %364 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %369 = x86.dm.vbroadcastss [%331 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %370 = x86.rss.vfmadd231ps %342, %360, %369 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %371 = x86.rss.vfmadd231ps %343, %361, %369 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %372 = x86.rss.vfmadd231ps %344, %362, %369 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %373 = x86.rss.vfmadd231ps %345, %363, %369 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %374 = x86.dm.vbroadcastss [%331 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %375 = x86.rss.vfmadd231ps %346, %360, %374 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %376 = x86.rss.vfmadd231ps %347, %361, %374 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %377 = x86.rss.vfmadd231ps %348, %362, %374 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %378 = x86.rss.vfmadd231ps %349, %363, %374 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %379 = x86.dm.vbroadcastss [%331 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %380 = x86.rss.vfmadd231ps %350, %360, %379 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %381 = x86.rss.vfmadd231ps %351, %361, %379 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %382 = x86.rss.vfmadd231ps %352, %362, %379 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %383 = x86.rss.vfmadd231ps %353, %363, %379 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %384 = x86.dm.vbroadcastss [%331 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %385 = x86.ri.add %331, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %386 = x86.ri.add %330, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %387 = x86.rss.vfmadd231ps %354, %360, %384 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %388 = x86.rss.vfmadd231ps %355, %361, %384 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %389 = x86.rss.vfmadd231ps %356, %362, %384 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %390 = x86.rss.vfmadd231ps %357, %363, %384 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %391 = x86.dm.vmovups [%386] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %392 = x86.dm.vmovups [%386 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %393 = x86.dm.vmovups [%386 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %394 = x86.dmk.vmovups[%386 + 192], %337 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %395 = x86.dm.vbroadcastss [%385] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %396 = x86.rss.vfmadd231ps %365, %391, %395 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %397 = x86.rss.vfmadd231ps %366, %392, %395 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %398 = x86.rss.vfmadd231ps %367, %393, %395 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %399 = x86.rss.vfmadd231ps %368, %394, %395 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %400 = x86.dm.vbroadcastss [%385 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %401 = x86.rss.vfmadd231ps %370, %391, %400 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %402 = x86.rss.vfmadd231ps %371, %392, %400 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %403 = x86.rss.vfmadd231ps %372, %393, %400 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %404 = x86.rss.vfmadd231ps %373, %394, %400 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %405 = x86.dm.vbroadcastss [%385 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %406 = x86.rss.vfmadd231ps %375, %391, %405 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %407 = x86.rss.vfmadd231ps %376, %392, %405 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %408 = x86.rss.vfmadd231ps %377, %393, %405 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %409 = x86.rss.vfmadd231ps %378, %394, %405 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %410 = x86.dm.vbroadcastss [%385 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %411 = x86.rss.vfmadd231ps %380, %391, %410 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %412 = x86.rss.vfmadd231ps %381, %392, %410 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %413 = x86.rss.vfmadd231ps %382, %393, %410 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %414 = x86.rss.vfmadd231ps %383, %394, %410 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %415 = x86.dm.vbroadcastss [%385 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %416 = x86.ri.add %385, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %417 = x86.ri.add %386, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %418 = x86.rss.vfmadd231ps %387, %391, %415 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %419 = x86.rss.vfmadd231ps %388, %392, %415 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %420 = x86.rss.vfmadd231ps %389, %393, %415 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %421 = x86.rss.vfmadd231ps %390, %394, %415 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %422 = x86.dm.vmovups [%417] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %423 = x86.dm.vmovups [%417 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %424 = x86.dm.vmovups [%417 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %425 = x86.dmk.vmovups[%417 + 192], %337 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %426 = x86.dm.vbroadcastss [%416] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %427 = x86.rss.vfmadd231ps %396, %422, %426 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %428 = x86.rss.vfmadd231ps %397, %423, %426 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %429 = x86.rss.vfmadd231ps %398, %424, %426 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %430 = x86.rss.vfmadd231ps %399, %425, %426 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %431 = x86.dm.vbroadcastss [%416 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %432 = x86.rss.vfmadd231ps %401, %422, %431 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %433 = x86.rss.vfmadd231ps %402, %423, %431 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %434 = x86.rss.vfmadd231ps %403, %424, %431 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %435 = x86.rss.vfmadd231ps %404, %425, %431 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %436 = x86.dm.vbroadcastss [%416 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %437 = x86.rss.vfmadd231ps %406, %422, %436 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %438 = x86.rss.vfmadd231ps %407, %423, %436 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %439 = x86.rss.vfmadd231ps %408, %424, %436 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %440 = x86.rss.vfmadd231ps %409, %425, %436 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %441 = x86.dm.vbroadcastss [%416 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %442 = x86.rss.vfmadd231ps %411, %422, %441 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %443 = x86.rss.vfmadd231ps %412, %423, %441 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %444 = x86.rss.vfmadd231ps %413, %424, %441 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %445 = x86.rss.vfmadd231ps %414, %425, %441 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %446 = x86.dm.vbroadcastss [%416 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %447 = x86.ri.add %416, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %448 = x86.ri.add %417, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %449 = x86.rss.vfmadd231ps %418, %422, %446 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %450 = x86.rss.vfmadd231ps %419, %423, %446 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %451 = x86.rss.vfmadd231ps %420, %424, %446 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %452 = x86.rss.vfmadd231ps %421, %425, %446 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %453 = x86.dm.vmovups [%448] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %454 = x86.dm.vmovups [%448 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %455 = x86.dm.vmovups [%448 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %456 = x86.dmk.vmovups[%448 + 192], %337 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %457 = x86.dm.vbroadcastss [%447] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %458 = x86.rss.vfmadd231ps %427, %453, %457 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %459 = x86.rss.vfmadd231ps %428, %454, %457 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %460 = x86.rss.vfmadd231ps %429, %455, %457 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %461 = x86.rss.vfmadd231ps %430, %456, %457 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %462 = x86.dm.vbroadcastss [%447 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %463 = x86.rss.vfmadd231ps %432, %453, %462 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %464 = x86.rss.vfmadd231ps %433, %454, %462 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %465 = x86.rss.vfmadd231ps %434, %455, %462 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %466 = x86.rss.vfmadd231ps %435, %456, %462 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %467 = x86.dm.vbroadcastss [%447 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %468 = x86.rss.vfmadd231ps %437, %453, %467 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %469 = x86.rss.vfmadd231ps %438, %454, %467 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %470 = x86.rss.vfmadd231ps %439, %455, %467 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %471 = x86.rss.vfmadd231ps %440, %456, %467 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %472 = x86.dm.vbroadcastss [%447 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %473 = x86.rss.vfmadd231ps %442, %453, %472 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %474 = x86.rss.vfmadd231ps %443, %454, %472 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %475 = x86.rss.vfmadd231ps %444, %455, %472 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %476 = x86.rss.vfmadd231ps %445, %456, %472 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %477 = x86.dm.vbroadcastss [%447 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %478 = x86.ri.add %447, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %479 = x86.ri.add %448, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %480 = x86.rss.vfmadd231ps %449, %453, %477 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %481 = x86.rss.vfmadd231ps %450, %454, %477 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %482 = x86.rss.vfmadd231ps %451, %455, %477 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %483 = x86.rss.vfmadd231ps %452, %456, %477 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %484 = x86.si.cmp %359, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %484 : !x86.rflags<rflags>, ^bb9(%479 : !x86.reg64<rdi>, %478 : !x86.reg64<rsi>, %332 : !x86.reg64<rdx>, %333 : !x86.reg64<rbp>, %334 : !x86.reg64<rsp>, %335 : !x86.reg64<r11>, %336 : !x86.reg64<r10>, %337 : !x86.avx512maskreg<k1>, %458 : !x86.avx512reg<zmm12>, %459 : !x86.avx512reg<zmm13>, %460 : !x86.avx512reg<zmm14>, %461 : !x86.avx512reg<zmm15>, %463 : !x86.avx512reg<zmm16>, %464 : !x86.avx512reg<zmm17>, %465 : !x86.avx512reg<zmm18>, %466 : !x86.avx512reg<zmm19>, %468 : !x86.avx512reg<zmm20>, %469 : !x86.avx512reg<zmm21>, %470 : !x86.avx512reg<zmm22>, %471 : !x86.avx512reg<zmm23>, %473 : !x86.avx512reg<zmm24>, %474 : !x86.avx512reg<zmm25>, %475 : !x86.avx512reg<zmm26>, %476 : !x86.avx512reg<zmm27>, %480 : !x86.avx512reg<zmm28>, %481 : !x86.avx512reg<zmm29>, %482 : !x86.avx512reg<zmm30>, %483 : !x86.avx512reg<zmm31>, %359 : !x86.reg64<r12>), ^bb10(%479 : !x86.reg64<rdi>, %478 : !x86.reg64<rsi>, %332 : !x86.reg64<rdx>, %333 : !x86.reg64<rbp>, %334 : !x86.reg64<rsp>, %335 : !x86.reg64<r11>, %336 : !x86.reg64<r10>, %337 : !x86.avx512maskreg<k1>, %458 : !x86.avx512reg<zmm12>, %459 : !x86.avx512reg<zmm13>, %460 : !x86.avx512reg<zmm14>, %461 : !x86.avx512reg<zmm15>, %463 : !x86.avx512reg<zmm16>, %464 : !x86.avx512reg<zmm17>, %465 : !x86.avx512reg<zmm18>, %466 : !x86.avx512reg<zmm19>, %468 : !x86.avx512reg<zmm20>, %469 : !x86.avx512reg<zmm21>, %470 : !x86.avx512reg<zmm22>, %471 : !x86.avx512reg<zmm23>, %473 : !x86.avx512reg<zmm24>, %474 : !x86.avx512reg<zmm25>, %475 : !x86.avx512reg<zmm26>, %476 : !x86.avx512reg<zmm27>, %480 : !x86.avx512reg<zmm28>, %481 : !x86.avx512reg<zmm29>, %482 : !x86.avx512reg<zmm30>, %483 : !x86.avx512reg<zmm31>, %359 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb10(%485: !x86.reg64<rdi>, %486: !x86.reg64<rsi>, %487: !x86.reg64<rdx>, %488: !x86.reg64<rbp>, %489: !x86.reg64<rsp>, %490: !x86.reg64<r11>, %491: !x86.reg64<r10>, %492: !x86.avx512maskreg<k1>, %493: !x86.avx512reg<zmm12>, %494: !x86.avx512reg<zmm13>, %495: !x86.avx512reg<zmm14>, %496: !x86.avx512reg<zmm15>, %497: !x86.avx512reg<zmm16>, %498: !x86.avx512reg<zmm17>, %499: !x86.avx512reg<zmm18>, %500: !x86.avx512reg<zmm19>, %501: !x86.avx512reg<zmm20>, %502: !x86.avx512reg<zmm21>, %503: !x86.avx512reg<zmm22>, %504: !x86.avx512reg<zmm23>, %505: !x86.avx512reg<zmm24>, %506: !x86.avx512reg<zmm25>, %507: !x86.avx512reg<zmm26>, %508: !x86.avx512reg<zmm27>, %509: !x86.avx512reg<zmm28>, %510: !x86.avx512reg<zmm29>, %511: !x86.avx512reg<zmm30>, %512: !x86.avx512reg<zmm31>, %513: !x86.reg64<r12>):
// CHECK-NEXT:      %514 = x86.ri.sub %486, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%487], %493 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 64], %494 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 128], %495 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%487 + 192], %496, %492 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 200], %497 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 264], %498 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 328], %499 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%487 + 392], %500, %492 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 400], %501 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 464], %502 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 528], %503 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%487 + 592], %504, %492 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 600], %505 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 664], %506 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 728], %507 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%487 + 792], %508, %492 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 800], %509 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 864], %510 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%487 + 928], %511 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%487 + 992], %512, %492 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %515 = x86.ri.add %487, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %516 = x86.ri.sub %485, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %517 = x86.si.cmp %491, 50 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %517 : !x86.rflags<rflags>, ^bb8(%516 : !x86.reg64<rdi>, %514 : !x86.reg64<rsi>, %515 : !x86.reg64<rdx>, %488 : !x86.reg64<rbp>, %489 : !x86.reg64<rsp>, %490 : !x86.reg64<r11>, %491 : !x86.reg64<r10>, %492 : !x86.avx512maskreg<k1>), ^bb11(%516 : !x86.reg64<rdi>, %514 : !x86.reg64<rsi>, %515 : !x86.reg64<rdx>, %488 : !x86.reg64<rbp>, %489 : !x86.reg64<rsp>, %490 : !x86.reg64<r11>, %491 : !x86.reg64<r10>, %492 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb11(%518: !x86.reg64<rdi>, %519: !x86.reg64<rsi>, %520: !x86.reg64<rdx>, %521: !x86.reg64<rbp>, %522: !x86.reg64<rsp>, %523: !x86.reg64<r11>, %524: !x86.reg64<r10>, %525: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      %526 = x86.ri.add %520, 800 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %527 = x86.ri.add %519, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %528 = x86.ri.sub %518, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %529 = x86.si.cmp %523, 38 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %529 : !x86.rflags<rflags>, ^bb7(%528 : !x86.reg64<rdi>, %527 : !x86.reg64<rsi>, %526 : !x86.reg64<rdx>, %521 : !x86.reg64<rbp>, %522 : !x86.reg64<rsp>, %523 : !x86.reg64<r11>), ^bb12(%528 : !x86.reg64<rdi>, %527 : !x86.reg64<rsi>, %526 : !x86.reg64<rdx>, %521 : !x86.reg64<rbp>, %522 : !x86.reg64<rsp>, %523 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb12(%530: !x86.reg64<rdi>, %531: !x86.reg64<rsi>, %532: !x86.reg64<rdx>, %533: !x86.reg64<rbp>, %534: !x86.reg64<rsp>, %535: !x86.reg64<r11>):
// CHECK-NEXT:      %536 = x86.ds.mov %533 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %537, %538 = x86.d.pop %536 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
