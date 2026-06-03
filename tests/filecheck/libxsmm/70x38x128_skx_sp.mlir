// RUN: libxsmm-gemm  dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm  dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r12
// CHECK-NEXT:      push r15
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
// CHECK-NEXT:      vmovups zmm8, [rdx]
// CHECK-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-NEXT:      vmovups zmm11, [rdx+192]
// CHECK-NEXT:      vmovups zmm12, [rdx+280]
// CHECK-NEXT:      vmovups zmm13, [rdx+344]
// CHECK-NEXT:      vmovups zmm14, [rdx+408]
// CHECK-NEXT:      vmovups zmm15, [rdx+472]
// CHECK-NEXT:      vmovups zmm16, [rdx+560]
// CHECK-NEXT:      vmovups zmm17, [rdx+624]
// CHECK-NEXT:      vmovups zmm18, [rdx+688]
// CHECK-NEXT:      vmovups zmm19, [rdx+752]
// CHECK-NEXT:      vmovups zmm20, [rdx+840]
// CHECK-NEXT:      vmovups zmm21, [rdx+904]
// CHECK-NEXT:      vmovups zmm22, [rdx+968]
// CHECK-NEXT:      vmovups zmm23, [rdx+1032]
// CHECK-NEXT:      vmovups zmm24, [rdx+1120]
// CHECK-NEXT:      vmovups zmm25, [rdx+1184]
// CHECK-NEXT:      vmovups zmm26, [rdx+1248]
// CHECK-NEXT:      vmovups zmm27, [rdx+1312]
// CHECK-NEXT:      vmovups zmm28, [rdx+1400]
// CHECK-NEXT:      vmovups zmm29, [rdx+1464]
// CHECK-NEXT:      vmovups zmm30, [rdx+1528]
// CHECK-NEXT:      vmovups zmm31, [rdx+1592]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
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
// CHECK-NEXT:      vmovups [rdx+192], zmm11
// CHECK-NEXT:      vmovups [rdx+280], zmm12
// CHECK-NEXT:      vmovups [rdx+344], zmm13
// CHECK-NEXT:      vmovups [rdx+408], zmm14
// CHECK-NEXT:      vmovups [rdx+472], zmm15
// CHECK-NEXT:      vmovups [rdx+560], zmm16
// CHECK-NEXT:      vmovups [rdx+624], zmm17
// CHECK-NEXT:      vmovups [rdx+688], zmm18
// CHECK-NEXT:      vmovups [rdx+752], zmm19
// CHECK-NEXT:      vmovups [rdx+840], zmm20
// CHECK-NEXT:      vmovups [rdx+904], zmm21
// CHECK-NEXT:      vmovups [rdx+968], zmm22
// CHECK-NEXT:      vmovups [rdx+1032], zmm23
// CHECK-NEXT:      vmovups [rdx+1120], zmm24
// CHECK-NEXT:      vmovups [rdx+1184], zmm25
// CHECK-NEXT:      vmovups [rdx+1248], zmm26
// CHECK-NEXT:      vmovups [rdx+1312], zmm27
// CHECK-NEXT:      vmovups [rdx+1400], zmm28
// CHECK-NEXT:      vmovups [rdx+1464], zmm29
// CHECK-NEXT:      vmovups [rdx+1528], zmm30
// CHECK-NEXT:      vmovups [rdx+1592], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 35584
// CHECK-NEXT:      cmp r10, 64
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      mov r15, 63
// CHECK-NEXT:      kmovw k1, r15d
// CHECK-NEXT:      mov r10, 64
// CHECK-NEXT:  [[SCF_M2_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 6
// CHECK-NEXT:      vmovups zmm26 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovups zmm27 {k1}{z}, [rdx+280]
// CHECK-NEXT:      vmovups zmm28 {k1}{z}, [rdx+560]
// CHECK-NEXT:      vmovups zmm29 {k1}{z}, [rdx+840]
// CHECK-NEXT:      vmovups zmm30 {k1}{z}, [rdx+1120]
// CHECK-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1400]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K2_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm26, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm26, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm26, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm26, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K2_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovups [rdx] {k1}, zmm26
// CHECK-NEXT:      vmovups [rdx+280] {k1}, zmm27
// CHECK-NEXT:      vmovups [rdx+560] {k1}, zmm28
// CHECK-NEXT:      vmovups [rdx+840] {k1}, zmm29
// CHECK-NEXT:      vmovups [rdx+1120] {k1}, zmm30
// CHECK-NEXT:      vmovups [rdx+1400] {k1}, zmm31
// CHECK-NEXT:      add rdx, 24
// CHECK-NEXT:      sub rdi, 35816
// CHECK-NEXT:      cmp r10, 70
// CHECK-NEXT:      jl [[SCF_M2_BODY]]
// CHECK-NEXT:      add rdx, 1400
// CHECK-NEXT:      add rsi, 3072
// CHECK-NEXT:      sub rdi, 280
// CHECK-NEXT:      cmp r11, 18
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov r11, 18
// CHECK-NEXT:  [[SCF_N2_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M3_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 64
// CHECK-NEXT:      vmovups zmm12, [rdx]
// CHECK-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-NEXT:      vmovups zmm15, [rdx+192]
// CHECK-NEXT:      vmovups zmm16, [rdx+280]
// CHECK-NEXT:      vmovups zmm17, [rdx+344]
// CHECK-NEXT:      vmovups zmm18, [rdx+408]
// CHECK-NEXT:      vmovups zmm19, [rdx+472]
// CHECK-NEXT:      vmovups zmm20, [rdx+560]
// CHECK-NEXT:      vmovups zmm21, [rdx+624]
// CHECK-NEXT:      vmovups zmm22, [rdx+688]
// CHECK-NEXT:      vmovups zmm23, [rdx+752]
// CHECK-NEXT:      vmovups zmm24, [rdx+840]
// CHECK-NEXT:      vmovups zmm25, [rdx+904]
// CHECK-NEXT:      vmovups zmm26, [rdx+968]
// CHECK-NEXT:      vmovups zmm27, [rdx+1032]
// CHECK-NEXT:      vmovups zmm28, [rdx+1120]
// CHECK-NEXT:      vmovups zmm29, [rdx+1184]
// CHECK-NEXT:      vmovups zmm30, [rdx+1248]
// CHECK-NEXT:      vmovups zmm31, [rdx+1312]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K3_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K3_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovups [rdx], zmm12
// CHECK-NEXT:      vmovups [rdx+64], zmm13
// CHECK-NEXT:      vmovups [rdx+128], zmm14
// CHECK-NEXT:      vmovups [rdx+192], zmm15
// CHECK-NEXT:      vmovups [rdx+280], zmm16
// CHECK-NEXT:      vmovups [rdx+344], zmm17
// CHECK-NEXT:      vmovups [rdx+408], zmm18
// CHECK-NEXT:      vmovups [rdx+472], zmm19
// CHECK-NEXT:      vmovups [rdx+560], zmm20
// CHECK-NEXT:      vmovups [rdx+624], zmm21
// CHECK-NEXT:      vmovups [rdx+688], zmm22
// CHECK-NEXT:      vmovups [rdx+752], zmm23
// CHECK-NEXT:      vmovups [rdx+840], zmm24
// CHECK-NEXT:      vmovups [rdx+904], zmm25
// CHECK-NEXT:      vmovups [rdx+968], zmm26
// CHECK-NEXT:      vmovups [rdx+1032], zmm27
// CHECK-NEXT:      vmovups [rdx+1120], zmm28
// CHECK-NEXT:      vmovups [rdx+1184], zmm29
// CHECK-NEXT:      vmovups [rdx+1248], zmm30
// CHECK-NEXT:      vmovups [rdx+1312], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 35584
// CHECK-NEXT:      cmp r10, 64
// CHECK-NEXT:      jl [[SCF_M3_BODY]]
// CHECK-NEXT:      mov r15, 63
// CHECK-NEXT:      kmovw k1, r15d
// CHECK-NEXT:      mov r10, 64
// CHECK-NEXT:  [[SCF_M4_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 6
// CHECK-NEXT:      vmovups zmm27 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovups zmm28 {k1}{z}, [rdx+280]
// CHECK-NEXT:      vmovups zmm29 {k1}{z}, [rdx+560]
// CHECK-NEXT:      vmovups zmm30 {k1}{z}, [rdx+840]
// CHECK-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1120]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[SCF_K4_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-NEXT:      vfmadd231ps zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-NEXT:      vfmadd231ps zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-NEXT:      cmp r12, 128
// CHECK-NEXT:      jl [[SCF_K4_BODY]]
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovups [rdx] {k1}, zmm27
// CHECK-NEXT:      vmovups [rdx+280] {k1}, zmm28
// CHECK-NEXT:      vmovups [rdx+560] {k1}, zmm29
// CHECK-NEXT:      vmovups [rdx+840] {k1}, zmm30
// CHECK-NEXT:      vmovups [rdx+1120] {k1}, zmm31
// CHECK-NEXT:      add rdx, 24
// CHECK-NEXT:      sub rdi, 35816
// CHECK-NEXT:      cmp r10, 70
// CHECK-NEXT:      jl [[SCF_M4_BODY]]
// CHECK-NEXT:      add rdx, 1120
// CHECK-NEXT:      add rsi, 2560
// CHECK-NEXT:      sub rdi, 280
// CHECK-NEXT:      cmp r11, 38
// CHECK-NEXT:      jl [[SCF_N2_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r15
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
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 6 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %26 = x86.ri.add %25, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %27 = x86.dm.vmovups [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-IR-LIBXSMM-NEXT:      %28 = x86.dm.vmovups [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-IR-LIBXSMM-NEXT:      %29 = x86.dm.vmovups [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.dm.vmovups [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vmovups [%21 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dm.vmovups [%21 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dm.vmovups [%21 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dm.vmovups [%21 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.dm.vmovups [%21 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dm.vmovups [%21 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.dm.vmovups [%21 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dm.vmovups [%21 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.dm.vmovups [%21 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dm.vmovups [%21 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.dm.vmovups [%21 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dm.vmovups [%21 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.dm.vmovups [%21 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.dm.vmovups [%21 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.dm.vmovups [%21 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dm.vmovups [%21 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.dm.vmovups [%21 + 1400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.dm.vmovups [%21 + 1464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.dm.vmovups [%21 + 1528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.dm.vmovups [%21 + 1592] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb2(%19 : !x86.reg64<rdi>, %20 : !x86.reg64<rsi>, %21 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>, %27 : !x86.avx512reg<zmm8>, %28 : !x86.avx512reg<zmm9>, %29 : !x86.avx512reg<zmm10>, %30 : !x86.avx512reg<zmm11>, %31 : !x86.avx512reg<zmm12>, %32 : !x86.avx512reg<zmm13>, %33 : !x86.avx512reg<zmm14>, %34 : !x86.avx512reg<zmm15>, %35 : !x86.avx512reg<zmm16>, %36 : !x86.avx512reg<zmm17>, %37 : !x86.avx512reg<zmm18>, %38 : !x86.avx512reg<zmm19>, %39 : !x86.avx512reg<zmm20>, %40 : !x86.avx512reg<zmm21>, %41 : !x86.avx512reg<zmm22>, %42 : !x86.avx512reg<zmm23>, %43 : !x86.avx512reg<zmm24>, %44 : !x86.avx512reg<zmm25>, %45 : !x86.avx512reg<zmm26>, %46 : !x86.avx512reg<zmm27>, %47 : !x86.avx512reg<zmm28>, %48 : !x86.avx512reg<zmm29>, %49 : !x86.avx512reg<zmm30>, %50 : !x86.avx512reg<zmm31>, %51 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%52: !x86.reg64<rdi>, %53: !x86.reg64<rsi>, %54: !x86.reg64<rdx>, %55: !x86.reg64<rbp>, %56: !x86.reg64<rsp>, %57: !x86.reg64<r11>, %58: !x86.reg64<r10>, %59: !x86.avx512reg<zmm8>, %60: !x86.avx512reg<zmm9>, %61: !x86.avx512reg<zmm10>, %62: !x86.avx512reg<zmm11>, %63: !x86.avx512reg<zmm12>, %64: !x86.avx512reg<zmm13>, %65: !x86.avx512reg<zmm14>, %66: !x86.avx512reg<zmm15>, %67: !x86.avx512reg<zmm16>, %68: !x86.avx512reg<zmm17>, %69: !x86.avx512reg<zmm18>, %70: !x86.avx512reg<zmm19>, %71: !x86.avx512reg<zmm20>, %72: !x86.avx512reg<zmm21>, %73: !x86.avx512reg<zmm22>, %74: !x86.avx512reg<zmm23>, %75: !x86.avx512reg<zmm24>, %76: !x86.avx512reg<zmm25>, %77: !x86.avx512reg<zmm26>, %78: !x86.avx512reg<zmm27>, %79: !x86.avx512reg<zmm28>, %80: !x86.avx512reg<zmm29>, %81: !x86.avx512reg<zmm30>, %82: !x86.avx512reg<zmm31>, %83: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l35"
// CHECK-IR-LIBXSMM-NEXT:      %84 = x86.ri.add %83, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %85 = x86.dm.vmovups [%52] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %86 = x86.dm.vmovups [%52 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %87 = x86.dm.vmovups [%52 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %88 = x86.dm.vmovups [%52 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.ri.add %52, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.rss.vfmadd231ps %79, %85, %114 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.rss.vfmadd231ps %80, %86, %114 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.rss.vfmadd231ps %81, %87, %114 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.rss.vfmadd231ps %82, %88, %114 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.dm.vmovups [%116] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.dm.vmovups [%116 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.dm.vmovups [%116 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.dm.vmovups [%116 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.ri.add %116, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.rss.vfmadd231ps %117, %121, %150 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.rss.vfmadd231ps %118, %122, %150 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.rss.vfmadd231ps %119, %123, %150 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.rss.vfmadd231ps %120, %124, %150 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.dm.vmovups [%152] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.dm.vmovups [%152 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.dm.vmovups [%152 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.dm.vmovups [%152 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-IR-LIBXSMM-NEXT:      %188 = x86.ri.add %152, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.rss.vfmadd231ps %153, %157, %186 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.rss.vfmadd231ps %154, %158, %186 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.rss.vfmadd231ps %155, %159, %186 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.rss.vfmadd231ps %156, %160, %186 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %193 = x86.dm.vmovups [%188] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %194 = x86.dm.vmovups [%188 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %195 = x86.dm.vmovups [%188 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %196 = x86.dm.vmovups [%188 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-IR-LIBXSMM-NEXT:      %224 = x86.ri.add %188, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %225 = x86.rss.vfmadd231ps %189, %193, %222 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %226 = x86.rss.vfmadd231ps %190, %194, %222 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %227 = x86.rss.vfmadd231ps %191, %195, %222 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %228 = x86.rss.vfmadd231ps %192, %196, %222 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %229 = x86.si.cmp %84, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %229 : !x86.rflags<rflags>, ^bb2(%224 : !x86.reg64<rdi>, %223 : !x86.reg64<rsi>, %54 : !x86.reg64<rdx>, %55 : !x86.reg64<rbp>, %56 : !x86.reg64<rsp>, %57 : !x86.reg64<r11>, %58 : !x86.reg64<r10>, %198 : !x86.avx512reg<zmm8>, %199 : !x86.avx512reg<zmm9>, %200 : !x86.avx512reg<zmm10>, %201 : !x86.avx512reg<zmm11>, %203 : !x86.avx512reg<zmm12>, %204 : !x86.avx512reg<zmm13>, %205 : !x86.avx512reg<zmm14>, %206 : !x86.avx512reg<zmm15>, %208 : !x86.avx512reg<zmm16>, %209 : !x86.avx512reg<zmm17>, %210 : !x86.avx512reg<zmm18>, %211 : !x86.avx512reg<zmm19>, %213 : !x86.avx512reg<zmm20>, %214 : !x86.avx512reg<zmm21>, %215 : !x86.avx512reg<zmm22>, %216 : !x86.avx512reg<zmm23>, %218 : !x86.avx512reg<zmm24>, %219 : !x86.avx512reg<zmm25>, %220 : !x86.avx512reg<zmm26>, %221 : !x86.avx512reg<zmm27>, %225 : !x86.avx512reg<zmm28>, %226 : !x86.avx512reg<zmm29>, %227 : !x86.avx512reg<zmm30>, %228 : !x86.avx512reg<zmm31>, %84 : !x86.reg64<r12>), ^bb3(%224 : !x86.reg64<rdi>, %223 : !x86.reg64<rsi>, %54 : !x86.reg64<rdx>, %55 : !x86.reg64<rbp>, %56 : !x86.reg64<rsp>, %57 : !x86.reg64<r11>, %58 : !x86.reg64<r10>, %198 : !x86.avx512reg<zmm8>, %199 : !x86.avx512reg<zmm9>, %200 : !x86.avx512reg<zmm10>, %201 : !x86.avx512reg<zmm11>, %203 : !x86.avx512reg<zmm12>, %204 : !x86.avx512reg<zmm13>, %205 : !x86.avx512reg<zmm14>, %206 : !x86.avx512reg<zmm15>, %208 : !x86.avx512reg<zmm16>, %209 : !x86.avx512reg<zmm17>, %210 : !x86.avx512reg<zmm18>, %211 : !x86.avx512reg<zmm19>, %213 : !x86.avx512reg<zmm20>, %214 : !x86.avx512reg<zmm21>, %215 : !x86.avx512reg<zmm22>, %216 : !x86.avx512reg<zmm23>, %218 : !x86.avx512reg<zmm24>, %219 : !x86.avx512reg<zmm25>, %220 : !x86.avx512reg<zmm26>, %221 : !x86.avx512reg<zmm27>, %225 : !x86.avx512reg<zmm28>, %226 : !x86.avx512reg<zmm29>, %227 : !x86.avx512reg<zmm30>, %228 : !x86.avx512reg<zmm31>, %84 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%230: !x86.reg64<rdi>, %231: !x86.reg64<rsi>, %232: !x86.reg64<rdx>, %233: !x86.reg64<rbp>, %234: !x86.reg64<rsp>, %235: !x86.reg64<r11>, %236: !x86.reg64<r10>, %237: !x86.avx512reg<zmm8>, %238: !x86.avx512reg<zmm9>, %239: !x86.avx512reg<zmm10>, %240: !x86.avx512reg<zmm11>, %241: !x86.avx512reg<zmm12>, %242: !x86.avx512reg<zmm13>, %243: !x86.avx512reg<zmm14>, %244: !x86.avx512reg<zmm15>, %245: !x86.avx512reg<zmm16>, %246: !x86.avx512reg<zmm17>, %247: !x86.avx512reg<zmm18>, %248: !x86.avx512reg<zmm19>, %249: !x86.avx512reg<zmm20>, %250: !x86.avx512reg<zmm21>, %251: !x86.avx512reg<zmm22>, %252: !x86.avx512reg<zmm23>, %253: !x86.avx512reg<zmm24>, %254: !x86.avx512reg<zmm25>, %255: !x86.avx512reg<zmm26>, %256: !x86.avx512reg<zmm27>, %257: !x86.avx512reg<zmm28>, %258: !x86.avx512reg<zmm29>, %259: !x86.avx512reg<zmm30>, %260: !x86.avx512reg<zmm31>, %261: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %262 = x86.ri.sub %231, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232], %237 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 64], %238 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 128], %239 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 192], %240 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 280], %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 344], %242 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 408], %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 472], %244 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 560], %245 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 624], %246 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 688], %247 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 752], %248 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 840], %249 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 904], %250 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 968], %251 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1032], %252 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1120], %253 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1184], %254 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1248], %255 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1312], %256 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1400], %257 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1464], %258 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1528], %259 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%232 + 1592], %260 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %263 = x86.ri.add %232, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %264 = x86.ri.sub %230, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %265 = x86.si.cmp %236, 64 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %265 : !x86.rflags<rflags>, ^bb1(%264 : !x86.reg64<rdi>, %262 : !x86.reg64<rsi>, %263 : !x86.reg64<rdx>, %233 : !x86.reg64<rbp>, %234 : !x86.reg64<rsp>, %235 : !x86.reg64<r11>, %236 : !x86.reg64<r10>), ^bb4(%264 : !x86.reg64<rdi>, %262 : !x86.reg64<rsi>, %263 : !x86.reg64<rdx>, %233 : !x86.reg64<rbp>, %234 : !x86.reg64<rsp>, %235 : !x86.reg64<r11>, %236 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb4(%266: !x86.reg64<rdi>, %267: !x86.reg64<rsi>, %268: !x86.reg64<rdx>, %269: !x86.reg64<rbp>, %270: !x86.reg64<rsp>, %271: !x86.reg64<r11>, %272: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %273 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %274 = x86.ks.kmovw %273 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %275 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb5(%266 : !x86.reg64<rdi>, %267 : !x86.reg64<rsi>, %268 : !x86.reg64<rdx>, %269 : !x86.reg64<rbp>, %270 : !x86.reg64<rsp>, %271 : !x86.reg64<r11>, %275 : !x86.reg64<r10>, %274 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb5(%276: !x86.reg64<rdi>, %277: !x86.reg64<rsi>, %278: !x86.reg64<rdx>, %279: !x86.reg64<rbp>, %280: !x86.reg64<rsp>, %281: !x86.reg64<r11>, %282: !x86.reg64<r10>, %283: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l36"
// CHECK-IR-LIBXSMM-NEXT:      %284 = x86.ri.add %282, 6 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %285 = x86.dmk.vmovups[%278], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %286 = x86.dmk.vmovups[%278 + 280], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %287 = x86.dmk.vmovups[%278 + 560], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %288 = x86.dmk.vmovups[%278 + 840], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %289 = x86.dmk.vmovups[%278 + 1120], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %290 = x86.dmk.vmovups[%278 + 1400], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %291 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb6(%276 : !x86.reg64<rdi>, %277 : !x86.reg64<rsi>, %278 : !x86.reg64<rdx>, %279 : !x86.reg64<rbp>, %280 : !x86.reg64<rsp>, %281 : !x86.reg64<r11>, %284 : !x86.reg64<r10>, %283 : !x86.avx512maskreg<k1>, %285 : !x86.avx512reg<zmm26>, %286 : !x86.avx512reg<zmm27>, %287 : !x86.avx512reg<zmm28>, %288 : !x86.avx512reg<zmm29>, %289 : !x86.avx512reg<zmm30>, %290 : !x86.avx512reg<zmm31>, %291 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb6(%292: !x86.reg64<rdi>, %293: !x86.reg64<rsi>, %294: !x86.reg64<rdx>, %295: !x86.reg64<rbp>, %296: !x86.reg64<rsp>, %297: !x86.reg64<r11>, %298: !x86.reg64<r10>, %299: !x86.avx512maskreg<k1>, %300: !x86.avx512reg<zmm26>, %301: !x86.avx512reg<zmm27>, %302: !x86.avx512reg<zmm28>, %303: !x86.avx512reg<zmm29>, %304: !x86.avx512reg<zmm30>, %305: !x86.avx512reg<zmm31>, %306: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l37"
// CHECK-IR-LIBXSMM-NEXT:      %307 = x86.ri.add %306, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %308 = x86.dmk.vmovups[%292], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %309 = x86.dm.vbroadcastss [%293] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %310 = x86.rss.vfmadd231ps %300, %308, %309 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %311 = x86.dm.vbroadcastss [%293 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %312 = x86.rss.vfmadd231ps %301, %308, %311 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %313 = x86.dm.vbroadcastss [%293 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.rss.vfmadd231ps %302, %308, %313 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.dm.vbroadcastss [%293 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.rss.vfmadd231ps %303, %308, %315 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.dm.vbroadcastss [%293 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %318 = x86.rss.vfmadd231ps %304, %308, %317 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %319 = x86.dm.vbroadcastss [%293 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %320 = x86.ri.add %293, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %321 = x86.ri.add %292, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %322 = x86.rss.vfmadd231ps %305, %308, %319 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %323 = x86.dmk.vmovups[%321], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %324 = x86.dm.vbroadcastss [%320] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %325 = x86.rss.vfmadd231ps %310, %323, %324 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %326 = x86.dm.vbroadcastss [%320 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %327 = x86.rss.vfmadd231ps %312, %323, %326 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %328 = x86.dm.vbroadcastss [%320 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %329 = x86.rss.vfmadd231ps %314, %323, %328 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %330 = x86.dm.vbroadcastss [%320 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %331 = x86.rss.vfmadd231ps %316, %323, %330 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %332 = x86.dm.vbroadcastss [%320 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %333 = x86.rss.vfmadd231ps %318, %323, %332 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %334 = x86.dm.vbroadcastss [%320 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %335 = x86.ri.add %320, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %336 = x86.ri.add %321, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %337 = x86.rss.vfmadd231ps %322, %323, %334 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %338 = x86.dmk.vmovups[%336], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %339 = x86.dm.vbroadcastss [%335] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %340 = x86.rss.vfmadd231ps %325, %338, %339 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %341 = x86.dm.vbroadcastss [%335 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %342 = x86.rss.vfmadd231ps %327, %338, %341 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %343 = x86.dm.vbroadcastss [%335 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %344 = x86.rss.vfmadd231ps %329, %338, %343 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %345 = x86.dm.vbroadcastss [%335 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %346 = x86.rss.vfmadd231ps %331, %338, %345 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %347 = x86.dm.vbroadcastss [%335 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %348 = x86.rss.vfmadd231ps %333, %338, %347 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %349 = x86.dm.vbroadcastss [%335 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %350 = x86.ri.add %335, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %351 = x86.ri.add %336, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %352 = x86.rss.vfmadd231ps %337, %338, %349 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %353 = x86.dmk.vmovups[%351], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %354 = x86.dm.vbroadcastss [%350] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %355 = x86.rss.vfmadd231ps %340, %353, %354 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %356 = x86.dm.vbroadcastss [%350 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %357 = x86.rss.vfmadd231ps %342, %353, %356 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %358 = x86.dm.vbroadcastss [%350 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %359 = x86.rss.vfmadd231ps %344, %353, %358 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %360 = x86.dm.vbroadcastss [%350 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %361 = x86.rss.vfmadd231ps %346, %353, %360 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %362 = x86.dm.vbroadcastss [%350 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %363 = x86.rss.vfmadd231ps %348, %353, %362 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %364 = x86.dm.vbroadcastss [%350 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %365 = x86.ri.add %350, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %366 = x86.ri.add %351, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.rss.vfmadd231ps %352, %353, %364 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %368 = x86.si.cmp %307, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %368 : !x86.rflags<rflags>, ^bb6(%366 : !x86.reg64<rdi>, %365 : !x86.reg64<rsi>, %294 : !x86.reg64<rdx>, %295 : !x86.reg64<rbp>, %296 : !x86.reg64<rsp>, %297 : !x86.reg64<r11>, %298 : !x86.reg64<r10>, %299 : !x86.avx512maskreg<k1>, %355 : !x86.avx512reg<zmm26>, %357 : !x86.avx512reg<zmm27>, %359 : !x86.avx512reg<zmm28>, %361 : !x86.avx512reg<zmm29>, %363 : !x86.avx512reg<zmm30>, %367 : !x86.avx512reg<zmm31>, %307 : !x86.reg64<r12>), ^bb7(%366 : !x86.reg64<rdi>, %365 : !x86.reg64<rsi>, %294 : !x86.reg64<rdx>, %295 : !x86.reg64<rbp>, %296 : !x86.reg64<rsp>, %297 : !x86.reg64<r11>, %298 : !x86.reg64<r10>, %299 : !x86.avx512maskreg<k1>, %355 : !x86.avx512reg<zmm26>, %357 : !x86.avx512reg<zmm27>, %359 : !x86.avx512reg<zmm28>, %361 : !x86.avx512reg<zmm29>, %363 : !x86.avx512reg<zmm30>, %367 : !x86.avx512reg<zmm31>, %307 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb7(%369: !x86.reg64<rdi>, %370: !x86.reg64<rsi>, %371: !x86.reg64<rdx>, %372: !x86.reg64<rbp>, %373: !x86.reg64<rsp>, %374: !x86.reg64<r11>, %375: !x86.reg64<r10>, %376: !x86.avx512maskreg<k1>, %377: !x86.avx512reg<zmm26>, %378: !x86.avx512reg<zmm27>, %379: !x86.avx512reg<zmm28>, %380: !x86.avx512reg<zmm29>, %381: !x86.avx512reg<zmm30>, %382: !x86.avx512reg<zmm31>, %383: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %384 = x86.ri.sub %370, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%371], %377, %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%371 + 280], %378, %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%371 + 560], %379, %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%371 + 840], %380, %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%371 + 1120], %381, %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%371 + 1400], %382, %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %385 = x86.ri.add %371, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %386 = x86.ri.sub %369, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %387 = x86.si.cmp %375, 70 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %387 : !x86.rflags<rflags>, ^bb5(%386 : !x86.reg64<rdi>, %384 : !x86.reg64<rsi>, %385 : !x86.reg64<rdx>, %372 : !x86.reg64<rbp>, %373 : !x86.reg64<rsp>, %374 : !x86.reg64<r11>, %375 : !x86.reg64<r10>, %376 : !x86.avx512maskreg<k1>), ^bb8(%386 : !x86.reg64<rdi>, %384 : !x86.reg64<rsi>, %385 : !x86.reg64<rdx>, %372 : !x86.reg64<rbp>, %373 : !x86.reg64<rsp>, %374 : !x86.reg64<r11>, %375 : !x86.reg64<r10>, %376 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb8(%388: !x86.reg64<rdi>, %389: !x86.reg64<rsi>, %390: !x86.reg64<rdx>, %391: !x86.reg64<rbp>, %392: !x86.reg64<rsp>, %393: !x86.reg64<r11>, %394: !x86.reg64<r10>, %395: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      %396 = x86.ri.add %390, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %397 = x86.ri.add %389, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %398 = x86.ri.sub %388, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %399 = x86.si.cmp %393, 18 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %399 : !x86.rflags<rflags>, ^bb0(%398 : !x86.reg64<rdi>, %397 : !x86.reg64<rsi>, %396 : !x86.reg64<rdx>, %391 : !x86.reg64<rbp>, %392 : !x86.reg64<rsp>, %393 : !x86.reg64<r11>), ^bb9(%398 : !x86.reg64<rdi>, %397 : !x86.reg64<rsi>, %396 : !x86.reg64<rdx>, %391 : !x86.reg64<rbp>, %392 : !x86.reg64<rsp>, %393 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb9(%400: !x86.reg64<rdi>, %401: !x86.reg64<rsi>, %402: !x86.reg64<rdx>, %403: !x86.reg64<rbp>, %404: !x86.reg64<rsp>, %405: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %406 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb10(%400 : !x86.reg64<rdi>, %401 : !x86.reg64<rsi>, %402 : !x86.reg64<rdx>, %403 : !x86.reg64<rbp>, %404 : !x86.reg64<rsp>, %406 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb10(%407: !x86.reg64<rdi>, %408: !x86.reg64<rsi>, %409: !x86.reg64<rdx>, %410: !x86.reg64<rbp>, %411: !x86.reg64<rsp>, %412: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l38"
// CHECK-IR-LIBXSMM-NEXT:      %413 = x86.ri.add %412, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %414 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb11(%407 : !x86.reg64<rdi>, %408 : !x86.reg64<rsi>, %409 : !x86.reg64<rdx>, %410 : !x86.reg64<rbp>, %411 : !x86.reg64<rsp>, %413 : !x86.reg64<r11>, %414 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb11(%415: !x86.reg64<rdi>, %416: !x86.reg64<rsi>, %417: !x86.reg64<rdx>, %418: !x86.reg64<rbp>, %419: !x86.reg64<rsp>, %420: !x86.reg64<r11>, %421: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l39"
// CHECK-IR-LIBXSMM-NEXT:      %422 = x86.ri.add %421, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %423 = x86.dm.vmovups [%417] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %424 = x86.dm.vmovups [%417 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %425 = x86.dm.vmovups [%417 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %426 = x86.dm.vmovups [%417 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %427 = x86.dm.vmovups [%417 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %428 = x86.dm.vmovups [%417 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %429 = x86.dm.vmovups [%417 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %430 = x86.dm.vmovups [%417 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %431 = x86.dm.vmovups [%417 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %432 = x86.dm.vmovups [%417 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %433 = x86.dm.vmovups [%417 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %434 = x86.dm.vmovups [%417 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %435 = x86.dm.vmovups [%417 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %436 = x86.dm.vmovups [%417 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %437 = x86.dm.vmovups [%417 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %438 = x86.dm.vmovups [%417 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %439 = x86.dm.vmovups [%417 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %440 = x86.dm.vmovups [%417 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %441 = x86.dm.vmovups [%417 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %442 = x86.dm.vmovups [%417 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %443 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb12(%415 : !x86.reg64<rdi>, %416 : !x86.reg64<rsi>, %417 : !x86.reg64<rdx>, %418 : !x86.reg64<rbp>, %419 : !x86.reg64<rsp>, %420 : !x86.reg64<r11>, %422 : !x86.reg64<r10>, %423 : !x86.avx512reg<zmm12>, %424 : !x86.avx512reg<zmm13>, %425 : !x86.avx512reg<zmm14>, %426 : !x86.avx512reg<zmm15>, %427 : !x86.avx512reg<zmm16>, %428 : !x86.avx512reg<zmm17>, %429 : !x86.avx512reg<zmm18>, %430 : !x86.avx512reg<zmm19>, %431 : !x86.avx512reg<zmm20>, %432 : !x86.avx512reg<zmm21>, %433 : !x86.avx512reg<zmm22>, %434 : !x86.avx512reg<zmm23>, %435 : !x86.avx512reg<zmm24>, %436 : !x86.avx512reg<zmm25>, %437 : !x86.avx512reg<zmm26>, %438 : !x86.avx512reg<zmm27>, %439 : !x86.avx512reg<zmm28>, %440 : !x86.avx512reg<zmm29>, %441 : !x86.avx512reg<zmm30>, %442 : !x86.avx512reg<zmm31>, %443 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb12(%444: !x86.reg64<rdi>, %445: !x86.reg64<rsi>, %446: !x86.reg64<rdx>, %447: !x86.reg64<rbp>, %448: !x86.reg64<rsp>, %449: !x86.reg64<r11>, %450: !x86.reg64<r10>, %451: !x86.avx512reg<zmm12>, %452: !x86.avx512reg<zmm13>, %453: !x86.avx512reg<zmm14>, %454: !x86.avx512reg<zmm15>, %455: !x86.avx512reg<zmm16>, %456: !x86.avx512reg<zmm17>, %457: !x86.avx512reg<zmm18>, %458: !x86.avx512reg<zmm19>, %459: !x86.avx512reg<zmm20>, %460: !x86.avx512reg<zmm21>, %461: !x86.avx512reg<zmm22>, %462: !x86.avx512reg<zmm23>, %463: !x86.avx512reg<zmm24>, %464: !x86.avx512reg<zmm25>, %465: !x86.avx512reg<zmm26>, %466: !x86.avx512reg<zmm27>, %467: !x86.avx512reg<zmm28>, %468: !x86.avx512reg<zmm29>, %469: !x86.avx512reg<zmm30>, %470: !x86.avx512reg<zmm31>, %471: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l40"
// CHECK-IR-LIBXSMM-NEXT:      %472 = x86.ri.add %471, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %473 = x86.dm.vmovups [%444] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %474 = x86.dm.vmovups [%444 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %475 = x86.dm.vmovups [%444 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %476 = x86.dm.vmovups [%444 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %477 = x86.dm.vbroadcastss [%445] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %478 = x86.rss.vfmadd231ps %451, %473, %477 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %479 = x86.rss.vfmadd231ps %452, %474, %477 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %480 = x86.rss.vfmadd231ps %453, %475, %477 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %481 = x86.rss.vfmadd231ps %454, %476, %477 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %482 = x86.dm.vbroadcastss [%445 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %483 = x86.rss.vfmadd231ps %455, %473, %482 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %484 = x86.rss.vfmadd231ps %456, %474, %482 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %485 = x86.rss.vfmadd231ps %457, %475, %482 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %486 = x86.rss.vfmadd231ps %458, %476, %482 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %487 = x86.dm.vbroadcastss [%445 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %488 = x86.rss.vfmadd231ps %459, %473, %487 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %489 = x86.rss.vfmadd231ps %460, %474, %487 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %490 = x86.rss.vfmadd231ps %461, %475, %487 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %491 = x86.rss.vfmadd231ps %462, %476, %487 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %492 = x86.dm.vbroadcastss [%445 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %493 = x86.rss.vfmadd231ps %463, %473, %492 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %494 = x86.rss.vfmadd231ps %464, %474, %492 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %495 = x86.rss.vfmadd231ps %465, %475, %492 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %496 = x86.rss.vfmadd231ps %466, %476, %492 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %497 = x86.dm.vbroadcastss [%445 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %498 = x86.ri.add %445, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %499 = x86.ri.add %444, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %500 = x86.rss.vfmadd231ps %467, %473, %497 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %501 = x86.rss.vfmadd231ps %468, %474, %497 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %502 = x86.rss.vfmadd231ps %469, %475, %497 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %503 = x86.rss.vfmadd231ps %470, %476, %497 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %504 = x86.dm.vmovups [%499] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %505 = x86.dm.vmovups [%499 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %506 = x86.dm.vmovups [%499 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %507 = x86.dm.vmovups [%499 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %508 = x86.dm.vbroadcastss [%498] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %509 = x86.rss.vfmadd231ps %478, %504, %508 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %510 = x86.rss.vfmadd231ps %479, %505, %508 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %511 = x86.rss.vfmadd231ps %480, %506, %508 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %512 = x86.rss.vfmadd231ps %481, %507, %508 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %513 = x86.dm.vbroadcastss [%498 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %514 = x86.rss.vfmadd231ps %483, %504, %513 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %515 = x86.rss.vfmadd231ps %484, %505, %513 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %516 = x86.rss.vfmadd231ps %485, %506, %513 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %517 = x86.rss.vfmadd231ps %486, %507, %513 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %518 = x86.dm.vbroadcastss [%498 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %519 = x86.rss.vfmadd231ps %488, %504, %518 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %520 = x86.rss.vfmadd231ps %489, %505, %518 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %521 = x86.rss.vfmadd231ps %490, %506, %518 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %522 = x86.rss.vfmadd231ps %491, %507, %518 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %523 = x86.dm.vbroadcastss [%498 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %524 = x86.rss.vfmadd231ps %493, %504, %523 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %525 = x86.rss.vfmadd231ps %494, %505, %523 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %526 = x86.rss.vfmadd231ps %495, %506, %523 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %527 = x86.rss.vfmadd231ps %496, %507, %523 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %528 = x86.dm.vbroadcastss [%498 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %529 = x86.ri.add %498, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %530 = x86.ri.add %499, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %531 = x86.rss.vfmadd231ps %500, %504, %528 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %532 = x86.rss.vfmadd231ps %501, %505, %528 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %533 = x86.rss.vfmadd231ps %502, %506, %528 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %534 = x86.rss.vfmadd231ps %503, %507, %528 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %535 = x86.dm.vmovups [%530] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %536 = x86.dm.vmovups [%530 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %537 = x86.dm.vmovups [%530 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %538 = x86.dm.vmovups [%530 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %539 = x86.dm.vbroadcastss [%529] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %540 = x86.rss.vfmadd231ps %509, %535, %539 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %541 = x86.rss.vfmadd231ps %510, %536, %539 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %542 = x86.rss.vfmadd231ps %511, %537, %539 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %543 = x86.rss.vfmadd231ps %512, %538, %539 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %544 = x86.dm.vbroadcastss [%529 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %545 = x86.rss.vfmadd231ps %514, %535, %544 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %546 = x86.rss.vfmadd231ps %515, %536, %544 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %547 = x86.rss.vfmadd231ps %516, %537, %544 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %548 = x86.rss.vfmadd231ps %517, %538, %544 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %549 = x86.dm.vbroadcastss [%529 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %550 = x86.rss.vfmadd231ps %519, %535, %549 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %551 = x86.rss.vfmadd231ps %520, %536, %549 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %552 = x86.rss.vfmadd231ps %521, %537, %549 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %553 = x86.rss.vfmadd231ps %522, %538, %549 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %554 = x86.dm.vbroadcastss [%529 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %555 = x86.rss.vfmadd231ps %524, %535, %554 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %556 = x86.rss.vfmadd231ps %525, %536, %554 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %557 = x86.rss.vfmadd231ps %526, %537, %554 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %558 = x86.rss.vfmadd231ps %527, %538, %554 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %559 = x86.dm.vbroadcastss [%529 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %560 = x86.ri.add %529, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %561 = x86.ri.add %530, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %562 = x86.rss.vfmadd231ps %531, %535, %559 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %563 = x86.rss.vfmadd231ps %532, %536, %559 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %564 = x86.rss.vfmadd231ps %533, %537, %559 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %565 = x86.rss.vfmadd231ps %534, %538, %559 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %566 = x86.dm.vmovups [%561] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %567 = x86.dm.vmovups [%561 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %568 = x86.dm.vmovups [%561 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %569 = x86.dm.vmovups [%561 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %570 = x86.dm.vbroadcastss [%560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %571 = x86.rss.vfmadd231ps %540, %566, %570 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %572 = x86.rss.vfmadd231ps %541, %567, %570 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %573 = x86.rss.vfmadd231ps %542, %568, %570 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %574 = x86.rss.vfmadd231ps %543, %569, %570 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %575 = x86.dm.vbroadcastss [%560 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %576 = x86.rss.vfmadd231ps %545, %566, %575 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %577 = x86.rss.vfmadd231ps %546, %567, %575 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %578 = x86.rss.vfmadd231ps %547, %568, %575 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %579 = x86.rss.vfmadd231ps %548, %569, %575 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %580 = x86.dm.vbroadcastss [%560 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %581 = x86.rss.vfmadd231ps %550, %566, %580 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %582 = x86.rss.vfmadd231ps %551, %567, %580 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %583 = x86.rss.vfmadd231ps %552, %568, %580 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %584 = x86.rss.vfmadd231ps %553, %569, %580 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %585 = x86.dm.vbroadcastss [%560 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %586 = x86.rss.vfmadd231ps %555, %566, %585 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %587 = x86.rss.vfmadd231ps %556, %567, %585 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %588 = x86.rss.vfmadd231ps %557, %568, %585 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %589 = x86.rss.vfmadd231ps %558, %569, %585 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %590 = x86.dm.vbroadcastss [%560 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %591 = x86.ri.add %560, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %592 = x86.ri.add %561, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %593 = x86.rss.vfmadd231ps %562, %566, %590 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %594 = x86.rss.vfmadd231ps %563, %567, %590 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %595 = x86.rss.vfmadd231ps %564, %568, %590 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %596 = x86.rss.vfmadd231ps %565, %569, %590 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %597 = x86.si.cmp %472, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %597 : !x86.rflags<rflags>, ^bb12(%592 : !x86.reg64<rdi>, %591 : !x86.reg64<rsi>, %446 : !x86.reg64<rdx>, %447 : !x86.reg64<rbp>, %448 : !x86.reg64<rsp>, %449 : !x86.reg64<r11>, %450 : !x86.reg64<r10>, %571 : !x86.avx512reg<zmm12>, %572 : !x86.avx512reg<zmm13>, %573 : !x86.avx512reg<zmm14>, %574 : !x86.avx512reg<zmm15>, %576 : !x86.avx512reg<zmm16>, %577 : !x86.avx512reg<zmm17>, %578 : !x86.avx512reg<zmm18>, %579 : !x86.avx512reg<zmm19>, %581 : !x86.avx512reg<zmm20>, %582 : !x86.avx512reg<zmm21>, %583 : !x86.avx512reg<zmm22>, %584 : !x86.avx512reg<zmm23>, %586 : !x86.avx512reg<zmm24>, %587 : !x86.avx512reg<zmm25>, %588 : !x86.avx512reg<zmm26>, %589 : !x86.avx512reg<zmm27>, %593 : !x86.avx512reg<zmm28>, %594 : !x86.avx512reg<zmm29>, %595 : !x86.avx512reg<zmm30>, %596 : !x86.avx512reg<zmm31>, %472 : !x86.reg64<r12>), ^bb13(%592 : !x86.reg64<rdi>, %591 : !x86.reg64<rsi>, %446 : !x86.reg64<rdx>, %447 : !x86.reg64<rbp>, %448 : !x86.reg64<rsp>, %449 : !x86.reg64<r11>, %450 : !x86.reg64<r10>, %571 : !x86.avx512reg<zmm12>, %572 : !x86.avx512reg<zmm13>, %573 : !x86.avx512reg<zmm14>, %574 : !x86.avx512reg<zmm15>, %576 : !x86.avx512reg<zmm16>, %577 : !x86.avx512reg<zmm17>, %578 : !x86.avx512reg<zmm18>, %579 : !x86.avx512reg<zmm19>, %581 : !x86.avx512reg<zmm20>, %582 : !x86.avx512reg<zmm21>, %583 : !x86.avx512reg<zmm22>, %584 : !x86.avx512reg<zmm23>, %586 : !x86.avx512reg<zmm24>, %587 : !x86.avx512reg<zmm25>, %588 : !x86.avx512reg<zmm26>, %589 : !x86.avx512reg<zmm27>, %593 : !x86.avx512reg<zmm28>, %594 : !x86.avx512reg<zmm29>, %595 : !x86.avx512reg<zmm30>, %596 : !x86.avx512reg<zmm31>, %472 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb13(%598: !x86.reg64<rdi>, %599: !x86.reg64<rsi>, %600: !x86.reg64<rdx>, %601: !x86.reg64<rbp>, %602: !x86.reg64<rsp>, %603: !x86.reg64<r11>, %604: !x86.reg64<r10>, %605: !x86.avx512reg<zmm12>, %606: !x86.avx512reg<zmm13>, %607: !x86.avx512reg<zmm14>, %608: !x86.avx512reg<zmm15>, %609: !x86.avx512reg<zmm16>, %610: !x86.avx512reg<zmm17>, %611: !x86.avx512reg<zmm18>, %612: !x86.avx512reg<zmm19>, %613: !x86.avx512reg<zmm20>, %614: !x86.avx512reg<zmm21>, %615: !x86.avx512reg<zmm22>, %616: !x86.avx512reg<zmm23>, %617: !x86.avx512reg<zmm24>, %618: !x86.avx512reg<zmm25>, %619: !x86.avx512reg<zmm26>, %620: !x86.avx512reg<zmm27>, %621: !x86.avx512reg<zmm28>, %622: !x86.avx512reg<zmm29>, %623: !x86.avx512reg<zmm30>, %624: !x86.avx512reg<zmm31>, %625: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %626 = x86.ri.sub %599, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600], %605 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 64], %606 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 128], %607 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 192], %608 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 280], %609 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 344], %610 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 408], %611 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 472], %612 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 560], %613 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 624], %614 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 688], %615 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 752], %616 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 840], %617 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 904], %618 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 968], %619 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 1032], %620 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 1120], %621 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 1184], %622 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 1248], %623 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%600 + 1312], %624 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %627 = x86.ri.add %600, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %628 = x86.ri.sub %598, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %629 = x86.si.cmp %604, 64 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %629 : !x86.rflags<rflags>, ^bb11(%628 : !x86.reg64<rdi>, %626 : !x86.reg64<rsi>, %627 : !x86.reg64<rdx>, %601 : !x86.reg64<rbp>, %602 : !x86.reg64<rsp>, %603 : !x86.reg64<r11>, %604 : !x86.reg64<r10>), ^bb14(%628 : !x86.reg64<rdi>, %626 : !x86.reg64<rsi>, %627 : !x86.reg64<rdx>, %601 : !x86.reg64<rbp>, %602 : !x86.reg64<rsp>, %603 : !x86.reg64<r11>, %604 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb14(%630: !x86.reg64<rdi>, %631: !x86.reg64<rsi>, %632: !x86.reg64<rdx>, %633: !x86.reg64<rbp>, %634: !x86.reg64<rsp>, %635: !x86.reg64<r11>, %636: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %637 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %638 = x86.ks.kmovw %637 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %639 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb15(%630 : !x86.reg64<rdi>, %631 : !x86.reg64<rsi>, %632 : !x86.reg64<rdx>, %633 : !x86.reg64<rbp>, %634 : !x86.reg64<rsp>, %635 : !x86.reg64<r11>, %639 : !x86.reg64<r10>, %638 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb15(%640: !x86.reg64<rdi>, %641: !x86.reg64<rsi>, %642: !x86.reg64<rdx>, %643: !x86.reg64<rbp>, %644: !x86.reg64<rsp>, %645: !x86.reg64<r11>, %646: !x86.reg64<r10>, %647: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l41"
// CHECK-IR-LIBXSMM-NEXT:      %648 = x86.ri.add %646, 6 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %649 = x86.dmk.vmovups[%642], %647 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %650 = x86.dmk.vmovups[%642 + 280], %647 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %651 = x86.dmk.vmovups[%642 + 560], %647 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %652 = x86.dmk.vmovups[%642 + 840], %647 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %653 = x86.dmk.vmovups[%642 + 1120], %647 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %654 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb16(%640 : !x86.reg64<rdi>, %641 : !x86.reg64<rsi>, %642 : !x86.reg64<rdx>, %643 : !x86.reg64<rbp>, %644 : !x86.reg64<rsp>, %645 : !x86.reg64<r11>, %648 : !x86.reg64<r10>, %647 : !x86.avx512maskreg<k1>, %649 : !x86.avx512reg<zmm27>, %650 : !x86.avx512reg<zmm28>, %651 : !x86.avx512reg<zmm29>, %652 : !x86.avx512reg<zmm30>, %653 : !x86.avx512reg<zmm31>, %654 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb16(%655: !x86.reg64<rdi>, %656: !x86.reg64<rsi>, %657: !x86.reg64<rdx>, %658: !x86.reg64<rbp>, %659: !x86.reg64<rsp>, %660: !x86.reg64<r11>, %661: !x86.reg64<r10>, %662: !x86.avx512maskreg<k1>, %663: !x86.avx512reg<zmm27>, %664: !x86.avx512reg<zmm28>, %665: !x86.avx512reg<zmm29>, %666: !x86.avx512reg<zmm30>, %667: !x86.avx512reg<zmm31>, %668: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l42"
// CHECK-IR-LIBXSMM-NEXT:      %669 = x86.ri.add %668, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %670 = x86.dmk.vmovups[%655], %662 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %671 = x86.dm.vbroadcastss [%656] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %672 = x86.rss.vfmadd231ps %663, %670, %671 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %673 = x86.dm.vbroadcastss [%656 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %674 = x86.rss.vfmadd231ps %664, %670, %673 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %675 = x86.dm.vbroadcastss [%656 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %676 = x86.rss.vfmadd231ps %665, %670, %675 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %677 = x86.dm.vbroadcastss [%656 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %678 = x86.rss.vfmadd231ps %666, %670, %677 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %679 = x86.dm.vbroadcastss [%656 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %680 = x86.ri.add %656, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %681 = x86.ri.add %655, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %682 = x86.rss.vfmadd231ps %667, %670, %679 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %683 = x86.dmk.vmovups[%681], %662 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %684 = x86.dm.vbroadcastss [%680] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %685 = x86.rss.vfmadd231ps %672, %683, %684 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %686 = x86.dm.vbroadcastss [%680 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %687 = x86.rss.vfmadd231ps %674, %683, %686 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %688 = x86.dm.vbroadcastss [%680 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %689 = x86.rss.vfmadd231ps %676, %683, %688 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %690 = x86.dm.vbroadcastss [%680 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %691 = x86.rss.vfmadd231ps %678, %683, %690 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %692 = x86.dm.vbroadcastss [%680 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %693 = x86.ri.add %680, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %694 = x86.ri.add %681, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %695 = x86.rss.vfmadd231ps %682, %683, %692 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %696 = x86.dmk.vmovups[%694], %662 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %697 = x86.dm.vbroadcastss [%693] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %698 = x86.rss.vfmadd231ps %685, %696, %697 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %699 = x86.dm.vbroadcastss [%693 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %700 = x86.rss.vfmadd231ps %687, %696, %699 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %701 = x86.dm.vbroadcastss [%693 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %702 = x86.rss.vfmadd231ps %689, %696, %701 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %703 = x86.dm.vbroadcastss [%693 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %704 = x86.rss.vfmadd231ps %691, %696, %703 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %705 = x86.dm.vbroadcastss [%693 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %706 = x86.ri.add %693, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %707 = x86.ri.add %694, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %708 = x86.rss.vfmadd231ps %695, %696, %705 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %709 = x86.dmk.vmovups[%707], %662 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %710 = x86.dm.vbroadcastss [%706] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %711 = x86.rss.vfmadd231ps %698, %709, %710 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %712 = x86.dm.vbroadcastss [%706 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %713 = x86.rss.vfmadd231ps %700, %709, %712 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %714 = x86.dm.vbroadcastss [%706 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %715 = x86.rss.vfmadd231ps %702, %709, %714 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %716 = x86.dm.vbroadcastss [%706 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %717 = x86.rss.vfmadd231ps %704, %709, %716 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %718 = x86.dm.vbroadcastss [%706 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %719 = x86.ri.add %706, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %720 = x86.ri.add %707, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %721 = x86.rss.vfmadd231ps %708, %709, %718 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %722 = x86.si.cmp %669, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %722 : !x86.rflags<rflags>, ^bb16(%720 : !x86.reg64<rdi>, %719 : !x86.reg64<rsi>, %657 : !x86.reg64<rdx>, %658 : !x86.reg64<rbp>, %659 : !x86.reg64<rsp>, %660 : !x86.reg64<r11>, %661 : !x86.reg64<r10>, %662 : !x86.avx512maskreg<k1>, %711 : !x86.avx512reg<zmm27>, %713 : !x86.avx512reg<zmm28>, %715 : !x86.avx512reg<zmm29>, %717 : !x86.avx512reg<zmm30>, %721 : !x86.avx512reg<zmm31>, %669 : !x86.reg64<r12>), ^bb17(%720 : !x86.reg64<rdi>, %719 : !x86.reg64<rsi>, %657 : !x86.reg64<rdx>, %658 : !x86.reg64<rbp>, %659 : !x86.reg64<rsp>, %660 : !x86.reg64<r11>, %661 : !x86.reg64<r10>, %662 : !x86.avx512maskreg<k1>, %711 : !x86.avx512reg<zmm27>, %713 : !x86.avx512reg<zmm28>, %715 : !x86.avx512reg<zmm29>, %717 : !x86.avx512reg<zmm30>, %721 : !x86.avx512reg<zmm31>, %669 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb17(%723: !x86.reg64<rdi>, %724: !x86.reg64<rsi>, %725: !x86.reg64<rdx>, %726: !x86.reg64<rbp>, %727: !x86.reg64<rsp>, %728: !x86.reg64<r11>, %729: !x86.reg64<r10>, %730: !x86.avx512maskreg<k1>, %731: !x86.avx512reg<zmm27>, %732: !x86.avx512reg<zmm28>, %733: !x86.avx512reg<zmm29>, %734: !x86.avx512reg<zmm30>, %735: !x86.avx512reg<zmm31>, %736: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %737 = x86.ri.sub %724, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%725], %731, %730 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%725 + 280], %732, %730 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%725 + 560], %733, %730 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%725 + 840], %734, %730 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%725 + 1120], %735, %730 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %738 = x86.ri.add %725, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %739 = x86.ri.sub %723, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %740 = x86.si.cmp %729, 70 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %740 : !x86.rflags<rflags>, ^bb15(%739 : !x86.reg64<rdi>, %737 : !x86.reg64<rsi>, %738 : !x86.reg64<rdx>, %726 : !x86.reg64<rbp>, %727 : !x86.reg64<rsp>, %728 : !x86.reg64<r11>, %729 : !x86.reg64<r10>, %730 : !x86.avx512maskreg<k1>), ^bb18(%739 : !x86.reg64<rdi>, %737 : !x86.reg64<rsi>, %738 : !x86.reg64<rdx>, %726 : !x86.reg64<rbp>, %727 : !x86.reg64<rsp>, %728 : !x86.reg64<r11>, %729 : !x86.reg64<r10>, %730 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb18(%741: !x86.reg64<rdi>, %742: !x86.reg64<rsi>, %743: !x86.reg64<rdx>, %744: !x86.reg64<rbp>, %745: !x86.reg64<rsp>, %746: !x86.reg64<r11>, %747: !x86.reg64<r10>, %748: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      %749 = x86.ri.add %743, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %750 = x86.ri.add %742, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %751 = x86.ri.sub %741, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %752 = x86.si.cmp %746, 38 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %752 : !x86.rflags<rflags>, ^bb10(%751 : !x86.reg64<rdi>, %750 : !x86.reg64<rsi>, %749 : !x86.reg64<rdx>, %744 : !x86.reg64<rbp>, %745 : !x86.reg64<rsp>, %746 : !x86.reg64<r11>), ^bb19(%751 : !x86.reg64<rdi>, %750 : !x86.reg64<rsi>, %749 : !x86.reg64<rdx>, %744 : !x86.reg64<rbp>, %745 : !x86.reg64<rsp>, %746 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb19(%753: !x86.reg64<rdi>, %754: !x86.reg64<rsi>, %755: !x86.reg64<rdx>, %756: !x86.reg64<rbp>, %757: !x86.reg64<rsp>, %758: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %759 = x86.ds.mov %756 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %760, %761 = x86.d.pop %759 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }
