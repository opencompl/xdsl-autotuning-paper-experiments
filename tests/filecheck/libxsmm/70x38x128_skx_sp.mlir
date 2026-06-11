// RUN: libxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM

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
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb5(%266 : !x86.reg64<rdi>, %267 : !x86.reg64<rsi>, %268 : !x86.reg64<rdx>, %269 : !x86.reg64<rbp>, %270 : !x86.reg64<rsp>, %271 : !x86.reg64<r11>, %275 : !x86.reg64<r10>, %273 : !x86.reg64<r15>, %274 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb5(%276: !x86.reg64<rdi>, %277: !x86.reg64<rsi>, %278: !x86.reg64<rdx>, %279: !x86.reg64<rbp>, %280: !x86.reg64<rsp>, %281: !x86.reg64<r11>, %282: !x86.reg64<r10>, %283: !x86.reg64<r15>, %284: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l36"
// CHECK-IR-LIBXSMM-NEXT:      %285 = x86.ri.add %282, 6 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %286 = x86.dmk.vmovups[%278], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %287 = x86.dmk.vmovups[%278 + 280], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %288 = x86.dmk.vmovups[%278 + 560], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %289 = x86.dmk.vmovups[%278 + 840], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %290 = x86.dmk.vmovups[%278 + 1120], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %291 = x86.dmk.vmovups[%278 + 1400], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %292 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb6(%276 : !x86.reg64<rdi>, %277 : !x86.reg64<rsi>, %278 : !x86.reg64<rdx>, %279 : !x86.reg64<rbp>, %280 : !x86.reg64<rsp>, %281 : !x86.reg64<r11>, %285 : !x86.reg64<r10>, %283 : !x86.reg64<r15>, %284 : !x86.avx512maskreg<k1>, %286 : !x86.avx512reg<zmm26>, %287 : !x86.avx512reg<zmm27>, %288 : !x86.avx512reg<zmm28>, %289 : !x86.avx512reg<zmm29>, %290 : !x86.avx512reg<zmm30>, %291 : !x86.avx512reg<zmm31>, %292 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb6(%293: !x86.reg64<rdi>, %294: !x86.reg64<rsi>, %295: !x86.reg64<rdx>, %296: !x86.reg64<rbp>, %297: !x86.reg64<rsp>, %298: !x86.reg64<r11>, %299: !x86.reg64<r10>, %300: !x86.reg64<r15>, %301: !x86.avx512maskreg<k1>, %302: !x86.avx512reg<zmm26>, %303: !x86.avx512reg<zmm27>, %304: !x86.avx512reg<zmm28>, %305: !x86.avx512reg<zmm29>, %306: !x86.avx512reg<zmm30>, %307: !x86.avx512reg<zmm31>, %308: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l37"
// CHECK-IR-LIBXSMM-NEXT:      %309 = x86.ri.add %308, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %310 = x86.dmk.vmovups[%293], %301 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %311 = x86.dm.vbroadcastss [%294] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %312 = x86.rss.vfmadd231ps %302, %310, %311 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %313 = x86.dm.vbroadcastss [%294 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.rss.vfmadd231ps %303, %310, %313 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.dm.vbroadcastss [%294 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.rss.vfmadd231ps %304, %310, %315 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.dm.vbroadcastss [%294 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %318 = x86.rss.vfmadd231ps %305, %310, %317 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %319 = x86.dm.vbroadcastss [%294 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %320 = x86.rss.vfmadd231ps %306, %310, %319 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %321 = x86.dm.vbroadcastss [%294 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %322 = x86.ri.add %294, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %323 = x86.ri.add %293, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %324 = x86.rss.vfmadd231ps %307, %310, %321 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %325 = x86.dmk.vmovups[%323], %301 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %326 = x86.dm.vbroadcastss [%322] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %327 = x86.rss.vfmadd231ps %312, %325, %326 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %328 = x86.dm.vbroadcastss [%322 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %329 = x86.rss.vfmadd231ps %314, %325, %328 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %330 = x86.dm.vbroadcastss [%322 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %331 = x86.rss.vfmadd231ps %316, %325, %330 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %332 = x86.dm.vbroadcastss [%322 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %333 = x86.rss.vfmadd231ps %318, %325, %332 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %334 = x86.dm.vbroadcastss [%322 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %335 = x86.rss.vfmadd231ps %320, %325, %334 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %336 = x86.dm.vbroadcastss [%322 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %337 = x86.ri.add %322, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %338 = x86.ri.add %323, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %339 = x86.rss.vfmadd231ps %324, %325, %336 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %340 = x86.dmk.vmovups[%338], %301 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %341 = x86.dm.vbroadcastss [%337] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %342 = x86.rss.vfmadd231ps %327, %340, %341 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %343 = x86.dm.vbroadcastss [%337 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %344 = x86.rss.vfmadd231ps %329, %340, %343 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %345 = x86.dm.vbroadcastss [%337 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %346 = x86.rss.vfmadd231ps %331, %340, %345 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %347 = x86.dm.vbroadcastss [%337 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %348 = x86.rss.vfmadd231ps %333, %340, %347 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %349 = x86.dm.vbroadcastss [%337 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %350 = x86.rss.vfmadd231ps %335, %340, %349 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %351 = x86.dm.vbroadcastss [%337 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %352 = x86.ri.add %337, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %353 = x86.ri.add %338, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %354 = x86.rss.vfmadd231ps %339, %340, %351 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %355 = x86.dmk.vmovups[%353], %301 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %356 = x86.dm.vbroadcastss [%352] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %357 = x86.rss.vfmadd231ps %342, %355, %356 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %358 = x86.dm.vbroadcastss [%352 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %359 = x86.rss.vfmadd231ps %344, %355, %358 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %360 = x86.dm.vbroadcastss [%352 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %361 = x86.rss.vfmadd231ps %346, %355, %360 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %362 = x86.dm.vbroadcastss [%352 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %363 = x86.rss.vfmadd231ps %348, %355, %362 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %364 = x86.dm.vbroadcastss [%352 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %365 = x86.rss.vfmadd231ps %350, %355, %364 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %366 = x86.dm.vbroadcastss [%352 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.ri.add %352, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %368 = x86.ri.add %353, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %369 = x86.rss.vfmadd231ps %354, %355, %366 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %370 = x86.si.cmp %309, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %370 : !x86.rflags<rflags>, ^bb6(%368 : !x86.reg64<rdi>, %367 : !x86.reg64<rsi>, %295 : !x86.reg64<rdx>, %296 : !x86.reg64<rbp>, %297 : !x86.reg64<rsp>, %298 : !x86.reg64<r11>, %299 : !x86.reg64<r10>, %300 : !x86.reg64<r15>, %301 : !x86.avx512maskreg<k1>, %357 : !x86.avx512reg<zmm26>, %359 : !x86.avx512reg<zmm27>, %361 : !x86.avx512reg<zmm28>, %363 : !x86.avx512reg<zmm29>, %365 : !x86.avx512reg<zmm30>, %369 : !x86.avx512reg<zmm31>, %309 : !x86.reg64<r12>), ^bb7(%368 : !x86.reg64<rdi>, %367 : !x86.reg64<rsi>, %295 : !x86.reg64<rdx>, %296 : !x86.reg64<rbp>, %297 : !x86.reg64<rsp>, %298 : !x86.reg64<r11>, %299 : !x86.reg64<r10>, %300 : !x86.reg64<r15>, %301 : !x86.avx512maskreg<k1>, %357 : !x86.avx512reg<zmm26>, %359 : !x86.avx512reg<zmm27>, %361 : !x86.avx512reg<zmm28>, %363 : !x86.avx512reg<zmm29>, %365 : !x86.avx512reg<zmm30>, %369 : !x86.avx512reg<zmm31>, %309 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb7(%371: !x86.reg64<rdi>, %372: !x86.reg64<rsi>, %373: !x86.reg64<rdx>, %374: !x86.reg64<rbp>, %375: !x86.reg64<rsp>, %376: !x86.reg64<r11>, %377: !x86.reg64<r10>, %378: !x86.reg64<r15>, %379: !x86.avx512maskreg<k1>, %380: !x86.avx512reg<zmm26>, %381: !x86.avx512reg<zmm27>, %382: !x86.avx512reg<zmm28>, %383: !x86.avx512reg<zmm29>, %384: !x86.avx512reg<zmm30>, %385: !x86.avx512reg<zmm31>, %386: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %387 = x86.ri.sub %372, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%373], %380, %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%373 + 280], %381, %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%373 + 560], %382, %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%373 + 840], %383, %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%373 + 1120], %384, %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%373 + 1400], %385, %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %388 = x86.ri.add %373, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %389 = x86.ri.sub %371, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %390 = x86.si.cmp %377, 70 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %390 : !x86.rflags<rflags>, ^bb5(%389 : !x86.reg64<rdi>, %387 : !x86.reg64<rsi>, %388 : !x86.reg64<rdx>, %374 : !x86.reg64<rbp>, %375 : !x86.reg64<rsp>, %376 : !x86.reg64<r11>, %377 : !x86.reg64<r10>, %378 : !x86.reg64<r15>, %379 : !x86.avx512maskreg<k1>), ^bb8(%389 : !x86.reg64<rdi>, %387 : !x86.reg64<rsi>, %388 : !x86.reg64<rdx>, %374 : !x86.reg64<rbp>, %375 : !x86.reg64<rsp>, %376 : !x86.reg64<r11>, %377 : !x86.reg64<r10>, %378 : !x86.reg64<r15>, %379 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb8(%391: !x86.reg64<rdi>, %392: !x86.reg64<rsi>, %393: !x86.reg64<rdx>, %394: !x86.reg64<rbp>, %395: !x86.reg64<rsp>, %396: !x86.reg64<r11>, %397: !x86.reg64<r10>, %398: !x86.reg64<r15>, %399: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      %400 = x86.ri.add %393, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %401 = x86.ri.add %392, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %402 = x86.ri.sub %391, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %403 = x86.si.cmp %396, 18 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %403 : !x86.rflags<rflags>, ^bb0(%402 : !x86.reg64<rdi>, %401 : !x86.reg64<rsi>, %400 : !x86.reg64<rdx>, %394 : !x86.reg64<rbp>, %395 : !x86.reg64<rsp>, %396 : !x86.reg64<r11>), ^bb9(%402 : !x86.reg64<rdi>, %401 : !x86.reg64<rsi>, %400 : !x86.reg64<rdx>, %394 : !x86.reg64<rbp>, %395 : !x86.reg64<rsp>, %396 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb9(%404: !x86.reg64<rdi>, %405: !x86.reg64<rsi>, %406: !x86.reg64<rdx>, %407: !x86.reg64<rbp>, %408: !x86.reg64<rsp>, %409: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %410 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb10(%404 : !x86.reg64<rdi>, %405 : !x86.reg64<rsi>, %406 : !x86.reg64<rdx>, %407 : !x86.reg64<rbp>, %408 : !x86.reg64<rsp>, %410 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb10(%411: !x86.reg64<rdi>, %412: !x86.reg64<rsi>, %413: !x86.reg64<rdx>, %414: !x86.reg64<rbp>, %415: !x86.reg64<rsp>, %416: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l38"
// CHECK-IR-LIBXSMM-NEXT:      %417 = x86.ri.add %416, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %418 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb11(%411 : !x86.reg64<rdi>, %412 : !x86.reg64<rsi>, %413 : !x86.reg64<rdx>, %414 : !x86.reg64<rbp>, %415 : !x86.reg64<rsp>, %417 : !x86.reg64<r11>, %418 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb11(%419: !x86.reg64<rdi>, %420: !x86.reg64<rsi>, %421: !x86.reg64<rdx>, %422: !x86.reg64<rbp>, %423: !x86.reg64<rsp>, %424: !x86.reg64<r11>, %425: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l39"
// CHECK-IR-LIBXSMM-NEXT:      %426 = x86.ri.add %425, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %427 = x86.dm.vmovups [%421] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %428 = x86.dm.vmovups [%421 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %429 = x86.dm.vmovups [%421 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %430 = x86.dm.vmovups [%421 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %431 = x86.dm.vmovups [%421 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %432 = x86.dm.vmovups [%421 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %433 = x86.dm.vmovups [%421 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %434 = x86.dm.vmovups [%421 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %435 = x86.dm.vmovups [%421 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %436 = x86.dm.vmovups [%421 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %437 = x86.dm.vmovups [%421 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %438 = x86.dm.vmovups [%421 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %439 = x86.dm.vmovups [%421 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %440 = x86.dm.vmovups [%421 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %441 = x86.dm.vmovups [%421 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %442 = x86.dm.vmovups [%421 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %443 = x86.dm.vmovups [%421 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %444 = x86.dm.vmovups [%421 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %445 = x86.dm.vmovups [%421 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %446 = x86.dm.vmovups [%421 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %447 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb12(%419 : !x86.reg64<rdi>, %420 : !x86.reg64<rsi>, %421 : !x86.reg64<rdx>, %422 : !x86.reg64<rbp>, %423 : !x86.reg64<rsp>, %424 : !x86.reg64<r11>, %426 : !x86.reg64<r10>, %427 : !x86.avx512reg<zmm12>, %428 : !x86.avx512reg<zmm13>, %429 : !x86.avx512reg<zmm14>, %430 : !x86.avx512reg<zmm15>, %431 : !x86.avx512reg<zmm16>, %432 : !x86.avx512reg<zmm17>, %433 : !x86.avx512reg<zmm18>, %434 : !x86.avx512reg<zmm19>, %435 : !x86.avx512reg<zmm20>, %436 : !x86.avx512reg<zmm21>, %437 : !x86.avx512reg<zmm22>, %438 : !x86.avx512reg<zmm23>, %439 : !x86.avx512reg<zmm24>, %440 : !x86.avx512reg<zmm25>, %441 : !x86.avx512reg<zmm26>, %442 : !x86.avx512reg<zmm27>, %443 : !x86.avx512reg<zmm28>, %444 : !x86.avx512reg<zmm29>, %445 : !x86.avx512reg<zmm30>, %446 : !x86.avx512reg<zmm31>, %447 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb12(%448: !x86.reg64<rdi>, %449: !x86.reg64<rsi>, %450: !x86.reg64<rdx>, %451: !x86.reg64<rbp>, %452: !x86.reg64<rsp>, %453: !x86.reg64<r11>, %454: !x86.reg64<r10>, %455: !x86.avx512reg<zmm12>, %456: !x86.avx512reg<zmm13>, %457: !x86.avx512reg<zmm14>, %458: !x86.avx512reg<zmm15>, %459: !x86.avx512reg<zmm16>, %460: !x86.avx512reg<zmm17>, %461: !x86.avx512reg<zmm18>, %462: !x86.avx512reg<zmm19>, %463: !x86.avx512reg<zmm20>, %464: !x86.avx512reg<zmm21>, %465: !x86.avx512reg<zmm22>, %466: !x86.avx512reg<zmm23>, %467: !x86.avx512reg<zmm24>, %468: !x86.avx512reg<zmm25>, %469: !x86.avx512reg<zmm26>, %470: !x86.avx512reg<zmm27>, %471: !x86.avx512reg<zmm28>, %472: !x86.avx512reg<zmm29>, %473: !x86.avx512reg<zmm30>, %474: !x86.avx512reg<zmm31>, %475: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l40"
// CHECK-IR-LIBXSMM-NEXT:      %476 = x86.ri.add %475, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %477 = x86.dm.vmovups [%448] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %478 = x86.dm.vmovups [%448 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %479 = x86.dm.vmovups [%448 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %480 = x86.dm.vmovups [%448 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %481 = x86.dm.vbroadcastss [%449] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %482 = x86.rss.vfmadd231ps %455, %477, %481 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %483 = x86.rss.vfmadd231ps %456, %478, %481 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %484 = x86.rss.vfmadd231ps %457, %479, %481 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %485 = x86.rss.vfmadd231ps %458, %480, %481 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %486 = x86.dm.vbroadcastss [%449 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %487 = x86.rss.vfmadd231ps %459, %477, %486 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %488 = x86.rss.vfmadd231ps %460, %478, %486 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %489 = x86.rss.vfmadd231ps %461, %479, %486 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %490 = x86.rss.vfmadd231ps %462, %480, %486 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %491 = x86.dm.vbroadcastss [%449 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %492 = x86.rss.vfmadd231ps %463, %477, %491 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %493 = x86.rss.vfmadd231ps %464, %478, %491 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %494 = x86.rss.vfmadd231ps %465, %479, %491 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %495 = x86.rss.vfmadd231ps %466, %480, %491 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %496 = x86.dm.vbroadcastss [%449 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %497 = x86.rss.vfmadd231ps %467, %477, %496 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %498 = x86.rss.vfmadd231ps %468, %478, %496 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %499 = x86.rss.vfmadd231ps %469, %479, %496 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %500 = x86.rss.vfmadd231ps %470, %480, %496 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %501 = x86.dm.vbroadcastss [%449 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %502 = x86.ri.add %449, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %503 = x86.ri.add %448, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %504 = x86.rss.vfmadd231ps %471, %477, %501 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %505 = x86.rss.vfmadd231ps %472, %478, %501 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %506 = x86.rss.vfmadd231ps %473, %479, %501 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %507 = x86.rss.vfmadd231ps %474, %480, %501 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %508 = x86.dm.vmovups [%503] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %509 = x86.dm.vmovups [%503 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %510 = x86.dm.vmovups [%503 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %511 = x86.dm.vmovups [%503 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %512 = x86.dm.vbroadcastss [%502] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %513 = x86.rss.vfmadd231ps %482, %508, %512 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %514 = x86.rss.vfmadd231ps %483, %509, %512 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %515 = x86.rss.vfmadd231ps %484, %510, %512 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %516 = x86.rss.vfmadd231ps %485, %511, %512 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %517 = x86.dm.vbroadcastss [%502 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %518 = x86.rss.vfmadd231ps %487, %508, %517 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %519 = x86.rss.vfmadd231ps %488, %509, %517 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %520 = x86.rss.vfmadd231ps %489, %510, %517 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %521 = x86.rss.vfmadd231ps %490, %511, %517 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %522 = x86.dm.vbroadcastss [%502 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %523 = x86.rss.vfmadd231ps %492, %508, %522 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %524 = x86.rss.vfmadd231ps %493, %509, %522 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %525 = x86.rss.vfmadd231ps %494, %510, %522 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %526 = x86.rss.vfmadd231ps %495, %511, %522 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %527 = x86.dm.vbroadcastss [%502 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %528 = x86.rss.vfmadd231ps %497, %508, %527 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %529 = x86.rss.vfmadd231ps %498, %509, %527 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %530 = x86.rss.vfmadd231ps %499, %510, %527 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %531 = x86.rss.vfmadd231ps %500, %511, %527 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %532 = x86.dm.vbroadcastss [%502 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %533 = x86.ri.add %502, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %534 = x86.ri.add %503, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %535 = x86.rss.vfmadd231ps %504, %508, %532 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %536 = x86.rss.vfmadd231ps %505, %509, %532 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %537 = x86.rss.vfmadd231ps %506, %510, %532 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %538 = x86.rss.vfmadd231ps %507, %511, %532 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %539 = x86.dm.vmovups [%534] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %540 = x86.dm.vmovups [%534 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %541 = x86.dm.vmovups [%534 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %542 = x86.dm.vmovups [%534 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %543 = x86.dm.vbroadcastss [%533] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %544 = x86.rss.vfmadd231ps %513, %539, %543 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %545 = x86.rss.vfmadd231ps %514, %540, %543 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %546 = x86.rss.vfmadd231ps %515, %541, %543 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %547 = x86.rss.vfmadd231ps %516, %542, %543 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %548 = x86.dm.vbroadcastss [%533 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %549 = x86.rss.vfmadd231ps %518, %539, %548 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %550 = x86.rss.vfmadd231ps %519, %540, %548 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %551 = x86.rss.vfmadd231ps %520, %541, %548 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %552 = x86.rss.vfmadd231ps %521, %542, %548 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %553 = x86.dm.vbroadcastss [%533 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %554 = x86.rss.vfmadd231ps %523, %539, %553 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %555 = x86.rss.vfmadd231ps %524, %540, %553 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %556 = x86.rss.vfmadd231ps %525, %541, %553 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %557 = x86.rss.vfmadd231ps %526, %542, %553 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %558 = x86.dm.vbroadcastss [%533 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %559 = x86.rss.vfmadd231ps %528, %539, %558 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %560 = x86.rss.vfmadd231ps %529, %540, %558 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %561 = x86.rss.vfmadd231ps %530, %541, %558 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %562 = x86.rss.vfmadd231ps %531, %542, %558 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %563 = x86.dm.vbroadcastss [%533 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %564 = x86.ri.add %533, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %565 = x86.ri.add %534, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %566 = x86.rss.vfmadd231ps %535, %539, %563 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %567 = x86.rss.vfmadd231ps %536, %540, %563 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %568 = x86.rss.vfmadd231ps %537, %541, %563 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %569 = x86.rss.vfmadd231ps %538, %542, %563 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %570 = x86.dm.vmovups [%565] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %571 = x86.dm.vmovups [%565 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %572 = x86.dm.vmovups [%565 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-IR-LIBXSMM-NEXT:      %573 = x86.dm.vmovups [%565 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-IR-LIBXSMM-NEXT:      %574 = x86.dm.vbroadcastss [%564] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %575 = x86.rss.vfmadd231ps %544, %570, %574 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %576 = x86.rss.vfmadd231ps %545, %571, %574 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %577 = x86.rss.vfmadd231ps %546, %572, %574 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %578 = x86.rss.vfmadd231ps %547, %573, %574 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %579 = x86.dm.vbroadcastss [%564 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %580 = x86.rss.vfmadd231ps %549, %570, %579 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %581 = x86.rss.vfmadd231ps %550, %571, %579 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %582 = x86.rss.vfmadd231ps %551, %572, %579 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %583 = x86.rss.vfmadd231ps %552, %573, %579 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %584 = x86.dm.vbroadcastss [%564 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %585 = x86.rss.vfmadd231ps %554, %570, %584 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %586 = x86.rss.vfmadd231ps %555, %571, %584 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %587 = x86.rss.vfmadd231ps %556, %572, %584 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %588 = x86.rss.vfmadd231ps %557, %573, %584 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %589 = x86.dm.vbroadcastss [%564 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %590 = x86.rss.vfmadd231ps %559, %570, %589 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %591 = x86.rss.vfmadd231ps %560, %571, %589 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %592 = x86.rss.vfmadd231ps %561, %572, %589 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %593 = x86.rss.vfmadd231ps %562, %573, %589 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %594 = x86.dm.vbroadcastss [%564 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %595 = x86.ri.add %564, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %596 = x86.ri.add %565, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %597 = x86.rss.vfmadd231ps %566, %570, %594 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %598 = x86.rss.vfmadd231ps %567, %571, %594 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %599 = x86.rss.vfmadd231ps %568, %572, %594 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %600 = x86.rss.vfmadd231ps %569, %573, %594 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %601 = x86.si.cmp %476, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %601 : !x86.rflags<rflags>, ^bb12(%596 : !x86.reg64<rdi>, %595 : !x86.reg64<rsi>, %450 : !x86.reg64<rdx>, %451 : !x86.reg64<rbp>, %452 : !x86.reg64<rsp>, %453 : !x86.reg64<r11>, %454 : !x86.reg64<r10>, %575 : !x86.avx512reg<zmm12>, %576 : !x86.avx512reg<zmm13>, %577 : !x86.avx512reg<zmm14>, %578 : !x86.avx512reg<zmm15>, %580 : !x86.avx512reg<zmm16>, %581 : !x86.avx512reg<zmm17>, %582 : !x86.avx512reg<zmm18>, %583 : !x86.avx512reg<zmm19>, %585 : !x86.avx512reg<zmm20>, %586 : !x86.avx512reg<zmm21>, %587 : !x86.avx512reg<zmm22>, %588 : !x86.avx512reg<zmm23>, %590 : !x86.avx512reg<zmm24>, %591 : !x86.avx512reg<zmm25>, %592 : !x86.avx512reg<zmm26>, %593 : !x86.avx512reg<zmm27>, %597 : !x86.avx512reg<zmm28>, %598 : !x86.avx512reg<zmm29>, %599 : !x86.avx512reg<zmm30>, %600 : !x86.avx512reg<zmm31>, %476 : !x86.reg64<r12>), ^bb13(%596 : !x86.reg64<rdi>, %595 : !x86.reg64<rsi>, %450 : !x86.reg64<rdx>, %451 : !x86.reg64<rbp>, %452 : !x86.reg64<rsp>, %453 : !x86.reg64<r11>, %454 : !x86.reg64<r10>, %575 : !x86.avx512reg<zmm12>, %576 : !x86.avx512reg<zmm13>, %577 : !x86.avx512reg<zmm14>, %578 : !x86.avx512reg<zmm15>, %580 : !x86.avx512reg<zmm16>, %581 : !x86.avx512reg<zmm17>, %582 : !x86.avx512reg<zmm18>, %583 : !x86.avx512reg<zmm19>, %585 : !x86.avx512reg<zmm20>, %586 : !x86.avx512reg<zmm21>, %587 : !x86.avx512reg<zmm22>, %588 : !x86.avx512reg<zmm23>, %590 : !x86.avx512reg<zmm24>, %591 : !x86.avx512reg<zmm25>, %592 : !x86.avx512reg<zmm26>, %593 : !x86.avx512reg<zmm27>, %597 : !x86.avx512reg<zmm28>, %598 : !x86.avx512reg<zmm29>, %599 : !x86.avx512reg<zmm30>, %600 : !x86.avx512reg<zmm31>, %476 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb13(%602: !x86.reg64<rdi>, %603: !x86.reg64<rsi>, %604: !x86.reg64<rdx>, %605: !x86.reg64<rbp>, %606: !x86.reg64<rsp>, %607: !x86.reg64<r11>, %608: !x86.reg64<r10>, %609: !x86.avx512reg<zmm12>, %610: !x86.avx512reg<zmm13>, %611: !x86.avx512reg<zmm14>, %612: !x86.avx512reg<zmm15>, %613: !x86.avx512reg<zmm16>, %614: !x86.avx512reg<zmm17>, %615: !x86.avx512reg<zmm18>, %616: !x86.avx512reg<zmm19>, %617: !x86.avx512reg<zmm20>, %618: !x86.avx512reg<zmm21>, %619: !x86.avx512reg<zmm22>, %620: !x86.avx512reg<zmm23>, %621: !x86.avx512reg<zmm24>, %622: !x86.avx512reg<zmm25>, %623: !x86.avx512reg<zmm26>, %624: !x86.avx512reg<zmm27>, %625: !x86.avx512reg<zmm28>, %626: !x86.avx512reg<zmm29>, %627: !x86.avx512reg<zmm30>, %628: !x86.avx512reg<zmm31>, %629: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %630 = x86.ri.sub %603, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604], %609 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 64], %610 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 128], %611 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 192], %612 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 280], %613 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 344], %614 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 408], %615 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 472], %616 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 560], %617 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 624], %618 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 688], %619 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 752], %620 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 840], %621 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 904], %622 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 968], %623 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 1032], %624 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 1120], %625 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 1184], %626 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 1248], %627 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovups [%604 + 1312], %628 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %631 = x86.ri.add %604, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %632 = x86.ri.sub %602, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %633 = x86.si.cmp %608, 64 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %633 : !x86.rflags<rflags>, ^bb11(%632 : !x86.reg64<rdi>, %630 : !x86.reg64<rsi>, %631 : !x86.reg64<rdx>, %605 : !x86.reg64<rbp>, %606 : !x86.reg64<rsp>, %607 : !x86.reg64<r11>, %608 : !x86.reg64<r10>), ^bb14(%632 : !x86.reg64<rdi>, %630 : !x86.reg64<rsi>, %631 : !x86.reg64<rdx>, %605 : !x86.reg64<rbp>, %606 : !x86.reg64<rsp>, %607 : !x86.reg64<r11>, %608 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb14(%634: !x86.reg64<rdi>, %635: !x86.reg64<rsi>, %636: !x86.reg64<rdx>, %637: !x86.reg64<rbp>, %638: !x86.reg64<rsp>, %639: !x86.reg64<r11>, %640: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %641 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %642 = x86.ks.kmovw %641 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %643 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb15(%634 : !x86.reg64<rdi>, %635 : !x86.reg64<rsi>, %636 : !x86.reg64<rdx>, %637 : !x86.reg64<rbp>, %638 : !x86.reg64<rsp>, %639 : !x86.reg64<r11>, %643 : !x86.reg64<r10>, %641 : !x86.reg64<r15>, %642 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb15(%644: !x86.reg64<rdi>, %645: !x86.reg64<rsi>, %646: !x86.reg64<rdx>, %647: !x86.reg64<rbp>, %648: !x86.reg64<rsp>, %649: !x86.reg64<r11>, %650: !x86.reg64<r10>, %651: !x86.reg64<r15>, %652: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l41"
// CHECK-IR-LIBXSMM-NEXT:      %653 = x86.ri.add %650, 6 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %654 = x86.dmk.vmovups[%646], %652 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %655 = x86.dmk.vmovups[%646 + 280], %652 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %656 = x86.dmk.vmovups[%646 + 560], %652 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %657 = x86.dmk.vmovups[%646 + 840], %652 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %658 = x86.dmk.vmovups[%646 + 1120], %652 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %659 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb16(%644 : !x86.reg64<rdi>, %645 : !x86.reg64<rsi>, %646 : !x86.reg64<rdx>, %647 : !x86.reg64<rbp>, %648 : !x86.reg64<rsp>, %649 : !x86.reg64<r11>, %653 : !x86.reg64<r10>, %651 : !x86.reg64<r15>, %652 : !x86.avx512maskreg<k1>, %654 : !x86.avx512reg<zmm27>, %655 : !x86.avx512reg<zmm28>, %656 : !x86.avx512reg<zmm29>, %657 : !x86.avx512reg<zmm30>, %658 : !x86.avx512reg<zmm31>, %659 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb16(%660: !x86.reg64<rdi>, %661: !x86.reg64<rsi>, %662: !x86.reg64<rdx>, %663: !x86.reg64<rbp>, %664: !x86.reg64<rsp>, %665: !x86.reg64<r11>, %666: !x86.reg64<r10>, %667: !x86.reg64<r15>, %668: !x86.avx512maskreg<k1>, %669: !x86.avx512reg<zmm27>, %670: !x86.avx512reg<zmm28>, %671: !x86.avx512reg<zmm29>, %672: !x86.avx512reg<zmm30>, %673: !x86.avx512reg<zmm31>, %674: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l42"
// CHECK-IR-LIBXSMM-NEXT:      %675 = x86.ri.add %674, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-IR-LIBXSMM-NEXT:      %676 = x86.dmk.vmovups[%660], %668 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %677 = x86.dm.vbroadcastss [%661] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %678 = x86.rss.vfmadd231ps %669, %676, %677 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %679 = x86.dm.vbroadcastss [%661 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %680 = x86.rss.vfmadd231ps %670, %676, %679 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %681 = x86.dm.vbroadcastss [%661 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %682 = x86.rss.vfmadd231ps %671, %676, %681 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %683 = x86.dm.vbroadcastss [%661 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %684 = x86.rss.vfmadd231ps %672, %676, %683 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %685 = x86.dm.vbroadcastss [%661 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %686 = x86.ri.add %661, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %687 = x86.ri.add %660, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %688 = x86.rss.vfmadd231ps %673, %676, %685 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %689 = x86.dmk.vmovups[%687], %668 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %690 = x86.dm.vbroadcastss [%686] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %691 = x86.rss.vfmadd231ps %678, %689, %690 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %692 = x86.dm.vbroadcastss [%686 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %693 = x86.rss.vfmadd231ps %680, %689, %692 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %694 = x86.dm.vbroadcastss [%686 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %695 = x86.rss.vfmadd231ps %682, %689, %694 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %696 = x86.dm.vbroadcastss [%686 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %697 = x86.rss.vfmadd231ps %684, %689, %696 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %698 = x86.dm.vbroadcastss [%686 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %699 = x86.ri.add %686, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %700 = x86.ri.add %687, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %701 = x86.rss.vfmadd231ps %688, %689, %698 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %702 = x86.dmk.vmovups[%700], %668 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %703 = x86.dm.vbroadcastss [%699] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %704 = x86.rss.vfmadd231ps %691, %702, %703 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %705 = x86.dm.vbroadcastss [%699 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %706 = x86.rss.vfmadd231ps %693, %702, %705 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %707 = x86.dm.vbroadcastss [%699 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %708 = x86.rss.vfmadd231ps %695, %702, %707 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %709 = x86.dm.vbroadcastss [%699 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %710 = x86.rss.vfmadd231ps %697, %702, %709 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %711 = x86.dm.vbroadcastss [%699 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %712 = x86.ri.add %699, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %713 = x86.ri.add %700, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %714 = x86.rss.vfmadd231ps %701, %702, %711 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %715 = x86.dmk.vmovups[%713], %668 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %716 = x86.dm.vbroadcastss [%712] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %717 = x86.rss.vfmadd231ps %704, %715, %716 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %718 = x86.dm.vbroadcastss [%712 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %719 = x86.rss.vfmadd231ps %706, %715, %718 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %720 = x86.dm.vbroadcastss [%712 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %721 = x86.rss.vfmadd231ps %708, %715, %720 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %722 = x86.dm.vbroadcastss [%712 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %723 = x86.rss.vfmadd231ps %710, %715, %722 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %724 = x86.dm.vbroadcastss [%712 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %725 = x86.ri.add %712, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %726 = x86.ri.add %713, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %727 = x86.rss.vfmadd231ps %714, %715, %724 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %728 = x86.si.cmp %675, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %728 : !x86.rflags<rflags>, ^bb16(%726 : !x86.reg64<rdi>, %725 : !x86.reg64<rsi>, %662 : !x86.reg64<rdx>, %663 : !x86.reg64<rbp>, %664 : !x86.reg64<rsp>, %665 : !x86.reg64<r11>, %666 : !x86.reg64<r10>, %667 : !x86.reg64<r15>, %668 : !x86.avx512maskreg<k1>, %717 : !x86.avx512reg<zmm27>, %719 : !x86.avx512reg<zmm28>, %721 : !x86.avx512reg<zmm29>, %723 : !x86.avx512reg<zmm30>, %727 : !x86.avx512reg<zmm31>, %675 : !x86.reg64<r12>), ^bb17(%726 : !x86.reg64<rdi>, %725 : !x86.reg64<rsi>, %662 : !x86.reg64<rdx>, %663 : !x86.reg64<rbp>, %664 : !x86.reg64<rsp>, %665 : !x86.reg64<r11>, %666 : !x86.reg64<r10>, %667 : !x86.reg64<r15>, %668 : !x86.avx512maskreg<k1>, %717 : !x86.avx512reg<zmm27>, %719 : !x86.avx512reg<zmm28>, %721 : !x86.avx512reg<zmm29>, %723 : !x86.avx512reg<zmm30>, %727 : !x86.avx512reg<zmm31>, %675 : !x86.reg64<r12>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb17(%729: !x86.reg64<rdi>, %730: !x86.reg64<rsi>, %731: !x86.reg64<rdx>, %732: !x86.reg64<rbp>, %733: !x86.reg64<rsp>, %734: !x86.reg64<r11>, %735: !x86.reg64<r10>, %736: !x86.reg64<r15>, %737: !x86.avx512maskreg<k1>, %738: !x86.avx512reg<zmm27>, %739: !x86.avx512reg<zmm28>, %740: !x86.avx512reg<zmm29>, %741: !x86.avx512reg<zmm30>, %742: !x86.avx512reg<zmm31>, %743: !x86.reg64<r12>):
// CHECK-IR-LIBXSMM-NEXT:      %744 = x86.ri.sub %730, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%731], %738, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%731 + 280], %739, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%731 + 560], %740, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%731 + 840], %741, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovups[%731 + 1120], %742, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %745 = x86.ri.add %731, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %746 = x86.ri.sub %729, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %747 = x86.si.cmp %735, 70 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %747 : !x86.rflags<rflags>, ^bb15(%746 : !x86.reg64<rdi>, %744 : !x86.reg64<rsi>, %745 : !x86.reg64<rdx>, %732 : !x86.reg64<rbp>, %733 : !x86.reg64<rsp>, %734 : !x86.reg64<r11>, %735 : !x86.reg64<r10>, %736 : !x86.reg64<r15>, %737 : !x86.avx512maskreg<k1>), ^bb18(%746 : !x86.reg64<rdi>, %744 : !x86.reg64<rsi>, %745 : !x86.reg64<rdx>, %732 : !x86.reg64<rbp>, %733 : !x86.reg64<rsp>, %734 : !x86.reg64<r11>, %735 : !x86.reg64<r10>, %736 : !x86.reg64<r15>, %737 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb18(%748: !x86.reg64<rdi>, %749: !x86.reg64<rsi>, %750: !x86.reg64<rdx>, %751: !x86.reg64<rbp>, %752: !x86.reg64<rsp>, %753: !x86.reg64<r11>, %754: !x86.reg64<r10>, %755: !x86.reg64<r15>, %756: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      %757 = x86.ri.add %750, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %758 = x86.ri.add %749, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %759 = x86.ri.sub %748, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %760 = x86.si.cmp %753, 38 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %760 : !x86.rflags<rflags>, ^bb10(%759 : !x86.reg64<rdi>, %758 : !x86.reg64<rsi>, %757 : !x86.reg64<rdx>, %751 : !x86.reg64<rbp>, %752 : !x86.reg64<rsp>, %753 : !x86.reg64<r11>), ^bb19(%759 : !x86.reg64<rdi>, %758 : !x86.reg64<rsi>, %757 : !x86.reg64<rdx>, %751 : !x86.reg64<rbp>, %752 : !x86.reg64<rsp>, %753 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb19(%761: !x86.reg64<rdi>, %762: !x86.reg64<rsi>, %763: !x86.reg64<rdx>, %764: !x86.reg64<rbp>, %765: !x86.reg64<rsp>, %766: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %767 = x86.ds.mov %764 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %768, %769 = x86.d.pop %767 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }

