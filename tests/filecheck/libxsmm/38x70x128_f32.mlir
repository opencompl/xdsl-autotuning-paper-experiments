// RUN: libxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

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
