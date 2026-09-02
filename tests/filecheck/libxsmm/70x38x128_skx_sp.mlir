// RUN: libxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push r12
// CHECK-MANUAL-NEXT:      push r15
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      mov rbp, rsp
// CHECK-MANUAL-NEXT:      sub rsp, 192
// CHECK-MANUAL-NEXT:      mov r10, -64
// CHECK-MANUAL-NEXT:      and rsp, r10
// CHECK-MANUAL-NEXT:      mov r11, 0
// CHECK-MANUAL-NEXT:  [[SCF_N_BODY_0:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 6
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[SCF_M_BODY_0:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 64
// CHECK-MANUAL-NEXT:      vmovups zmm8, [rdx]
// CHECK-MANUAL-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovups zmm11, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovups zmm12, [rdx+280]
// CHECK-MANUAL-NEXT:      vmovups zmm13, [rdx+344]
// CHECK-MANUAL-NEXT:      vmovups zmm14, [rdx+408]
// CHECK-MANUAL-NEXT:      vmovups zmm15, [rdx+472]
// CHECK-MANUAL-NEXT:      vmovups zmm16, [rdx+560]
// CHECK-MANUAL-NEXT:      vmovups zmm17, [rdx+624]
// CHECK-MANUAL-NEXT:      vmovups zmm18, [rdx+688]
// CHECK-MANUAL-NEXT:      vmovups zmm19, [rdx+752]
// CHECK-MANUAL-NEXT:      vmovups zmm20, [rdx+840]
// CHECK-MANUAL-NEXT:      vmovups zmm21, [rdx+904]
// CHECK-MANUAL-NEXT:      vmovups zmm22, [rdx+968]
// CHECK-MANUAL-NEXT:      vmovups zmm23, [rdx+1032]
// CHECK-MANUAL-NEXT:      vmovups zmm24, [rdx+1120]
// CHECK-MANUAL-NEXT:      vmovups zmm25, [rdx+1184]
// CHECK-MANUAL-NEXT:      vmovups zmm26, [rdx+1248]
// CHECK-MANUAL-NEXT:      vmovups zmm27, [rdx+1312]
// CHECK-MANUAL-NEXT:      vmovups zmm28, [rdx+1400]
// CHECK-MANUAL-NEXT:      vmovups zmm29, [rdx+1464]
// CHECK-MANUAL-NEXT:      vmovups zmm30, [rdx+1528]
// CHECK-MANUAL-NEXT:      vmovups zmm31, [rdx+1592]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[SCF_K_BODY_0:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      cmp r12, 128
// CHECK-MANUAL-NEXT:      jl [[SCF_K_BODY_0]]
// CHECK-LIBXSMM-NEXT:     sub rsi, 512
// CHECK-MANUAL-NEXT:      vmovups [rdx], zmm8
// CHECK-MANUAL-NEXT:      vmovups [rdx+64], zmm9
// CHECK-MANUAL-NEXT:      vmovups [rdx+128], zmm10
// CHECK-MANUAL-NEXT:      vmovups [rdx+192], zmm11
// CHECK-MANUAL-NEXT:      vmovups [rdx+280], zmm12
// CHECK-MANUAL-NEXT:      vmovups [rdx+344], zmm13
// CHECK-MANUAL-NEXT:      vmovups [rdx+408], zmm14
// CHECK-MANUAL-NEXT:      vmovups [rdx+472], zmm15
// CHECK-MANUAL-NEXT:      vmovups [rdx+560], zmm16
// CHECK-MANUAL-NEXT:      vmovups [rdx+624], zmm17
// CHECK-MANUAL-NEXT:      vmovups [rdx+688], zmm18
// CHECK-MANUAL-NEXT:      vmovups [rdx+752], zmm19
// CHECK-MANUAL-NEXT:      vmovups [rdx+840], zmm20
// CHECK-MANUAL-NEXT:      vmovups [rdx+904], zmm21
// CHECK-MANUAL-NEXT:      vmovups [rdx+968], zmm22
// CHECK-MANUAL-NEXT:      vmovups [rdx+1032], zmm23
// CHECK-MANUAL-NEXT:      vmovups [rdx+1120], zmm24
// CHECK-MANUAL-NEXT:      vmovups [rdx+1184], zmm25
// CHECK-MANUAL-NEXT:      vmovups [rdx+1248], zmm26
// CHECK-MANUAL-NEXT:      vmovups [rdx+1312], zmm27
// CHECK-MANUAL-NEXT:      vmovups [rdx+1400], zmm28
// CHECK-MANUAL-NEXT:      vmovups [rdx+1464], zmm29
// CHECK-MANUAL-NEXT:      vmovups [rdx+1528], zmm30
// CHECK-MANUAL-NEXT:      vmovups [rdx+1592], zmm31
// CHECK-COMPXSMM-NEXT:    sub rdi, 35584
// CHECK-COMPXSMM-NEXT:    sub rsi, 512
// CHECK-MANUAL-NEXT:      add rdx, 256
// CHECK-LIBXSMM-NEXT:     sub rdi, 35584
// CHECK-MANUAL-NEXT:      cmp r10, 64
// CHECK-MANUAL-NEXT:      jl [[SCF_M_BODY_0]]
// CHECK-MANUAL-NEXT:      mov r15, 63
// CHECK-MANUAL-NEXT:      kmovw k1, r15d
// CHECK-LIBXSMM-NEXT:     mov r10, 64
// CHECK-COMPXSMM-NEXT:    mov r10, 0
// CHECK-MANUAL-NEXT:  [[SCF_M_BODY_1:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 6
// CHECK-MANUAL-NEXT:      vmovups zmm26 {k1}{z}, [rdx]
// CHECK-MANUAL-NEXT:      vmovups zmm27 {k1}{z}, [rdx+280]
// CHECK-MANUAL-NEXT:      vmovups zmm28 {k1}{z}, [rdx+560]
// CHECK-MANUAL-NEXT:      vmovups zmm29 {k1}{z}, [rdx+840]
// CHECK-MANUAL-NEXT:      vmovups zmm30 {k1}{z}, [rdx+1120]
// CHECK-MANUAL-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1400]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[SCF_K_BODY_1:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-MANUAL-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-MANUAL-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-MANUAL-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-MANUAL-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-MANUAL-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-MANUAL-NEXT:      vmovups zmm0 {k1}{z}, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm1 {k1}{z}, [rdi+280]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm0, [rsi]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm0, [rsi+512]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+1024]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1536]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+2048]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2560]{1to16}
// CHECK-MANUAL-NEXT:      vmovups zmm0 {k1}{z}, [rdi+560]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+4]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+516]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+1028]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+1540]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+2052]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+2564]{1to16}
// CHECK-MANUAL-NEXT:      vmovups zmm1 {k1}{z}, [rdi+840]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm0, [rsi+8]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm0, [rsi+520]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+1032]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1544]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+2056]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2568]{1to16}
// CHECK-MANUAL-NEXT:      add rdi, 1120
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+12]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+524]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+1036]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+1548]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+2060]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+2572]{1to16}
// CHECK-MANUAL-NEXT:      add rsi, 16
// CHECK-MANUAL-NEXT:      vaddps zmm26, zmm20, zmm26
// CHECK-MANUAL-NEXT:      vaddps zmm27, zmm21, zmm27
// CHECK-MANUAL-NEXT:      vaddps zmm28, zmm22, zmm28
// CHECK-MANUAL-NEXT:      vaddps zmm29, zmm23, zmm29
// CHECK-MANUAL-NEXT:      vaddps zmm30, zmm24, zmm30
// CHECK-MANUAL-NEXT:      vaddps zmm31, zmm25, zmm31
// CHECK-MANUAL-NEXT:      cmp r12, 128
// CHECK-MANUAL-NEXT:      jl [[SCF_K_BODY_1]]
// CHECK-LIBXSMM-NEXT:     sub rsi, 512
// CHECK-MANUAL-NEXT:      vmovups [rdx] {k1}, zmm26
// CHECK-MANUAL-NEXT:      vmovups [rdx+280] {k1}, zmm27
// CHECK-MANUAL-NEXT:      vmovups [rdx+560] {k1}, zmm28
// CHECK-MANUAL-NEXT:      vmovups [rdx+840] {k1}, zmm29
// CHECK-MANUAL-NEXT:      vmovups [rdx+1120] {k1}, zmm30
// CHECK-MANUAL-NEXT:      vmovups [rdx+1400] {k1}, zmm31
// CHECK-COMPXSMM-NEXT:    sub rdi, 35816
// CHECK-COMPXSMM-NEXT:    sub rsi, 512
// CHECK-MANUAL-NEXT:      add rdx, 24
// CHECK-LIBXSMM-NEXT:     sub rdi, 35816
// CHECK-LIBXSMM-NEXT:     cmp r10, 70
// CHECK-COMPXSMM-NEXT:    cmp r10, 6
// CHECK-MANUAL-NEXT:      jl [[SCF_M_BODY_1]]
// CHECK-LIBXSMM-NEXT:     add rdx, 1400
// CHECK-COMPXSMM-NEXT:    sub rdi, 280
// CHECK-MANUAL-NEXT:      add rsi, 3072
// CHECK-LIBXSMM-NEXT:     sub rdi, 280
// CHECK-COMPXSMM-NEXT:    add rdx, 1400
// CHECK-MANUAL-NEXT:      cmp r11, 18
// CHECK-MANUAL-NEXT:      jl [[SCF_N_BODY_0]]
// CHECK-LIBXSMM-NEXT:     mov r11, 18
// CHECK-COMPXSMM-NEXT:    mov r11, 0
// CHECK-MANUAL-NEXT:  [[SCF_N_BODY_1:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 5
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[SCF_M_BODY_2:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 64
// CHECK-MANUAL-NEXT:      vmovups zmm12, [rdx]
// CHECK-MANUAL-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovups zmm15, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovups zmm16, [rdx+280]
// CHECK-MANUAL-NEXT:      vmovups zmm17, [rdx+344]
// CHECK-MANUAL-NEXT:      vmovups zmm18, [rdx+408]
// CHECK-MANUAL-NEXT:      vmovups zmm19, [rdx+472]
// CHECK-MANUAL-NEXT:      vmovups zmm20, [rdx+560]
// CHECK-MANUAL-NEXT:      vmovups zmm21, [rdx+624]
// CHECK-MANUAL-NEXT:      vmovups zmm22, [rdx+688]
// CHECK-MANUAL-NEXT:      vmovups zmm23, [rdx+752]
// CHECK-MANUAL-NEXT:      vmovups zmm24, [rdx+840]
// CHECK-MANUAL-NEXT:      vmovups zmm25, [rdx+904]
// CHECK-MANUAL-NEXT:      vmovups zmm26, [rdx+968]
// CHECK-MANUAL-NEXT:      vmovups zmm27, [rdx+1032]
// CHECK-MANUAL-NEXT:      vmovups zmm28, [rdx+1120]
// CHECK-MANUAL-NEXT:      vmovups zmm29, [rdx+1184]
// CHECK-MANUAL-NEXT:      vmovups zmm30, [rdx+1248]
// CHECK-MANUAL-NEXT:      vmovups zmm31, [rdx+1312]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[SCF_K_BODY_2:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      vmovups zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-MANUAL-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-MANUAL-NEXT:      add rdi, 280
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm3, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm4, zmm0
// CHECK-MANUAL-NEXT:      cmp r12, 128
// CHECK-MANUAL-NEXT:      jl [[SCF_K_BODY_2]]
// CHECK-LIBXSMM-NEXT:     sub rsi, 512
// CHECK-MANUAL-NEXT:      vmovups [rdx], zmm12
// CHECK-MANUAL-NEXT:      vmovups [rdx+64], zmm13
// CHECK-MANUAL-NEXT:      vmovups [rdx+128], zmm14
// CHECK-MANUAL-NEXT:      vmovups [rdx+192], zmm15
// CHECK-MANUAL-NEXT:      vmovups [rdx+280], zmm16
// CHECK-MANUAL-NEXT:      vmovups [rdx+344], zmm17
// CHECK-MANUAL-NEXT:      vmovups [rdx+408], zmm18
// CHECK-MANUAL-NEXT:      vmovups [rdx+472], zmm19
// CHECK-MANUAL-NEXT:      vmovups [rdx+560], zmm20
// CHECK-MANUAL-NEXT:      vmovups [rdx+624], zmm21
// CHECK-MANUAL-NEXT:      vmovups [rdx+688], zmm22
// CHECK-MANUAL-NEXT:      vmovups [rdx+752], zmm23
// CHECK-MANUAL-NEXT:      vmovups [rdx+840], zmm24
// CHECK-MANUAL-NEXT:      vmovups [rdx+904], zmm25
// CHECK-MANUAL-NEXT:      vmovups [rdx+968], zmm26
// CHECK-MANUAL-NEXT:      vmovups [rdx+1032], zmm27
// CHECK-MANUAL-NEXT:      vmovups [rdx+1120], zmm28
// CHECK-MANUAL-NEXT:      vmovups [rdx+1184], zmm29
// CHECK-MANUAL-NEXT:      vmovups [rdx+1248], zmm30
// CHECK-MANUAL-NEXT:      vmovups [rdx+1312], zmm31
// CHECK-COMPXSMM-NEXT:    sub rdi, 35584
// CHECK-COMPXSMM-NEXT:    sub rsi, 512
// CHECK-MANUAL-NEXT:      add rdx, 256
// CHECK-LIBXSMM-NEXT:     sub rdi, 35584
// CHECK-MANUAL-NEXT:      cmp r10, 64
// CHECK-MANUAL-NEXT:      jl [[SCF_M_BODY_2]]
// CHECK-MANUAL-NEXT:      mov r15, 63
// CHECK-MANUAL-NEXT:      kmovw k1, r15d
// CHECK-LIBXSMM-NEXT:     mov r10, 64
// CHECK-COMPXSMM-NEXT:    mov r10, 0
// CHECK-MANUAL-NEXT:  [[SCF_M_BODY_3:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 6
// CHECK-MANUAL-NEXT:      vmovups zmm27 {k1}{z}, [rdx]
// CHECK-MANUAL-NEXT:      vmovups zmm28 {k1}{z}, [rdx+280]
// CHECK-MANUAL-NEXT:      vmovups zmm29 {k1}{z}, [rdx+560]
// CHECK-MANUAL-NEXT:      vmovups zmm30 {k1}{z}, [rdx+840]
// CHECK-MANUAL-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1120]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[SCF_K_BODY_3:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-MANUAL-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-MANUAL-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-MANUAL-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-MANUAL-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-MANUAL-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-MANUAL-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-MANUAL-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-MANUAL-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-MANUAL-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-MANUAL-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-MANUAL-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-MANUAL-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-MANUAL-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-MANUAL-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-MANUAL-NEXT:      vmovups zmm0 {k1}{z}, [rdi]
// CHECK-MANUAL-NEXT:      vmovups zmm1 {k1}{z}, [rdi+280]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm27, zmm0, [rsi]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+512]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1024]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+1536]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2048]{1to16}
// CHECK-MANUAL-NEXT:      vmovups zmm0 {k1}{z}, [rdi+560]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+4]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+516]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+1028]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+1540]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm26, zmm1, [rsi+2052]{1to16}
// CHECK-MANUAL-NEXT:      vmovups zmm1 {k1}{z}, [rdi+840]
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm17, zmm0, [rsi+8]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm18, zmm0, [rsi+520]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm19, zmm0, [rsi+1032]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm20, zmm0, [rsi+1544]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm21, zmm0, [rsi+2056]{1to16}
// CHECK-MANUAL-NEXT:      add rdi, 1120
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm12, zmm1, [rsi+12]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm13, zmm1, [rsi+524]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm14, zmm1, [rsi+1036]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm15, zmm1, [rsi+1548]{1to16}
// CHECK-MANUAL-NEXT:      vfmadd231ps zmm16, zmm1, [rsi+2060]{1to16}
// CHECK-MANUAL-NEXT:      add rsi, 16
// CHECK-MANUAL-NEXT:      vaddps zmm27, zmm22, zmm27
// CHECK-MANUAL-NEXT:      vaddps zmm28, zmm23, zmm28
// CHECK-MANUAL-NEXT:      vaddps zmm29, zmm24, zmm29
// CHECK-MANUAL-NEXT:      vaddps zmm30, zmm25, zmm30
// CHECK-MANUAL-NEXT:      vaddps zmm31, zmm26, zmm31
// CHECK-MANUAL-NEXT:      vaddps zmm27, zmm17, zmm27
// CHECK-MANUAL-NEXT:      vaddps zmm28, zmm18, zmm28
// CHECK-MANUAL-NEXT:      vaddps zmm29, zmm19, zmm29
// CHECK-MANUAL-NEXT:      vaddps zmm30, zmm20, zmm30
// CHECK-MANUAL-NEXT:      vaddps zmm31, zmm21, zmm31
// CHECK-MANUAL-NEXT:      vaddps zmm27, zmm12, zmm27
// CHECK-MANUAL-NEXT:      vaddps zmm28, zmm13, zmm28
// CHECK-MANUAL-NEXT:      vaddps zmm29, zmm14, zmm29
// CHECK-MANUAL-NEXT:      vaddps zmm30, zmm15, zmm30
// CHECK-MANUAL-NEXT:      vaddps zmm31, zmm16, zmm31
// CHECK-MANUAL-NEXT:      cmp r12, 128
// CHECK-MANUAL-NEXT:      jl [[SCF_K_BODY_3]]
// CHECK-LIBXSMM-NEXT:     sub rsi, 512
// CHECK-MANUAL-NEXT:      vmovups [rdx] {k1}, zmm27
// CHECK-MANUAL-NEXT:      vmovups [rdx+280] {k1}, zmm28
// CHECK-MANUAL-NEXT:      vmovups [rdx+560] {k1}, zmm29
// CHECK-MANUAL-NEXT:      vmovups [rdx+840] {k1}, zmm30
// CHECK-MANUAL-NEXT:      vmovups [rdx+1120] {k1}, zmm31
// CHECK-COMPXSMM-NEXT:    sub rdi, 35816
// CHECK-COMPXSMM-NEXT:    sub rsi, 512
// CHECK-MANUAL-NEXT:      add rdx, 24
// CHECK-LIBXSMM-NEXT:     sub rdi, 35816
// CHECK-LIBXSMM-NEXT:     cmp r10, 70
// CHECK-COMPXSMM-NEXT:    cmp r10, 6
// CHECK-MANUAL-NEXT:      jl [[SCF_M_BODY_3]]
// CHECK-LIBXSMM-NEXT:     add rdx, 1120
// CHECK-COMPXSMM-NEXT:    sub rdi, 280
// CHECK-MANUAL-NEXT:      add rsi, 2560
// CHECK-LIBXSMM-NEXT:     sub rdi, 280
// CHECK-COMPXSMM-NEXT:    add rdx, 1120
// CHECK-LIBXSMM-NEXT:     cmp r11, 38
// CHECK-COMPXSMM-NEXT:    cmp r11, 20
// CHECK-MANUAL-NEXT:      jl [[SCF_N_BODY_1]]
// CHECK-MANUAL-NEXT:      mov rsp, rbp
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      pop r15
// CHECK-MANUAL-NEXT:      pop r12
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
// CHECK-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %26 = x86.ri.add %25, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %27 = x86.dm.vmovups [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %28 = x86.dm.vmovups [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %29 = x86.dm.vmovups [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %30 = x86.dm.vmovups [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %31 = x86.dm.vmovups [%21 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %32 = x86.dm.vmovups [%21 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %33 = x86.dm.vmovups [%21 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %34 = x86.dm.vmovups [%21 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %35 = x86.dm.vmovups [%21 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %36 = x86.dm.vmovups [%21 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %37 = x86.dm.vmovups [%21 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %38 = x86.dm.vmovups [%21 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %39 = x86.dm.vmovups [%21 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %40 = x86.dm.vmovups [%21 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %41 = x86.dm.vmovups [%21 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %42 = x86.dm.vmovups [%21 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %43 = x86.dm.vmovups [%21 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %44 = x86.dm.vmovups [%21 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %45 = x86.dm.vmovups [%21 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %46 = x86.dm.vmovups [%21 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %47 = x86.dm.vmovups [%21 + 1400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %48 = x86.dm.vmovups [%21 + 1464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %49 = x86.dm.vmovups [%21 + 1528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %50 = x86.dm.vmovups [%21 + 1592] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %51 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb3(%19 : !x86.reg64<rdi>, %20 : !x86.reg64<rsi>, %21 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>, %27 : !x86.avx512reg<zmm8>, %28 : !x86.avx512reg<zmm9>, %29 : !x86.avx512reg<zmm10>, %30 : !x86.avx512reg<zmm11>, %31 : !x86.avx512reg<zmm12>, %32 : !x86.avx512reg<zmm13>, %33 : !x86.avx512reg<zmm14>, %34 : !x86.avx512reg<zmm15>, %35 : !x86.avx512reg<zmm16>, %36 : !x86.avx512reg<zmm17>, %37 : !x86.avx512reg<zmm18>, %38 : !x86.avx512reg<zmm19>, %39 : !x86.avx512reg<zmm20>, %40 : !x86.avx512reg<zmm21>, %41 : !x86.avx512reg<zmm22>, %42 : !x86.avx512reg<zmm23>, %43 : !x86.avx512reg<zmm24>, %44 : !x86.avx512reg<zmm25>, %45 : !x86.avx512reg<zmm26>, %46 : !x86.avx512reg<zmm27>, %47 : !x86.avx512reg<zmm28>, %48 : !x86.avx512reg<zmm29>, %49 : !x86.avx512reg<zmm30>, %50 : !x86.avx512reg<zmm31>, %51 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb3(%52: !x86.reg64<rdi>, %53: !x86.reg64<rsi>, %54: !x86.reg64<rdx>, %55: !x86.reg64<rbp>, %56: !x86.reg64<rsp>, %57: !x86.reg64<r11>, %58: !x86.reg64<r10>, %59: !x86.avx512reg<zmm8>, %60: !x86.avx512reg<zmm9>, %61: !x86.avx512reg<zmm10>, %62: !x86.avx512reg<zmm11>, %63: !x86.avx512reg<zmm12>, %64: !x86.avx512reg<zmm13>, %65: !x86.avx512reg<zmm14>, %66: !x86.avx512reg<zmm15>, %67: !x86.avx512reg<zmm16>, %68: !x86.avx512reg<zmm17>, %69: !x86.avx512reg<zmm18>, %70: !x86.avx512reg<zmm19>, %71: !x86.avx512reg<zmm20>, %72: !x86.avx512reg<zmm21>, %73: !x86.avx512reg<zmm22>, %74: !x86.avx512reg<zmm23>, %75: !x86.avx512reg<zmm24>, %76: !x86.avx512reg<zmm25>, %77: !x86.avx512reg<zmm26>, %78: !x86.avx512reg<zmm27>, %79: !x86.avx512reg<zmm28>, %80: !x86.avx512reg<zmm29>, %81: !x86.avx512reg<zmm30>, %82: !x86.avx512reg<zmm31>, %83: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l35"
// CHECK-NEXT:      %84 = x86.ri.add %83, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %85 = x86.dm.vmovups [%52] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %86 = x86.dm.vmovups [%52 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %87 = x86.dm.vmovups [%52 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %88 = x86.dm.vmovups [%52 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %89 = x86.dm.vbroadcastss [%53] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %90 = x86.rss.vfmadd231ps %59, %85, %89 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231ps %60, %86, %89 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %92 = x86.rss.vfmadd231ps %61, %87, %89 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %93 = x86.rss.vfmadd231ps %62, %88, %89 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %94 = x86.dm.vbroadcastss [%53 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %95 = x86.rss.vfmadd231ps %63, %85, %94 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %96 = x86.rss.vfmadd231ps %64, %86, %94 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %97 = x86.rss.vfmadd231ps %65, %87, %94 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %98 = x86.rss.vfmadd231ps %66, %88, %94 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %99 = x86.dm.vbroadcastss [%53 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %100 = x86.rss.vfmadd231ps %67, %85, %99 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %101 = x86.rss.vfmadd231ps %68, %86, %99 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %102 = x86.rss.vfmadd231ps %69, %87, %99 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %103 = x86.rss.vfmadd231ps %70, %88, %99 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %104 = x86.dm.vbroadcastss [%53 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %105 = x86.rss.vfmadd231ps %71, %85, %104 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %106 = x86.rss.vfmadd231ps %72, %86, %104 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %107 = x86.rss.vfmadd231ps %73, %87, %104 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %108 = x86.rss.vfmadd231ps %74, %88, %104 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %109 = x86.dm.vbroadcastss [%53 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %110 = x86.rss.vfmadd231ps %75, %85, %109 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %111 = x86.rss.vfmadd231ps %76, %86, %109 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %112 = x86.rss.vfmadd231ps %77, %87, %109 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %113 = x86.rss.vfmadd231ps %78, %88, %109 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %114 = x86.dm.vbroadcastss [%53 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %115 = x86.ri.add %53, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %116 = x86.ri.add %52, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %117 = x86.rss.vfmadd231ps %79, %85, %114 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %118 = x86.rss.vfmadd231ps %80, %86, %114 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %119 = x86.rss.vfmadd231ps %81, %87, %114 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %120 = x86.rss.vfmadd231ps %82, %88, %114 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %121 = x86.dm.vmovups [%116] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %122 = x86.dm.vmovups [%116 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %123 = x86.dm.vmovups [%116 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %124 = x86.dm.vmovups [%116 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %125 = x86.dm.vbroadcastss [%115] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %126 = x86.rss.vfmadd231ps %90, %121, %125 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %127 = x86.rss.vfmadd231ps %91, %122, %125 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %128 = x86.rss.vfmadd231ps %92, %123, %125 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %129 = x86.rss.vfmadd231ps %93, %124, %125 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %130 = x86.dm.vbroadcastss [%115 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %131 = x86.rss.vfmadd231ps %95, %121, %130 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %132 = x86.rss.vfmadd231ps %96, %122, %130 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %133 = x86.rss.vfmadd231ps %97, %123, %130 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %134 = x86.rss.vfmadd231ps %98, %124, %130 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %135 = x86.dm.vbroadcastss [%115 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %136 = x86.rss.vfmadd231ps %100, %121, %135 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %137 = x86.rss.vfmadd231ps %101, %122, %135 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %138 = x86.rss.vfmadd231ps %102, %123, %135 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %139 = x86.rss.vfmadd231ps %103, %124, %135 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %140 = x86.dm.vbroadcastss [%115 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %141 = x86.rss.vfmadd231ps %105, %121, %140 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %142 = x86.rss.vfmadd231ps %106, %122, %140 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %143 = x86.rss.vfmadd231ps %107, %123, %140 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %144 = x86.rss.vfmadd231ps %108, %124, %140 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %145 = x86.dm.vbroadcastss [%115 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %146 = x86.rss.vfmadd231ps %110, %121, %145 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %147 = x86.rss.vfmadd231ps %111, %122, %145 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %148 = x86.rss.vfmadd231ps %112, %123, %145 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %149 = x86.rss.vfmadd231ps %113, %124, %145 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %150 = x86.dm.vbroadcastss [%115 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %151 = x86.ri.add %115, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %152 = x86.ri.add %116, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %153 = x86.rss.vfmadd231ps %117, %121, %150 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %154 = x86.rss.vfmadd231ps %118, %122, %150 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %155 = x86.rss.vfmadd231ps %119, %123, %150 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %156 = x86.rss.vfmadd231ps %120, %124, %150 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %157 = x86.dm.vmovups [%152] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %158 = x86.dm.vmovups [%152 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %159 = x86.dm.vmovups [%152 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %160 = x86.dm.vmovups [%152 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %161 = x86.dm.vbroadcastss [%151] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %162 = x86.rss.vfmadd231ps %126, %157, %161 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %163 = x86.rss.vfmadd231ps %127, %158, %161 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %164 = x86.rss.vfmadd231ps %128, %159, %161 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %165 = x86.rss.vfmadd231ps %129, %160, %161 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %166 = x86.dm.vbroadcastss [%151 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %167 = x86.rss.vfmadd231ps %131, %157, %166 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %168 = x86.rss.vfmadd231ps %132, %158, %166 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %169 = x86.rss.vfmadd231ps %133, %159, %166 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %170 = x86.rss.vfmadd231ps %134, %160, %166 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %171 = x86.dm.vbroadcastss [%151 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %172 = x86.rss.vfmadd231ps %136, %157, %171 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %173 = x86.rss.vfmadd231ps %137, %158, %171 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %174 = x86.rss.vfmadd231ps %138, %159, %171 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %175 = x86.rss.vfmadd231ps %139, %160, %171 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %176 = x86.dm.vbroadcastss [%151 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %177 = x86.rss.vfmadd231ps %141, %157, %176 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %178 = x86.rss.vfmadd231ps %142, %158, %176 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %179 = x86.rss.vfmadd231ps %143, %159, %176 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %180 = x86.rss.vfmadd231ps %144, %160, %176 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %181 = x86.dm.vbroadcastss [%151 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %182 = x86.rss.vfmadd231ps %146, %157, %181 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %183 = x86.rss.vfmadd231ps %147, %158, %181 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %184 = x86.rss.vfmadd231ps %148, %159, %181 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %185 = x86.rss.vfmadd231ps %149, %160, %181 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %186 = x86.dm.vbroadcastss [%151 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %187 = x86.ri.add %151, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %188 = x86.ri.add %152, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %189 = x86.rss.vfmadd231ps %153, %157, %186 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %190 = x86.rss.vfmadd231ps %154, %158, %186 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %191 = x86.rss.vfmadd231ps %155, %159, %186 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %192 = x86.rss.vfmadd231ps %156, %160, %186 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %193 = x86.dm.vmovups [%188] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %194 = x86.dm.vmovups [%188 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %195 = x86.dm.vmovups [%188 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %196 = x86.dm.vmovups [%188 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %197 = x86.dm.vbroadcastss [%187] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %198 = x86.rss.vfmadd231ps %162, %193, %197 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %199 = x86.rss.vfmadd231ps %163, %194, %197 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %200 = x86.rss.vfmadd231ps %164, %195, %197 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %201 = x86.rss.vfmadd231ps %165, %196, %197 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %202 = x86.dm.vbroadcastss [%187 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %203 = x86.rss.vfmadd231ps %167, %193, %202 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %204 = x86.rss.vfmadd231ps %168, %194, %202 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %205 = x86.rss.vfmadd231ps %169, %195, %202 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %206 = x86.rss.vfmadd231ps %170, %196, %202 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %207 = x86.dm.vbroadcastss [%187 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %208 = x86.rss.vfmadd231ps %172, %193, %207 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %209 = x86.rss.vfmadd231ps %173, %194, %207 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %210 = x86.rss.vfmadd231ps %174, %195, %207 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %211 = x86.rss.vfmadd231ps %175, %196, %207 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %212 = x86.dm.vbroadcastss [%187 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %213 = x86.rss.vfmadd231ps %177, %193, %212 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %214 = x86.rss.vfmadd231ps %178, %194, %212 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %215 = x86.rss.vfmadd231ps %179, %195, %212 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %216 = x86.rss.vfmadd231ps %180, %196, %212 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %217 = x86.dm.vbroadcastss [%187 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %218 = x86.rss.vfmadd231ps %182, %193, %217 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %219 = x86.rss.vfmadd231ps %183, %194, %217 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %220 = x86.rss.vfmadd231ps %184, %195, %217 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %221 = x86.rss.vfmadd231ps %185, %196, %217 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %222 = x86.dm.vbroadcastss [%187 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %223 = x86.ri.add %187, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %224 = x86.ri.add %188, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %225 = x86.rss.vfmadd231ps %189, %193, %222 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %226 = x86.rss.vfmadd231ps %190, %194, %222 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %227 = x86.rss.vfmadd231ps %191, %195, %222 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %228 = x86.rss.vfmadd231ps %192, %196, %222 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %229 = x86.si.cmp %84, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %229 : !x86.rflags<rflags>, ^bb3(%224 : !x86.reg64<rdi>, %223 : !x86.reg64<rsi>, %54 : !x86.reg64<rdx>, %55 : !x86.reg64<rbp>, %56 : !x86.reg64<rsp>, %57 : !x86.reg64<r11>, %58 : !x86.reg64<r10>, %198 : !x86.avx512reg<zmm8>, %199 : !x86.avx512reg<zmm9>, %200 : !x86.avx512reg<zmm10>, %201 : !x86.avx512reg<zmm11>, %203 : !x86.avx512reg<zmm12>, %204 : !x86.avx512reg<zmm13>, %205 : !x86.avx512reg<zmm14>, %206 : !x86.avx512reg<zmm15>, %208 : !x86.avx512reg<zmm16>, %209 : !x86.avx512reg<zmm17>, %210 : !x86.avx512reg<zmm18>, %211 : !x86.avx512reg<zmm19>, %213 : !x86.avx512reg<zmm20>, %214 : !x86.avx512reg<zmm21>, %215 : !x86.avx512reg<zmm22>, %216 : !x86.avx512reg<zmm23>, %218 : !x86.avx512reg<zmm24>, %219 : !x86.avx512reg<zmm25>, %220 : !x86.avx512reg<zmm26>, %221 : !x86.avx512reg<zmm27>, %225 : !x86.avx512reg<zmm28>, %226 : !x86.avx512reg<zmm29>, %227 : !x86.avx512reg<zmm30>, %228 : !x86.avx512reg<zmm31>, %84 : !x86.reg64<r12>), ^bb4(%224 : !x86.reg64<rdi>, %223 : !x86.reg64<rsi>, %54 : !x86.reg64<rdx>, %55 : !x86.reg64<rbp>, %56 : !x86.reg64<rsp>, %57 : !x86.reg64<r11>, %58 : !x86.reg64<r10>, %198 : !x86.avx512reg<zmm8>, %199 : !x86.avx512reg<zmm9>, %200 : !x86.avx512reg<zmm10>, %201 : !x86.avx512reg<zmm11>, %203 : !x86.avx512reg<zmm12>, %204 : !x86.avx512reg<zmm13>, %205 : !x86.avx512reg<zmm14>, %206 : !x86.avx512reg<zmm15>, %208 : !x86.avx512reg<zmm16>, %209 : !x86.avx512reg<zmm17>, %210 : !x86.avx512reg<zmm18>, %211 : !x86.avx512reg<zmm19>, %213 : !x86.avx512reg<zmm20>, %214 : !x86.avx512reg<zmm21>, %215 : !x86.avx512reg<zmm22>, %216 : !x86.avx512reg<zmm23>, %218 : !x86.avx512reg<zmm24>, %219 : !x86.avx512reg<zmm25>, %220 : !x86.avx512reg<zmm26>, %221 : !x86.avx512reg<zmm27>, %225 : !x86.avx512reg<zmm28>, %226 : !x86.avx512reg<zmm29>, %227 : !x86.avx512reg<zmm30>, %228 : !x86.avx512reg<zmm31>, %84 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb4(%230: !x86.reg64<rdi>, %231: !x86.reg64<rsi>, %232: !x86.reg64<rdx>, %233: !x86.reg64<rbp>, %234: !x86.reg64<rsp>, %235: !x86.reg64<r11>, %236: !x86.reg64<r10>, %237: !x86.avx512reg<zmm8>, %238: !x86.avx512reg<zmm9>, %239: !x86.avx512reg<zmm10>, %240: !x86.avx512reg<zmm11>, %241: !x86.avx512reg<zmm12>, %242: !x86.avx512reg<zmm13>, %243: !x86.avx512reg<zmm14>, %244: !x86.avx512reg<zmm15>, %245: !x86.avx512reg<zmm16>, %246: !x86.avx512reg<zmm17>, %247: !x86.avx512reg<zmm18>, %248: !x86.avx512reg<zmm19>, %249: !x86.avx512reg<zmm20>, %250: !x86.avx512reg<zmm21>, %251: !x86.avx512reg<zmm22>, %252: !x86.avx512reg<zmm23>, %253: !x86.avx512reg<zmm24>, %254: !x86.avx512reg<zmm25>, %255: !x86.avx512reg<zmm26>, %256: !x86.avx512reg<zmm27>, %257: !x86.avx512reg<zmm28>, %258: !x86.avx512reg<zmm29>, %259: !x86.avx512reg<zmm30>, %260: !x86.avx512reg<zmm31>, %261: !x86.reg64<r12>):
// CHECK-NEXT:      %262 = x86.ri.sub %231, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%232], %237 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 64], %238 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 128], %239 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 192], %240 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 280], %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 344], %242 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 408], %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 472], %244 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 560], %245 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 624], %246 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 688], %247 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 752], %248 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 840], %249 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 904], %250 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 968], %251 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1032], %252 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1120], %253 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1184], %254 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1248], %255 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1312], %256 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1400], %257 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1464], %258 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1528], %259 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%232 + 1592], %260 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %263 = x86.ri.add %232, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %264 = x86.ri.sub %230, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %265 = x86.si.cmp %236, 64 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %265 : !x86.rflags<rflags>, ^bb2(%264 : !x86.reg64<rdi>, %262 : !x86.reg64<rsi>, %263 : !x86.reg64<rdx>, %233 : !x86.reg64<rbp>, %234 : !x86.reg64<rsp>, %235 : !x86.reg64<r11>, %236 : !x86.reg64<r10>), ^bb5(%264 : !x86.reg64<rdi>, %262 : !x86.reg64<rsi>, %263 : !x86.reg64<rdx>, %233 : !x86.reg64<rbp>, %234 : !x86.reg64<rsp>, %235 : !x86.reg64<r11>, %236 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb5(%266: !x86.reg64<rdi>, %267: !x86.reg64<rsi>, %268: !x86.reg64<rdx>, %269: !x86.reg64<rbp>, %270: !x86.reg64<rsp>, %271: !x86.reg64<r11>, %272: !x86.reg64<r10>):
// CHECK-NEXT:      %273 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %274 = x86.ks.kmovw %273 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %275 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb6(%266 : !x86.reg64<rdi>, %267 : !x86.reg64<rsi>, %268 : !x86.reg64<rdx>, %269 : !x86.reg64<rbp>, %270 : !x86.reg64<rsp>, %271 : !x86.reg64<r11>, %275 : !x86.reg64<r10>, %274 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb6(%276: !x86.reg64<rdi>, %277: !x86.reg64<rsi>, %278: !x86.reg64<rdx>, %279: !x86.reg64<rbp>, %280: !x86.reg64<rsp>, %281: !x86.reg64<r11>, %282: !x86.reg64<r10>, %283: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      x86.label "l36"
// CHECK-NEXT:      %284 = x86.ri.add %282, 6 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %285 = x86.dmk.vmovups[%278], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %286 = x86.dmk.vmovups[%278 + 280], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %287 = x86.dmk.vmovups[%278 + 560], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %288 = x86.dmk.vmovups[%278 + 840], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %289 = x86.dmk.vmovups[%278 + 1120], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %290 = x86.dmk.vmovups[%278 + 1400], %283 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %291 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb7(%276 : !x86.reg64<rdi>, %277 : !x86.reg64<rsi>, %278 : !x86.reg64<rdx>, %279 : !x86.reg64<rbp>, %280 : !x86.reg64<rsp>, %281 : !x86.reg64<r11>, %284 : !x86.reg64<r10>, %283 : !x86.avx512maskreg<k1>, %285 : !x86.avx512reg<zmm26>, %286 : !x86.avx512reg<zmm27>, %287 : !x86.avx512reg<zmm28>, %288 : !x86.avx512reg<zmm29>, %289 : !x86.avx512reg<zmm30>, %290 : !x86.avx512reg<zmm31>, %291 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb7(%292: !x86.reg64<rdi>, %293: !x86.reg64<rsi>, %294: !x86.reg64<rdx>, %295: !x86.reg64<rbp>, %296: !x86.reg64<rsp>, %297: !x86.reg64<r11>, %298: !x86.reg64<r10>, %299: !x86.avx512maskreg<k1>, %300: !x86.avx512reg<zmm26>, %301: !x86.avx512reg<zmm27>, %302: !x86.avx512reg<zmm28>, %303: !x86.avx512reg<zmm29>, %304: !x86.avx512reg<zmm30>, %305: !x86.avx512reg<zmm31>, %306: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l37"
// CHECK-NEXT:      %307 = x86.ri.add %306, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %308 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-NEXT:      %309 = x86.dss.vpxord %308, %308 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %310 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-NEXT:      %311 = x86.dss.vpxord %310, %310 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %312 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-NEXT:      %313 = x86.dss.vpxord %312, %312 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %314 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-NEXT:      %315 = x86.dss.vpxord %314, %314 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %316 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-NEXT:      %317 = x86.dss.vpxord %316, %316 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %318 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-NEXT:      %319 = x86.dss.vpxord %318, %318 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %320 = x86.dmk.vmovups[%292], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %321 = x86.dmk.vmovups[%292 + 280], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %322 = x86.rsm.vfmadd231ps %300, %320, [%293] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %323 = x86.rsm.vfmadd231ps %301, %320, [%293 + 512] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %324 = x86.rsm.vfmadd231ps %302, %320, [%293 + 1024] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %325 = x86.rsm.vfmadd231ps %303, %320, [%293 + 1536] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %326 = x86.rsm.vfmadd231ps %304, %320, [%293 + 2048] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %327 = x86.rsm.vfmadd231ps %305, %320, [%293 + 2560] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %328 = x86.dmk.vmovups[%292 + 560], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %329 = x86.rsm.vfmadd231ps %309, %321, [%293 + 4] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %330 = x86.rsm.vfmadd231ps %311, %321, [%293 + 516] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %331 = x86.rsm.vfmadd231ps %313, %321, [%293 + 1028] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %332 = x86.rsm.vfmadd231ps %315, %321, [%293 + 1540] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %333 = x86.rsm.vfmadd231ps %317, %321, [%293 + 2052] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %334 = x86.rsm.vfmadd231ps %319, %321, [%293 + 2564] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %335 = x86.dmk.vmovups[%292 + 840], %299 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %336 = x86.rsm.vfmadd231ps %322, %328, [%293 + 8] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %337 = x86.rsm.vfmadd231ps %323, %328, [%293 + 520] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %338 = x86.rsm.vfmadd231ps %324, %328, [%293 + 1032] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %339 = x86.rsm.vfmadd231ps %325, %328, [%293 + 1544] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %340 = x86.rsm.vfmadd231ps %326, %328, [%293 + 2056] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %341 = x86.rsm.vfmadd231ps %327, %328, [%293 + 2568] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %342 = x86.ri.add %292, 1120 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %343 = x86.rsm.vfmadd231ps %329, %335, [%293 + 12] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %344 = x86.rsm.vfmadd231ps %330, %335, [%293 + 524] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %345 = x86.rsm.vfmadd231ps %331, %335, [%293 + 1036] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %346 = x86.rsm.vfmadd231ps %332, %335, [%293 + 1548] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %347 = x86.rsm.vfmadd231ps %333, %335, [%293 + 2060] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %348 = x86.rsm.vfmadd231ps %334, %335, [%293 + 2572] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %349 = x86.ri.add %293, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %350 = x86.dss.vaddps %343, %336 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %351 = x86.dss.vaddps %344, %337 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %352 = x86.dss.vaddps %345, %338 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %353 = x86.dss.vaddps %346, %339 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %354 = x86.dss.vaddps %347, %340 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %355 = x86.dss.vaddps %348, %341 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %356 = x86.si.cmp %307, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %356 : !x86.rflags<rflags>, ^bb7(%342 : !x86.reg64<rdi>, %349 : !x86.reg64<rsi>, %294 : !x86.reg64<rdx>, %295 : !x86.reg64<rbp>, %296 : !x86.reg64<rsp>, %297 : !x86.reg64<r11>, %298 : !x86.reg64<r10>, %299 : !x86.avx512maskreg<k1>, %350 : !x86.avx512reg<zmm26>, %351 : !x86.avx512reg<zmm27>, %352 : !x86.avx512reg<zmm28>, %353 : !x86.avx512reg<zmm29>, %354 : !x86.avx512reg<zmm30>, %355 : !x86.avx512reg<zmm31>, %307 : !x86.reg64<r12>), ^bb8(%342 : !x86.reg64<rdi>, %349 : !x86.reg64<rsi>, %294 : !x86.reg64<rdx>, %295 : !x86.reg64<rbp>, %296 : !x86.reg64<rsp>, %297 : !x86.reg64<r11>, %298 : !x86.reg64<r10>, %299 : !x86.avx512maskreg<k1>, %350 : !x86.avx512reg<zmm26>, %351 : !x86.avx512reg<zmm27>, %352 : !x86.avx512reg<zmm28>, %353 : !x86.avx512reg<zmm29>, %354 : !x86.avx512reg<zmm30>, %355 : !x86.avx512reg<zmm31>, %307 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb8(%357: !x86.reg64<rdi>, %358: !x86.reg64<rsi>, %359: !x86.reg64<rdx>, %360: !x86.reg64<rbp>, %361: !x86.reg64<rsp>, %362: !x86.reg64<r11>, %363: !x86.reg64<r10>, %364: !x86.avx512maskreg<k1>, %365: !x86.avx512reg<zmm26>, %366: !x86.avx512reg<zmm27>, %367: !x86.avx512reg<zmm28>, %368: !x86.avx512reg<zmm29>, %369: !x86.avx512reg<zmm30>, %370: !x86.avx512reg<zmm31>, %371: !x86.reg64<r12>):
// CHECK-NEXT:      %372 = x86.ri.sub %358, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.msk.vmovups[%359], %365, %364 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%359 + 280], %366, %364 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%359 + 560], %367, %364 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%359 + 840], %368, %364 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%359 + 1120], %369, %364 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%359 + 1400], %370, %364 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %373 = x86.ri.add %359, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %374 = x86.ri.sub %357, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %375 = x86.si.cmp %363, 70 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %375 : !x86.rflags<rflags>, ^bb6(%374 : !x86.reg64<rdi>, %372 : !x86.reg64<rsi>, %373 : !x86.reg64<rdx>, %360 : !x86.reg64<rbp>, %361 : !x86.reg64<rsp>, %362 : !x86.reg64<r11>, %363 : !x86.reg64<r10>, %364 : !x86.avx512maskreg<k1>), ^bb9(%374 : !x86.reg64<rdi>, %372 : !x86.reg64<rsi>, %373 : !x86.reg64<rdx>, %360 : !x86.reg64<rbp>, %361 : !x86.reg64<rsp>, %362 : !x86.reg64<r11>, %363 : !x86.reg64<r10>, %364 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb9(%376: !x86.reg64<rdi>, %377: !x86.reg64<rsi>, %378: !x86.reg64<rdx>, %379: !x86.reg64<rbp>, %380: !x86.reg64<rsp>, %381: !x86.reg64<r11>, %382: !x86.reg64<r10>, %383: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      %384 = x86.ri.add %378, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %385 = x86.ri.add %377, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %386 = x86.ri.sub %376, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %387 = x86.si.cmp %381, 18 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %387 : !x86.rflags<rflags>, ^bb1(%386 : !x86.reg64<rdi>, %385 : !x86.reg64<rsi>, %384 : !x86.reg64<rdx>, %379 : !x86.reg64<rbp>, %380 : !x86.reg64<rsp>, %381 : !x86.reg64<r11>), ^bb10(%386 : !x86.reg64<rdi>, %385 : !x86.reg64<rsi>, %384 : !x86.reg64<rdx>, %379 : !x86.reg64<rbp>, %380 : !x86.reg64<rsp>, %381 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb10(%388: !x86.reg64<rdi>, %389: !x86.reg64<rsi>, %390: !x86.reg64<rdx>, %391: !x86.reg64<rbp>, %392: !x86.reg64<rsp>, %393: !x86.reg64<r11>):
// CHECK-NEXT:      %394 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:      x86.fallthrough ^bb11(%388 : !x86.reg64<rdi>, %389 : !x86.reg64<rsi>, %390 : !x86.reg64<rdx>, %391 : !x86.reg64<rbp>, %392 : !x86.reg64<rsp>, %394 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb11(%395: !x86.reg64<rdi>, %396: !x86.reg64<rsi>, %397: !x86.reg64<rdx>, %398: !x86.reg64<rbp>, %399: !x86.reg64<rsp>, %400: !x86.reg64<r11>):
// CHECK-NEXT:      x86.label "l38"
// CHECK-NEXT:      %401 = x86.ri.add %400, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %402 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb12(%395 : !x86.reg64<rdi>, %396 : !x86.reg64<rsi>, %397 : !x86.reg64<rdx>, %398 : !x86.reg64<rbp>, %399 : !x86.reg64<rsp>, %401 : !x86.reg64<r11>, %402 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb12(%403: !x86.reg64<rdi>, %404: !x86.reg64<rsi>, %405: !x86.reg64<rdx>, %406: !x86.reg64<rbp>, %407: !x86.reg64<rsp>, %408: !x86.reg64<r11>, %409: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l39"
// CHECK-NEXT:      %410 = x86.ri.add %409, 64 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %411 = x86.dm.vmovups [%405] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %412 = x86.dm.vmovups [%405 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %413 = x86.dm.vmovups [%405 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %414 = x86.dm.vmovups [%405 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %415 = x86.dm.vmovups [%405 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %416 = x86.dm.vmovups [%405 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %417 = x86.dm.vmovups [%405 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %418 = x86.dm.vmovups [%405 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %419 = x86.dm.vmovups [%405 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %420 = x86.dm.vmovups [%405 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %421 = x86.dm.vmovups [%405 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %422 = x86.dm.vmovups [%405 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %423 = x86.dm.vmovups [%405 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %424 = x86.dm.vmovups [%405 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %425 = x86.dm.vmovups [%405 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %426 = x86.dm.vmovups [%405 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %427 = x86.dm.vmovups [%405 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %428 = x86.dm.vmovups [%405 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %429 = x86.dm.vmovups [%405 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %430 = x86.dm.vmovups [%405 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %431 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb13(%403 : !x86.reg64<rdi>, %404 : !x86.reg64<rsi>, %405 : !x86.reg64<rdx>, %406 : !x86.reg64<rbp>, %407 : !x86.reg64<rsp>, %408 : !x86.reg64<r11>, %410 : !x86.reg64<r10>, %411 : !x86.avx512reg<zmm12>, %412 : !x86.avx512reg<zmm13>, %413 : !x86.avx512reg<zmm14>, %414 : !x86.avx512reg<zmm15>, %415 : !x86.avx512reg<zmm16>, %416 : !x86.avx512reg<zmm17>, %417 : !x86.avx512reg<zmm18>, %418 : !x86.avx512reg<zmm19>, %419 : !x86.avx512reg<zmm20>, %420 : !x86.avx512reg<zmm21>, %421 : !x86.avx512reg<zmm22>, %422 : !x86.avx512reg<zmm23>, %423 : !x86.avx512reg<zmm24>, %424 : !x86.avx512reg<zmm25>, %425 : !x86.avx512reg<zmm26>, %426 : !x86.avx512reg<zmm27>, %427 : !x86.avx512reg<zmm28>, %428 : !x86.avx512reg<zmm29>, %429 : !x86.avx512reg<zmm30>, %430 : !x86.avx512reg<zmm31>, %431 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb13(%432: !x86.reg64<rdi>, %433: !x86.reg64<rsi>, %434: !x86.reg64<rdx>, %435: !x86.reg64<rbp>, %436: !x86.reg64<rsp>, %437: !x86.reg64<r11>, %438: !x86.reg64<r10>, %439: !x86.avx512reg<zmm12>, %440: !x86.avx512reg<zmm13>, %441: !x86.avx512reg<zmm14>, %442: !x86.avx512reg<zmm15>, %443: !x86.avx512reg<zmm16>, %444: !x86.avx512reg<zmm17>, %445: !x86.avx512reg<zmm18>, %446: !x86.avx512reg<zmm19>, %447: !x86.avx512reg<zmm20>, %448: !x86.avx512reg<zmm21>, %449: !x86.avx512reg<zmm22>, %450: !x86.avx512reg<zmm23>, %451: !x86.avx512reg<zmm24>, %452: !x86.avx512reg<zmm25>, %453: !x86.avx512reg<zmm26>, %454: !x86.avx512reg<zmm27>, %455: !x86.avx512reg<zmm28>, %456: !x86.avx512reg<zmm29>, %457: !x86.avx512reg<zmm30>, %458: !x86.avx512reg<zmm31>, %459: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l40"
// CHECK-NEXT:      %460 = x86.ri.add %459, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %461 = x86.dm.vmovups [%432] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %462 = x86.dm.vmovups [%432 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %463 = x86.dm.vmovups [%432 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %464 = x86.dm.vmovups [%432 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %465 = x86.dm.vbroadcastss [%433] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %466 = x86.rss.vfmadd231ps %439, %461, %465 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %467 = x86.rss.vfmadd231ps %440, %462, %465 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %468 = x86.rss.vfmadd231ps %441, %463, %465 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %469 = x86.rss.vfmadd231ps %442, %464, %465 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %470 = x86.dm.vbroadcastss [%433 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %471 = x86.rss.vfmadd231ps %443, %461, %470 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %472 = x86.rss.vfmadd231ps %444, %462, %470 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %473 = x86.rss.vfmadd231ps %445, %463, %470 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %474 = x86.rss.vfmadd231ps %446, %464, %470 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %475 = x86.dm.vbroadcastss [%433 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %476 = x86.rss.vfmadd231ps %447, %461, %475 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %477 = x86.rss.vfmadd231ps %448, %462, %475 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %478 = x86.rss.vfmadd231ps %449, %463, %475 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %479 = x86.rss.vfmadd231ps %450, %464, %475 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %480 = x86.dm.vbroadcastss [%433 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %481 = x86.rss.vfmadd231ps %451, %461, %480 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %482 = x86.rss.vfmadd231ps %452, %462, %480 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %483 = x86.rss.vfmadd231ps %453, %463, %480 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %484 = x86.rss.vfmadd231ps %454, %464, %480 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %485 = x86.dm.vbroadcastss [%433 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %486 = x86.ri.add %433, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %487 = x86.ri.add %432, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %488 = x86.rss.vfmadd231ps %455, %461, %485 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %489 = x86.rss.vfmadd231ps %456, %462, %485 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %490 = x86.rss.vfmadd231ps %457, %463, %485 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %491 = x86.rss.vfmadd231ps %458, %464, %485 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %492 = x86.dm.vmovups [%487] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %493 = x86.dm.vmovups [%487 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %494 = x86.dm.vmovups [%487 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %495 = x86.dm.vmovups [%487 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %496 = x86.dm.vbroadcastss [%486] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %497 = x86.rss.vfmadd231ps %466, %492, %496 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %498 = x86.rss.vfmadd231ps %467, %493, %496 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %499 = x86.rss.vfmadd231ps %468, %494, %496 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %500 = x86.rss.vfmadd231ps %469, %495, %496 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %501 = x86.dm.vbroadcastss [%486 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %502 = x86.rss.vfmadd231ps %471, %492, %501 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %503 = x86.rss.vfmadd231ps %472, %493, %501 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %504 = x86.rss.vfmadd231ps %473, %494, %501 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %505 = x86.rss.vfmadd231ps %474, %495, %501 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %506 = x86.dm.vbroadcastss [%486 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %507 = x86.rss.vfmadd231ps %476, %492, %506 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %508 = x86.rss.vfmadd231ps %477, %493, %506 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %509 = x86.rss.vfmadd231ps %478, %494, %506 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %510 = x86.rss.vfmadd231ps %479, %495, %506 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %511 = x86.dm.vbroadcastss [%486 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %512 = x86.rss.vfmadd231ps %481, %492, %511 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %513 = x86.rss.vfmadd231ps %482, %493, %511 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %514 = x86.rss.vfmadd231ps %483, %494, %511 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %515 = x86.rss.vfmadd231ps %484, %495, %511 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %516 = x86.dm.vbroadcastss [%486 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %517 = x86.ri.add %486, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %518 = x86.ri.add %487, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %519 = x86.rss.vfmadd231ps %488, %492, %516 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %520 = x86.rss.vfmadd231ps %489, %493, %516 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %521 = x86.rss.vfmadd231ps %490, %494, %516 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %522 = x86.rss.vfmadd231ps %491, %495, %516 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %523 = x86.dm.vmovups [%518] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %524 = x86.dm.vmovups [%518 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %525 = x86.dm.vmovups [%518 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %526 = x86.dm.vmovups [%518 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %527 = x86.dm.vbroadcastss [%517] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %528 = x86.rss.vfmadd231ps %497, %523, %527 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %529 = x86.rss.vfmadd231ps %498, %524, %527 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %530 = x86.rss.vfmadd231ps %499, %525, %527 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %531 = x86.rss.vfmadd231ps %500, %526, %527 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %532 = x86.dm.vbroadcastss [%517 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %533 = x86.rss.vfmadd231ps %502, %523, %532 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %534 = x86.rss.vfmadd231ps %503, %524, %532 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %535 = x86.rss.vfmadd231ps %504, %525, %532 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %536 = x86.rss.vfmadd231ps %505, %526, %532 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %537 = x86.dm.vbroadcastss [%517 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %538 = x86.rss.vfmadd231ps %507, %523, %537 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %539 = x86.rss.vfmadd231ps %508, %524, %537 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %540 = x86.rss.vfmadd231ps %509, %525, %537 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %541 = x86.rss.vfmadd231ps %510, %526, %537 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %542 = x86.dm.vbroadcastss [%517 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %543 = x86.rss.vfmadd231ps %512, %523, %542 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %544 = x86.rss.vfmadd231ps %513, %524, %542 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %545 = x86.rss.vfmadd231ps %514, %525, %542 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %546 = x86.rss.vfmadd231ps %515, %526, %542 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %547 = x86.dm.vbroadcastss [%517 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %548 = x86.ri.add %517, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %549 = x86.ri.add %518, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %550 = x86.rss.vfmadd231ps %519, %523, %547 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %551 = x86.rss.vfmadd231ps %520, %524, %547 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %552 = x86.rss.vfmadd231ps %521, %525, %547 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %553 = x86.rss.vfmadd231ps %522, %526, %547 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %554 = x86.dm.vmovups [%549] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %555 = x86.dm.vmovups [%549 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %556 = x86.dm.vmovups [%549 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:      %557 = x86.dm.vmovups [%549 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %558 = x86.dm.vbroadcastss [%548] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %559 = x86.rss.vfmadd231ps %528, %554, %558 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %560 = x86.rss.vfmadd231ps %529, %555, %558 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %561 = x86.rss.vfmadd231ps %530, %556, %558 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %562 = x86.rss.vfmadd231ps %531, %557, %558 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %563 = x86.dm.vbroadcastss [%548 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %564 = x86.rss.vfmadd231ps %533, %554, %563 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %565 = x86.rss.vfmadd231ps %534, %555, %563 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %566 = x86.rss.vfmadd231ps %535, %556, %563 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %567 = x86.rss.vfmadd231ps %536, %557, %563 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %568 = x86.dm.vbroadcastss [%548 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %569 = x86.rss.vfmadd231ps %538, %554, %568 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %570 = x86.rss.vfmadd231ps %539, %555, %568 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %571 = x86.rss.vfmadd231ps %540, %556, %568 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %572 = x86.rss.vfmadd231ps %541, %557, %568 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %573 = x86.dm.vbroadcastss [%548 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %574 = x86.rss.vfmadd231ps %543, %554, %573 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %575 = x86.rss.vfmadd231ps %544, %555, %573 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %576 = x86.rss.vfmadd231ps %545, %556, %573 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %577 = x86.rss.vfmadd231ps %546, %557, %573 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %578 = x86.dm.vbroadcastss [%548 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %579 = x86.ri.add %548, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %580 = x86.ri.add %549, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %581 = x86.rss.vfmadd231ps %550, %554, %578 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %582 = x86.rss.vfmadd231ps %551, %555, %578 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %583 = x86.rss.vfmadd231ps %552, %556, %578 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %584 = x86.rss.vfmadd231ps %553, %557, %578 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %585 = x86.si.cmp %460, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %585 : !x86.rflags<rflags>, ^bb13(%580 : !x86.reg64<rdi>, %579 : !x86.reg64<rsi>, %434 : !x86.reg64<rdx>, %435 : !x86.reg64<rbp>, %436 : !x86.reg64<rsp>, %437 : !x86.reg64<r11>, %438 : !x86.reg64<r10>, %559 : !x86.avx512reg<zmm12>, %560 : !x86.avx512reg<zmm13>, %561 : !x86.avx512reg<zmm14>, %562 : !x86.avx512reg<zmm15>, %564 : !x86.avx512reg<zmm16>, %565 : !x86.avx512reg<zmm17>, %566 : !x86.avx512reg<zmm18>, %567 : !x86.avx512reg<zmm19>, %569 : !x86.avx512reg<zmm20>, %570 : !x86.avx512reg<zmm21>, %571 : !x86.avx512reg<zmm22>, %572 : !x86.avx512reg<zmm23>, %574 : !x86.avx512reg<zmm24>, %575 : !x86.avx512reg<zmm25>, %576 : !x86.avx512reg<zmm26>, %577 : !x86.avx512reg<zmm27>, %581 : !x86.avx512reg<zmm28>, %582 : !x86.avx512reg<zmm29>, %583 : !x86.avx512reg<zmm30>, %584 : !x86.avx512reg<zmm31>, %460 : !x86.reg64<r12>), ^bb14(%580 : !x86.reg64<rdi>, %579 : !x86.reg64<rsi>, %434 : !x86.reg64<rdx>, %435 : !x86.reg64<rbp>, %436 : !x86.reg64<rsp>, %437 : !x86.reg64<r11>, %438 : !x86.reg64<r10>, %559 : !x86.avx512reg<zmm12>, %560 : !x86.avx512reg<zmm13>, %561 : !x86.avx512reg<zmm14>, %562 : !x86.avx512reg<zmm15>, %564 : !x86.avx512reg<zmm16>, %565 : !x86.avx512reg<zmm17>, %566 : !x86.avx512reg<zmm18>, %567 : !x86.avx512reg<zmm19>, %569 : !x86.avx512reg<zmm20>, %570 : !x86.avx512reg<zmm21>, %571 : !x86.avx512reg<zmm22>, %572 : !x86.avx512reg<zmm23>, %574 : !x86.avx512reg<zmm24>, %575 : !x86.avx512reg<zmm25>, %576 : !x86.avx512reg<zmm26>, %577 : !x86.avx512reg<zmm27>, %581 : !x86.avx512reg<zmm28>, %582 : !x86.avx512reg<zmm29>, %583 : !x86.avx512reg<zmm30>, %584 : !x86.avx512reg<zmm31>, %460 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb14(%586: !x86.reg64<rdi>, %587: !x86.reg64<rsi>, %588: !x86.reg64<rdx>, %589: !x86.reg64<rbp>, %590: !x86.reg64<rsp>, %591: !x86.reg64<r11>, %592: !x86.reg64<r10>, %593: !x86.avx512reg<zmm12>, %594: !x86.avx512reg<zmm13>, %595: !x86.avx512reg<zmm14>, %596: !x86.avx512reg<zmm15>, %597: !x86.avx512reg<zmm16>, %598: !x86.avx512reg<zmm17>, %599: !x86.avx512reg<zmm18>, %600: !x86.avx512reg<zmm19>, %601: !x86.avx512reg<zmm20>, %602: !x86.avx512reg<zmm21>, %603: !x86.avx512reg<zmm22>, %604: !x86.avx512reg<zmm23>, %605: !x86.avx512reg<zmm24>, %606: !x86.avx512reg<zmm25>, %607: !x86.avx512reg<zmm26>, %608: !x86.avx512reg<zmm27>, %609: !x86.avx512reg<zmm28>, %610: !x86.avx512reg<zmm29>, %611: !x86.avx512reg<zmm30>, %612: !x86.avx512reg<zmm31>, %613: !x86.reg64<r12>):
// CHECK-NEXT:      %614 = x86.ri.sub %587, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%588], %593 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 64], %594 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 128], %595 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 192], %596 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 280], %597 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 344], %598 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 408], %599 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 472], %600 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 560], %601 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 624], %602 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 688], %603 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 752], %604 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 840], %605 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 904], %606 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 968], %607 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 1032], %608 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 1120], %609 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 1184], %610 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 1248], %611 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%588 + 1312], %612 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %615 = x86.ri.add %588, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %616 = x86.ri.sub %586, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %617 = x86.si.cmp %592, 64 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %617 : !x86.rflags<rflags>, ^bb12(%616 : !x86.reg64<rdi>, %614 : !x86.reg64<rsi>, %615 : !x86.reg64<rdx>, %589 : !x86.reg64<rbp>, %590 : !x86.reg64<rsp>, %591 : !x86.reg64<r11>, %592 : !x86.reg64<r10>), ^bb15(%616 : !x86.reg64<rdi>, %614 : !x86.reg64<rsi>, %615 : !x86.reg64<rdx>, %589 : !x86.reg64<rbp>, %590 : !x86.reg64<rsp>, %591 : !x86.reg64<r11>, %592 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb15(%618: !x86.reg64<rdi>, %619: !x86.reg64<rsi>, %620: !x86.reg64<rdx>, %621: !x86.reg64<rbp>, %622: !x86.reg64<rsp>, %623: !x86.reg64<r11>, %624: !x86.reg64<r10>):
// CHECK-NEXT:      %625 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %626 = x86.ks.kmovw %625 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %627 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb16(%618 : !x86.reg64<rdi>, %619 : !x86.reg64<rsi>, %620 : !x86.reg64<rdx>, %621 : !x86.reg64<rbp>, %622 : !x86.reg64<rsp>, %623 : !x86.reg64<r11>, %627 : !x86.reg64<r10>, %626 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb16(%628: !x86.reg64<rdi>, %629: !x86.reg64<rsi>, %630: !x86.reg64<rdx>, %631: !x86.reg64<rbp>, %632: !x86.reg64<rsp>, %633: !x86.reg64<r11>, %634: !x86.reg64<r10>, %635: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      x86.label "l41"
// CHECK-NEXT:      %636 = x86.ri.add %634, 6 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %637 = x86.dmk.vmovups[%630], %635 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %638 = x86.dmk.vmovups[%630 + 280], %635 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %639 = x86.dmk.vmovups[%630 + 560], %635 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %640 = x86.dmk.vmovups[%630 + 840], %635 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %641 = x86.dmk.vmovups[%630 + 1120], %635 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %642 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb17(%628 : !x86.reg64<rdi>, %629 : !x86.reg64<rsi>, %630 : !x86.reg64<rdx>, %631 : !x86.reg64<rbp>, %632 : !x86.reg64<rsp>, %633 : !x86.reg64<r11>, %636 : !x86.reg64<r10>, %635 : !x86.avx512maskreg<k1>, %637 : !x86.avx512reg<zmm27>, %638 : !x86.avx512reg<zmm28>, %639 : !x86.avx512reg<zmm29>, %640 : !x86.avx512reg<zmm30>, %641 : !x86.avx512reg<zmm31>, %642 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb17(%643: !x86.reg64<rdi>, %644: !x86.reg64<rsi>, %645: !x86.reg64<rdx>, %646: !x86.reg64<rbp>, %647: !x86.reg64<rsp>, %648: !x86.reg64<r11>, %649: !x86.reg64<r10>, %650: !x86.avx512maskreg<k1>, %651: !x86.avx512reg<zmm27>, %652: !x86.avx512reg<zmm28>, %653: !x86.avx512reg<zmm29>, %654: !x86.avx512reg<zmm30>, %655: !x86.avx512reg<zmm31>, %656: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l42"
// CHECK-NEXT:      %657 = x86.ri.add %656, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %658 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-NEXT:      %659 = x86.dss.vpxord %658, %658 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %660 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-NEXT:      %661 = x86.dss.vpxord %660, %660 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %662 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-NEXT:      %663 = x86.dss.vpxord %662, %662 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %664 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-NEXT:      %665 = x86.dss.vpxord %664, %664 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %666 = x86.get_avx_register : !x86.avx512reg<zmm26>
// CHECK-NEXT:      %667 = x86.dss.vpxord %666, %666 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %668 = x86.get_avx_register : !x86.avx512reg<zmm17>
// CHECK-NEXT:      %669 = x86.dss.vpxord %668, %668 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm17>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %670 = x86.get_avx_register : !x86.avx512reg<zmm18>
// CHECK-NEXT:      %671 = x86.dss.vpxord %670, %670 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm18>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %672 = x86.get_avx_register : !x86.avx512reg<zmm19>
// CHECK-NEXT:      %673 = x86.dss.vpxord %672, %672 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm19>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %674 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-NEXT:      %675 = x86.dss.vpxord %674, %674 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %676 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-NEXT:      %677 = x86.dss.vpxord %676, %676 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %678 = x86.get_avx_register : !x86.avx512reg<zmm12>
// CHECK-NEXT:      %679 = x86.dss.vpxord %678, %678 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm12>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %680 = x86.get_avx_register : !x86.avx512reg<zmm13>
// CHECK-NEXT:      %681 = x86.dss.vpxord %680, %680 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm13>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %682 = x86.get_avx_register : !x86.avx512reg<zmm14>
// CHECK-NEXT:      %683 = x86.dss.vpxord %682, %682 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm14>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %684 = x86.get_avx_register : !x86.avx512reg<zmm15>
// CHECK-NEXT:      %685 = x86.dss.vpxord %684, %684 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm15>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %686 = x86.get_avx_register : !x86.avx512reg<zmm16>
// CHECK-NEXT:      %687 = x86.dss.vpxord %686, %686 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm16>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %688 = x86.dmk.vmovups[%643], %650 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %689 = x86.dmk.vmovups[%643 + 280], %650 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %690 = x86.rsm.vfmadd231ps %651, %688, [%644] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %691 = x86.rsm.vfmadd231ps %652, %688, [%644 + 512] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %692 = x86.rsm.vfmadd231ps %653, %688, [%644 + 1024] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %693 = x86.rsm.vfmadd231ps %654, %688, [%644 + 1536] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %694 = x86.rsm.vfmadd231ps %655, %688, [%644 + 2048] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %695 = x86.dmk.vmovups[%643 + 560], %650 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %696 = x86.rsm.vfmadd231ps %659, %689, [%644 + 4] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %697 = x86.rsm.vfmadd231ps %661, %689, [%644 + 516] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %698 = x86.rsm.vfmadd231ps %663, %689, [%644 + 1028] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %699 = x86.rsm.vfmadd231ps %665, %689, [%644 + 1540] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %700 = x86.rsm.vfmadd231ps %667, %689, [%644 + 2052] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %701 = x86.dmk.vmovups[%643 + 840], %650 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %702 = x86.rsm.vfmadd231ps %669, %695, [%644 + 8] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %703 = x86.rsm.vfmadd231ps %671, %695, [%644 + 520] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %704 = x86.rsm.vfmadd231ps %673, %695, [%644 + 1032] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %705 = x86.rsm.vfmadd231ps %675, %695, [%644 + 1544] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %706 = x86.rsm.vfmadd231ps %677, %695, [%644 + 2056] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %707 = x86.ri.add %643, 1120 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %708 = x86.rsm.vfmadd231ps %679, %701, [%644 + 12] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %709 = x86.rsm.vfmadd231ps %681, %701, [%644 + 524] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %710 = x86.rsm.vfmadd231ps %683, %701, [%644 + 1036] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %711 = x86.rsm.vfmadd231ps %685, %701, [%644 + 1548] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %712 = x86.rsm.vfmadd231ps %687, %701, [%644 + 2060] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %713 = x86.ri.add %644, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %714 = x86.dss.vaddps %696, %690 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %715 = x86.dss.vaddps %697, %691 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %716 = x86.dss.vaddps %698, %692 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %717 = x86.dss.vaddps %699, %693 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %718 = x86.dss.vaddps %700, %694 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %719 = x86.dss.vaddps %702, %714 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %720 = x86.dss.vaddps %703, %715 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %721 = x86.dss.vaddps %704, %716 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %722 = x86.dss.vaddps %705, %717 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %723 = x86.dss.vaddps %706, %718 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %724 = x86.dss.vaddps %708, %719 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %725 = x86.dss.vaddps %709, %720 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %726 = x86.dss.vaddps %710, %721 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %727 = x86.dss.vaddps %711, %722 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %728 = x86.dss.vaddps %712, %723 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %729 = x86.si.cmp %657, 128 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %729 : !x86.rflags<rflags>, ^bb17(%707 : !x86.reg64<rdi>, %713 : !x86.reg64<rsi>, %645 : !x86.reg64<rdx>, %646 : !x86.reg64<rbp>, %647 : !x86.reg64<rsp>, %648 : !x86.reg64<r11>, %649 : !x86.reg64<r10>, %650 : !x86.avx512maskreg<k1>, %724 : !x86.avx512reg<zmm27>, %725 : !x86.avx512reg<zmm28>, %726 : !x86.avx512reg<zmm29>, %727 : !x86.avx512reg<zmm30>, %728 : !x86.avx512reg<zmm31>, %657 : !x86.reg64<r12>), ^bb18(%707 : !x86.reg64<rdi>, %713 : !x86.reg64<rsi>, %645 : !x86.reg64<rdx>, %646 : !x86.reg64<rbp>, %647 : !x86.reg64<rsp>, %648 : !x86.reg64<r11>, %649 : !x86.reg64<r10>, %650 : !x86.avx512maskreg<k1>, %724 : !x86.avx512reg<zmm27>, %725 : !x86.avx512reg<zmm28>, %726 : !x86.avx512reg<zmm29>, %727 : !x86.avx512reg<zmm30>, %728 : !x86.avx512reg<zmm31>, %657 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb18(%730: !x86.reg64<rdi>, %731: !x86.reg64<rsi>, %732: !x86.reg64<rdx>, %733: !x86.reg64<rbp>, %734: !x86.reg64<rsp>, %735: !x86.reg64<r11>, %736: !x86.reg64<r10>, %737: !x86.avx512maskreg<k1>, %738: !x86.avx512reg<zmm27>, %739: !x86.avx512reg<zmm28>, %740: !x86.avx512reg<zmm29>, %741: !x86.avx512reg<zmm30>, %742: !x86.avx512reg<zmm31>, %743: !x86.reg64<r12>):
// CHECK-NEXT:      %744 = x86.ri.sub %731, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.msk.vmovups[%732], %738, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%732 + 280], %739, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%732 + 560], %740, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%732 + 840], %741, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%732 + 1120], %742, %737 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %745 = x86.ri.add %732, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %746 = x86.ri.sub %730, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %747 = x86.si.cmp %736, 70 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %747 : !x86.rflags<rflags>, ^bb16(%746 : !x86.reg64<rdi>, %744 : !x86.reg64<rsi>, %745 : !x86.reg64<rdx>, %733 : !x86.reg64<rbp>, %734 : !x86.reg64<rsp>, %735 : !x86.reg64<r11>, %736 : !x86.reg64<r10>, %737 : !x86.avx512maskreg<k1>), ^bb19(%746 : !x86.reg64<rdi>, %744 : !x86.reg64<rsi>, %745 : !x86.reg64<rdx>, %733 : !x86.reg64<rbp>, %734 : !x86.reg64<rsp>, %735 : !x86.reg64<r11>, %736 : !x86.reg64<r10>, %737 : !x86.avx512maskreg<k1>)
// CHECK-NEXT:    ^bb19(%748: !x86.reg64<rdi>, %749: !x86.reg64<rsi>, %750: !x86.reg64<rdx>, %751: !x86.reg64<rbp>, %752: !x86.reg64<rsp>, %753: !x86.reg64<r11>, %754: !x86.reg64<r10>, %755: !x86.avx512maskreg<k1>):
// CHECK-NEXT:      %756 = x86.ri.add %750, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %757 = x86.ri.add %749, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %758 = x86.ri.sub %748, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %759 = x86.si.cmp %753, 38 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %759 : !x86.rflags<rflags>, ^bb11(%758 : !x86.reg64<rdi>, %757 : !x86.reg64<rsi>, %756 : !x86.reg64<rdx>, %751 : !x86.reg64<rbp>, %752 : !x86.reg64<rsp>, %753 : !x86.reg64<r11>), ^bb20(%758 : !x86.reg64<rdi>, %757 : !x86.reg64<rsi>, %756 : !x86.reg64<rdx>, %751 : !x86.reg64<rbp>, %752 : !x86.reg64<rsp>, %753 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb20(%760: !x86.reg64<rdi>, %761: !x86.reg64<rsi>, %762: !x86.reg64<rdx>, %763: !x86.reg64<rbp>, %764: !x86.reg64<rsp>, %765: !x86.reg64<r11>):
// CHECK-NEXT:      %766 = x86.ds.mov %763 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %767, %768 = x86.d.pop %766 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
