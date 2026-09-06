// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 70 : i64, n = 38 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdx+344]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdx+408]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdx+472]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx+624]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+688]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+752]
// CHECK-REGALLOC-NEXT:      vmovups zmm11, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+904]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+968]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+1032]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+1184]
// CHECK-REGALLOC-NEXT:      vmovups zmm5, [rdx+1248]
// CHECK-REGALLOC-NEXT:      vmovups zmm6, [rdx+1312]
// CHECK-REGALLOC-NEXT:      vmovups zmm7, [rdx+1400]
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx+1464]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+1528]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+1592]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280], zmm4
// CHECK-REGALLOC-NEXT:      vmovups [rdx+344], zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+408], zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+472], zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560], zmm0
// CHECK-REGALLOC-NEXT:      vmovups [rdx+624], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+688], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+752], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840], zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+904], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+968], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1032], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1184], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1248], zmm5
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1312], zmm6
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1400], zmm7
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1464], zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1528], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1592], zmm10
// CHECK-REGALLOC-NEXT:      sub rdi, 35584
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      mov rcx, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      vmovups zmm10 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm9 {k1}{z}, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm8 {k1}{z}, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm7 {k1}{z}, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm6 {k1}{z}, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm5 {k1}{z}, [rdx+1400]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-REGALLOC-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-REGALLOC-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-REGALLOC-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-REGALLOC-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-REGALLOC-NEXT:      vpxord zmm11, zmm11, zmm11
// CHECK-REGALLOC-NEXT:      vmovups zmm18 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm17 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm18, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm18, [rsi+512]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm18, [rsi+1024]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm18, [rsi+1536]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm18, [rsi+2048]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm18, [rsi+2560]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm18 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm17, [rsi+4]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm17, [rsi+516]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm17, [rsi+1028]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm17, [rsi+1540]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm17, [rsi+2052]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm17, [rsi+2564]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm17 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm18, [rsi+8]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm18, [rsi+520]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm18, [rsi+1032]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm18, [rsi+1544]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm18, [rsi+2056]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm18, [rsi+2568]{1to16}
// CHECK-REGALLOC-NEXT:      add rdi, 1120
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm17, [rsi+12]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm17, [rsi+524]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm17, [rsi+1036]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm17, [rsi+1548]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm17, [rsi+2060]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm17, [rsi+2572]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 16
// CHECK-REGALLOC-NEXT:      vaddps zmm10, zmm16, zmm10
// CHECK-REGALLOC-NEXT:      vaddps zmm9, zmm15, zmm9
// CHECK-REGALLOC-NEXT:      vaddps zmm8, zmm14, zmm8
// CHECK-REGALLOC-NEXT:      vaddps zmm7, zmm13, zmm7
// CHECK-REGALLOC-NEXT:      vaddps zmm6, zmm12, zmm6
// CHECK-REGALLOC-NEXT:      vaddps zmm5, zmm11, zmm5
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280] {k1}, zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560] {k1}, zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840] {k1}, zmm7
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120] {k1}, zmm6
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1400] {k1}, zmm5
// CHECK-REGALLOC-NEXT:      sub rdi, 35816
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      add rdx, 1400
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      vmovups zmm5, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm6, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm7, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+344]
// CHECK-REGALLOC-NEXT:      vmovups zmm11, [rdx+408]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+472]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+624]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+688]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+752]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+904]
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx+968]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdx+1032]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdx+1184]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdx+1248]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx+1312]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm5
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm6
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm7
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+344], zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+408], zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+472], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+624], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+688], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+752], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+904], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+968], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1032], zmm0
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120], zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1184], zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1248], zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1312], zmm4
// CHECK-REGALLOC-NEXT:      sub rdi, 35584
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      mov rcx, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      vmovups zmm4 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm3 {k1}{z}, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm2 {k1}{z}, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdx+1120]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-REGALLOC-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-REGALLOC-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-REGALLOC-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-REGALLOC-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-REGALLOC-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-REGALLOC-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-REGALLOC-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-REGALLOC-NEXT:      vpxord zmm11, zmm11, zmm11
// CHECK-REGALLOC-NEXT:      vpxord zmm10, zmm10, zmm10
// CHECK-REGALLOC-NEXT:      vpxord zmm9, zmm9, zmm9
// CHECK-REGALLOC-NEXT:      vpxord zmm8, zmm8, zmm8
// CHECK-REGALLOC-NEXT:      vpxord zmm7, zmm7, zmm7
// CHECK-REGALLOC-NEXT:      vpxord zmm6, zmm6, zmm6
// CHECK-REGALLOC-NEXT:      vpxord zmm5, zmm5, zmm5
// CHECK-REGALLOC-NEXT:      vmovups zmm21 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm20 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm21, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm21, [rsi+512]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm21, [rsi+1024]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm21, [rsi+1536]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm21, [rsi+2048]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm21 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm20, [rsi+4]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm20, [rsi+516]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm20, [rsi+1028]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm20, [rsi+1540]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm20, [rsi+2052]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm20 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm21, [rsi+8]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm21, [rsi+520]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm21, [rsi+1032]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm21, [rsi+1544]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm21, [rsi+2056]{1to16}
// CHECK-REGALLOC-NEXT:      add rdi, 1120
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm20, [rsi+12]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm20, [rsi+524]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm20, [rsi+1036]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm20, [rsi+1548]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm20, [rsi+2060]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 16
// CHECK-REGALLOC-NEXT:      vaddps zmm4, zmm19, zmm4
// CHECK-REGALLOC-NEXT:      vaddps zmm3, zmm18, zmm3
// CHECK-REGALLOC-NEXT:      vaddps zmm2, zmm17, zmm2
// CHECK-REGALLOC-NEXT:      vaddps zmm1, zmm16, zmm1
// CHECK-REGALLOC-NEXT:      vaddps zmm0, zmm15, zmm0
// CHECK-REGALLOC-NEXT:      vaddps zmm4, zmm14, zmm4
// CHECK-REGALLOC-NEXT:      vaddps zmm3, zmm13, zmm3
// CHECK-REGALLOC-NEXT:      vaddps zmm2, zmm12, zmm2
// CHECK-REGALLOC-NEXT:      vaddps zmm1, zmm11, zmm1
// CHECK-REGALLOC-NEXT:      vaddps zmm0, zmm10, zmm0
// CHECK-REGALLOC-NEXT:      vaddps zmm4, zmm9, zmm4
// CHECK-REGALLOC-NEXT:      vaddps zmm3, zmm8, zmm3
// CHECK-REGALLOC-NEXT:      vaddps zmm2, zmm7, zmm2
// CHECK-REGALLOC-NEXT:      vaddps zmm1, zmm6, zmm1
// CHECK-REGALLOC-NEXT:      vaddps zmm0, zmm5, zmm0
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm4
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280] {k1}, zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560] {k1}, zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840] {k1}, zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120] {k1}, zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 35816
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      add rdx, 1120
// CHECK-REGALLOC-NEXT:      cmp rax, 20
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      ret
