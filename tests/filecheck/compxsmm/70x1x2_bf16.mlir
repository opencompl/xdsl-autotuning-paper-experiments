// RUN: compxsmm-gemm dense %t matmul 70 1 2 70 2 70 1 1 1 1 cpx nopf BF16F32 && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul 70 1 2 70 2 70 1 1 1 1 cpx nopf BF16F32 --disable-regalloc && xdsl-opt %t -f mlir -p 'xsmm-apply-schedule{strategy=libxsmm-cpx-bf16 disable-regalloc=true},x86-regalloc-verify-liveness,x86-allocate-registers{arch=avx512},convert-x86-scf-to-x86,x86-prologue-epilogue-insertion' -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// BF16 A and B accumulate into F32 C: a block of 64 rows whose word-interleaved
// accumulators are un-shuffled before C is added, then a remainder of 6 rows
// masked on C (k1, F32 lanes) and on A (k3, BF16 lanes).

// CHECK:       x86_func.func public @matmul(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:  %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 70 : i64, n = 1 : i64, k = 2 : i64, lda = 70 : i64, ldb = 2 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", ab_datatype = bf16} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul
// CHECK-REGALLOC-NEXT:  matmul:
// CHECK-REGALLOC-NEXT:      vpxord zmm3, zmm3, zmm3
// CHECK-REGALLOC-NEXT:      vpxord zmm1, zmm1, zmm1
// CHECK-REGALLOC-NEXT:      vpxord zmm0, zmm0, zmm0
// CHECK-REGALLOC-NEXT:      vpxord zmm4, zmm4, zmm4
// CHECK-REGALLOC-NEXT:      vmovdqu16 zmm7, [rdi]
// CHECK-REGALLOC-NEXT:      vmovdqu16 zmm6, [rdi+140]
// CHECK-REGALLOC-NEXT:      vmovdqu16 zmm5, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovdqu16 zmm2, [rdi+204]
// CHECK-REGALLOC-NEXT:      vpunpcklwd zmm8, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vpunpckhwd zmm7, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vpunpcklwd zmm6, zmm5, zmm2
// CHECK-REGALLOC-NEXT:      vpunpckhwd zmm5, zmm5, zmm2
// CHECK-REGALLOC-NEXT:      vpbroadcastd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm3, zmm2, zmm8
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm1, zmm2, zmm7
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm0, zmm2, zmm6
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm4, zmm2, zmm5
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm2, zmm3, zmm1, 68
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm3, zmm3, zmm1, 238
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm2, zmm2, zmm2, 216
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm3, zmm3, zmm3, 216
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm1, zmm0, zmm4, 68
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm0, zmm0, zmm4, 238
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm1, zmm1, zmm1, 216
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm0, zmm0, zmm0, 216
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx]
// CHECK-REGALLOC-NEXT:      vaddps zmm2, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx+64]
// CHECK-REGALLOC-NEXT:      vaddps zmm3, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx+128]
// CHECK-REGALLOC-NEXT:      vaddps zmm1, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx+192]
// CHECK-REGALLOC-NEXT:      vaddps zmm0, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 152
// CHECK-REGALLOC-NEXT:      sub rsi, 4
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      mov rax, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, eax
// CHECK-REGALLOC-NEXT:      mov rax, 63
// CHECK-REGALLOC-NEXT:      kmovd k3, eax
// CHECK-REGALLOC-NEXT:      vpxord zmm0, zmm0, zmm0
// CHECK-REGALLOC-NEXT:      vmovdqu16 zmm1 {k3}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovdqu16 zmm3 {k3}{z}, [rdi+140]
// CHECK-REGALLOC-NEXT:      vpunpcklwd zmm2, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vpunpckhwd zmm1, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm2, zmm2, zmm1, 68
// CHECK-REGALLOC-NEXT:      vpbroadcastd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm0, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vshufi64x2 zmm0, zmm0, zmm0, 216
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vaddps zmm0, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 268
// CHECK-REGALLOC-NEXT:      sub rsi, 4
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 140
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdx, 0
// CHECK-REGALLOC-NEXT:      ret
