// RUN: compxsmm-gemm dense %t matmul 70 1 2 70 2 70 1 1 1 1 cpx nopf BF16F32_VNNI_A && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul 70 1 2 70 2 70 1 1 1 1 cpx nopf BF16F32_VNNI_A --disable-regalloc && xdsl-opt %t -f mlir -p 'xsmm-apply-schedule{strategy=libxsmm-cpx-bf16-vnni disable-regalloc=true},x86-regalloc-verify-liveness,x86-allocate-registers{arch=avx512},convert-x86-scf-to-x86,x86-prologue-epilogue-insertion' -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// BF16 A packed in K pairs accumulates into F32 C with no interleave: a block of
// 64 rows broadcasts the B pair into a register, and the remainder of 6, one
// vector, takes it as a broadcast memory operand under C's k1 mask.

// CHECK:       x86_func.func public @matmul(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:  %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 70 : i64, n = 1 : i64, k = 2 : i64, lda = 70 : i64, ldb = 2 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", ab_datatype = bf16, vnni_a} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul
// CHECK-REGALLOC-NEXT:  matmul:
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm7, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm6, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm5, [rdi+192]
// CHECK-REGALLOC-NEXT:      vpbroadcastd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm3, zmm4, zmm8
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm2, zmm4, zmm7
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm1, zmm4, zmm6
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm0, zmm4, zmm5
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 24
// CHECK-REGALLOC-NEXT:      sub rsi, 4
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      mov rax, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, eax
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vdpbf16ps zmm0, zmm1, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 256
// CHECK-REGALLOC-NEXT:      sub rsi, 4
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdx, 0
// CHECK-REGALLOC-NEXT:      ret
