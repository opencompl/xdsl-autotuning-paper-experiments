// RUN: libxsmm-gemm dense %t matmul 70 1 2 70 2 70 1 1 1 1 cpx nopf BF16F32 && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// The flat BF16 kernel LIBXSMM emits for CPX: an unmasked block of 64 rows, then a
// remainder of 6 whose A loads take the hoisted k3 mask.

// CHECK:       matmul:
// CHECK:           and rsp, r10
// CHECK-NEXT:      push rbx
// CHECK-NEXT:      mov rbx, 63
// CHECK-NEXT:      kmovd k3, ebx
// CHECK-NEXT:      pop rbx
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 1
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 64
// CHECK-NEXT:      vpxord zmm28, zmm28, zmm28
// CHECK-NEXT:      vpxord zmm29, zmm29, zmm29
// CHECK-NEXT:      vpxord zmm30, zmm30, zmm30
// CHECK-NEXT:      vpxord zmm31, zmm31, zmm31
// CHECK-NEXT:      vmovdqu16 zmm0, [rdi]
// CHECK-NEXT:      vmovdqu16 zmm6, [rdi+140]
// CHECK-NEXT:      vmovdqu16 zmm5, [rdi+64]
// CHECK-NEXT:      vmovdqu16 zmm7, [rdi+204]
// CHECK-NEXT:      vpunpcklwd zmm1, zmm0, zmm6
// CHECK-NEXT:      vpunpckhwd zmm2, zmm0, zmm6
// CHECK-NEXT:      vpunpcklwd zmm3, zmm5, zmm7
// CHECK-NEXT:      vpunpckhwd zmm4, zmm5, zmm7
// CHECK-NEXT:      vpbroadcastd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vdpbf16ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vdpbf16ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vdpbf16ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vdpbf16ps zmm31, zmm4, zmm0
// CHECK-NEXT:      sub rsi, 4
// CHECK-NEXT:      vshufi64x2 zmm0, zmm28, zmm29, 68
// CHECK-NEXT:      vshufi64x2 zmm1, zmm28, zmm29, 238
// CHECK-NEXT:      vshufi64x2 zmm28, zmm0, zmm0, 216
// CHECK-NEXT:      vshufi64x2 zmm29, zmm1, zmm1, 216
// CHECK-NEXT:      vshufi64x2 zmm0, zmm30, zmm31, 68
// CHECK-NEXT:      vshufi64x2 zmm1, zmm30, zmm31, 238
// CHECK-NEXT:      vshufi64x2 zmm30, zmm0, zmm0, 216
// CHECK-NEXT:      vshufi64x2 zmm31, zmm1, zmm1, 216
// CHECK-NEXT:      vmovups zmm27, [rdx]
// CHECK-NEXT:      vaddps zmm28, zmm28, zmm27
// CHECK-NEXT:      vmovups zmm27, [rdx+64]
// CHECK-NEXT:      vaddps zmm29, zmm29, zmm27
// CHECK-NEXT:      vmovups zmm27, [rdx+128]
// CHECK-NEXT:      vaddps zmm30, zmm30, zmm27
// CHECK-NEXT:      vmovups zmm27, [rdx+192]
// CHECK-NEXT:      vaddps zmm31, zmm31, zmm27
// CHECK-NEXT:      vmovups [rdx], zmm28
// CHECK-NEXT:      vmovups [rdx+64], zmm29
// CHECK-NEXT:      vmovups [rdx+128], zmm30
// CHECK-NEXT:      vmovups [rdx+192], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 152
// CHECK-NEXT:      cmp r10, 64
// CHECK-NEXT:      jl [[M_BODY]]
// CHECK-NEXT:      mov r15, 63
// CHECK-NEXT:      kmovw k1, r15d
// CHECK-NEXT:      mov r10, 64
// CHECK-NEXT:  [[M_TAIL:^\S+]]:
// CHECK-NEXT:      add r10, 6
// CHECK-NEXT:      vpxord zmm31, zmm31, zmm31
// CHECK-NEXT:      vmovdqu16 ymm0 {k3}{z}, [rdi]
// CHECK-NEXT:      vmovdqu16 ymm6 {k3}{z}, [rdi+140]
// CHECK-NEXT:      vpunpcklwd zmm1, zmm0, zmm6
// CHECK-NEXT:      vpunpckhwd zmm2, zmm0, zmm6
// CHECK-NEXT:      vinserti32x8 zmm1, zmm1, ymm2, 1
// CHECK-NEXT:      vpbroadcastd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vdpbf16ps zmm31, zmm1, zmm0
// CHECK-NEXT:      sub rsi, 4
// CHECK-NEXT:      vshufi64x2 zmm31, zmm31, zmm31, 216
// CHECK-NEXT:      vmovups zmm30 {k1}{z}, [rdx]
// CHECK-NEXT:      vaddps zmm31, zmm31, zmm30
// CHECK-NEXT:      vmovups [rdx] {k1}, zmm31
// CHECK-NEXT:      add rdx, 24
// CHECK-NEXT:      sub rdi, 268
// CHECK-NEXT:      cmp r10, 70
// CHECK-NEXT:      jl [[M_TAIL]]
// CHECK-NEXT:      add rdx, 0
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      sub rdi, 140
// CHECK-NEXT:      cmp r11, 1
// CHECK-NEXT:      jl [[N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
