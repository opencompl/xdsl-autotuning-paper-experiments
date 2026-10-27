// RUN: libxsmm-gemm dense %t matmul 70 1 2 70 2 70 1 1 1 1 cpx nopf BF16F32_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// The BF16 kernel LIBXSMM emits for CPX when A is packed in K pairs: no
// interleave, so A loads as F32 vectors and C stays in M order. A block of 64
// rows broadcasts the B pair into a register; the remainder of 6, one vector,
// takes it as a broadcast memory operand, and C's k1 mask covers A's pairs too.

// CHECK:       matmul:
// CHECK:           and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 1
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 64
// CHECK-NEXT:      vmovups zmm28, [rdx]
// CHECK-NEXT:      vmovups zmm29, [rdx+64]
// CHECK-NEXT:      vmovups zmm30, [rdx+128]
// CHECK-NEXT:      vmovups zmm31, [rdx+192]
// CHECK-NEXT:      vmovups zmm1, [rdi]
// CHECK-NEXT:      vmovups zmm2, [rdi+64]
// CHECK-NEXT:      vmovups zmm3, [rdi+128]
// CHECK-NEXT:      vmovups zmm4, [rdi+192]
// CHECK-NEXT:      vpbroadcastd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vdpbf16ps zmm28, zmm1, zmm0
// CHECK-NEXT:      vdpbf16ps zmm29, zmm2, zmm0
// CHECK-NEXT:      vdpbf16ps zmm30, zmm3, zmm0
// CHECK-NEXT:      vdpbf16ps zmm31, zmm4, zmm0
// CHECK-NEXT:      sub rsi, 4
// CHECK-NEXT:      vmovups [rdx], zmm28
// CHECK-NEXT:      vmovups [rdx+64], zmm29
// CHECK-NEXT:      vmovups [rdx+128], zmm30
// CHECK-NEXT:      vmovups [rdx+192], zmm31
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      sub rdi, 24
// CHECK-NEXT:      cmp r10, 64
// CHECK-NEXT:      jl [[M_BODY]]
// CHECK-NEXT:      mov r15, 63
// CHECK-NEXT:      kmovw k1, r15d
// CHECK-NEXT:      mov r10, 64
// CHECK-NEXT:  [[M_TAIL:^\S+]]:
// CHECK-NEXT:      add r10, 6
// CHECK-NEXT:      vmovups zmm31 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovups zmm0 {k1}{z}, [rdi]
// CHECK-NEXT:      add rdi, 280
// CHECK-NEXT:      vdpbf16ps zmm31, zmm0, [rsi]{1to16}
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      sub rsi, 4
// CHECK-NEXT:      vmovups [rdx] {k1}, zmm31
// CHECK-NEXT:      add rdx, 24
// CHECK-NEXT:      sub rdi, 256
// CHECK-NEXT:      cmp r10, 70
// CHECK-NEXT:      jl [[M_TAIL]]
// CHECK-NEXT:      add rdx, 0
// CHECK-NEXT:      add rsi, 4
// CHECK-NEXT:      sub rdi, 280
// CHECK-NEXT:      cmp r11, 1
// CHECK-NEXT:      jl [[N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
