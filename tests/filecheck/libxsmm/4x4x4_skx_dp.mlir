// RUN: libxsmm-gemm dense %t matmul_bac 4 4 4 4 4 4 1 1 0 0 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  l33:
// CHECK-NEXT:      add r11, 4
// CHECK-NEXT:      mov r15, 15
// CHECK-NEXT:      kmovb k1, r15d
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  l34:
// CHECK-NEXT:      add r10, 4
// CHECK-NEXT:      vmovupd zmm28 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+32]
// CHECK-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+64]
// CHECK-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+96]
// CHECK-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-NEXT:      vpxord zmm27, zmm27, zmm27
// CHECK-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi]
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+32]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+32]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+64]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+96]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+8]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+40]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+72]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+104]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+96]
// CHECK-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+16]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+48]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+80]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+112]{1to8}
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+24]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+56]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+88]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+120]{1to8}
// CHECK-NEXT:      add rsi, 32
// CHECK-NEXT:      vaddpd zmm28, zmm24, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm25, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm26, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm27, zmm31
// CHECK-NEXT:      vaddpd zmm28, zmm20, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm21, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm22, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm23, zmm31
// CHECK-NEXT:      vaddpd zmm28, zmm16, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm17, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm18, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm19, zmm31
// CHECK-NEXT:      vmovupd [rdx] {k1}, zmm28
// CHECK-NEXT:      vmovupd [rdx+32] {k1}, zmm29
// CHECK-NEXT:      vmovupd [rdx+64] {k1}, zmm30
// CHECK-NEXT:      vmovupd [rdx+96] {k1}, zmm31
// CHECK-NEXT:      add rdx, 32
// CHECK-NEXT:      sub rdi, 96
// CHECK-NEXT:      cmp r10, 4
// CHECK-NEXT:      jl l34
// CHECK-NEXT:      add rdx, 96
// CHECK-NEXT:      add rsi, 128
// CHECK-NEXT:      sub rdi, 32
// CHECK-NEXT:      cmp r11, 4
// CHECK-NEXT:      jl l33
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
