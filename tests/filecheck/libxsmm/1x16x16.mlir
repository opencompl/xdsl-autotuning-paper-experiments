// RUN: libxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  l33:
// CHECK-NEXT:      add r11, 1
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  l34:
// CHECK-NEXT:      add r10, 16
// CHECK-NEXT:      vmovapd zmm30, [rdx]
// CHECK-NEXT:      vmovapd zmm31, [rdx+64]
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd [rdx], zmm30
// CHECK-NEXT:      vmovapd [rdx+64], zmm31
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      sub rdi, 1920
// CHECK-NEXT:      cmp r10, 16
// CHECK-NEXT:      jl l34
// CHECK-NEXT:      add rdx, 0
// CHECK-NEXT:      add rsi, 128
// CHECK-NEXT:      sub rdi, 128
// CHECK-NEXT:      cmp r11, 1
// CHECK-NEXT:      jl l33
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
