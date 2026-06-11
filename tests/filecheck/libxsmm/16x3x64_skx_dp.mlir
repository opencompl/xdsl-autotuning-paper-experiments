// RUN: libxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r12
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  l33:
// CHECK-NEXT:      add r11, 3
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  l34:
// CHECK-NEXT:      add r10, 16
// CHECK-NEXT:      vmovapd zmm26, [rdx]
// CHECK-NEXT:      vmovapd zmm27, [rdx+64]
// CHECK-NEXT:      vmovapd zmm28, [rdx+128]
// CHECK-NEXT:      vmovapd zmm29, [rdx+192]
// CHECK-NEXT:      vmovapd zmm30, [rdx+256]
// CHECK-NEXT:      vmovapd zmm31, [rdx+320]
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  l35:
// CHECK-NEXT:      add r12, 4
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      cmp r12, 64
// CHECK-NEXT:      jl l35
// CHECK-NEXT:      sub rsi, 512
// CHECK-NEXT:      vmovapd [rdx], zmm26
// CHECK-NEXT:      vmovapd [rdx+64], zmm27
// CHECK-NEXT:      vmovapd [rdx+128], zmm28
// CHECK-NEXT:      vmovapd [rdx+192], zmm29
// CHECK-NEXT:      vmovapd [rdx+256], zmm30
// CHECK-NEXT:      vmovapd [rdx+320], zmm31
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      sub rdi, 8064
// CHECK-NEXT:      cmp r10, 16
// CHECK-NEXT:      jl l34
// CHECK-NEXT:      add rdx, 256
// CHECK-NEXT:      add rsi, 1536
// CHECK-NEXT:      sub rdi, 128
// CHECK-NEXT:      cmp r11, 3
// CHECK-NEXT:      jl l33
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r12
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
