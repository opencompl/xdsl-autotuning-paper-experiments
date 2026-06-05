// RUN: libxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

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
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r15, 127
// CHECK-NEXT:      kmovb k1, r15d
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 7
// CHECK-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+56]
// CHECK-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+112]
// CHECK-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+168]
// CHECK-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+224]
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm29, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 56
// CHECK-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-NEXT:      vmovupd [rdx+56] {k1}, zmm28
// CHECK-NEXT:      vmovupd [rdx+112] {k1}, zmm29
// CHECK-NEXT:      vmovupd [rdx+168] {k1}, zmm30
// CHECK-NEXT:      vmovupd [rdx+224] {k1}, zmm31
// CHECK-NEXT:      add rdx, 56
// CHECK-NEXT:      sub rdi, 840
// CHECK-NEXT:      cmp r10, 7
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      add rdx, 224
// CHECK-NEXT:      add rsi, 640
// CHECK-NEXT:      sub rdi, 56
// CHECK-NEXT:      cmp r11, 5
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
