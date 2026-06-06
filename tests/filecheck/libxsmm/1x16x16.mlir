// RUN: libxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p x86-allocate-registers,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov [[TMP:\S+]], -64
// CHECK-NEXT:      and rsp, [[TMP]]
// CHECK-NEXT:      mov [[N:\S+]], 0
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add [[N]], 1
// CHECK-NEXT:      mov [[M:\S+]], 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add [[M]], 16
// CHECK-NEXT:      vmovapd [[VEC_ACC0:\S+]], [rdx]
// CHECK-NEXT:      vmovapd [[VEC_ACC1:\S+]], [rdx+64]
// CHECK-NEXT:      vmovapd [[VEC_A0:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A1:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B0:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A0]], [[VEC_B0]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A1]], [[VEC_B0]]
// CHECK-NEXT:      vmovapd [[VEC_A2:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A3:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B1:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A2]], [[VEC_B1]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A3]], [[VEC_B1]]
// CHECK-NEXT:      vmovapd [[VEC_A4:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A5:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B2:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A4]], [[VEC_B2]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A5]], [[VEC_B2]]
// CHECK-NEXT:      vmovapd [[VEC_A6:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A7:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B3:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A6]], [[VEC_B3]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A7]], [[VEC_B3]]
// CHECK-NEXT:      vmovapd [[VEC_A8:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A9:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B4:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A8]], [[VEC_B4]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A9]], [[VEC_B4]]
// CHECK-NEXT:      vmovapd [[VEC_A10:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A11:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B5:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A10]], [[VEC_B5]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A11]], [[VEC_B5]]
// CHECK-NEXT:      vmovapd [[VEC_A12:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A13:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B6:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A12]], [[VEC_B6]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A13]], [[VEC_B6]]
// CHECK-NEXT:      vmovapd [[VEC_A14:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A15:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B7:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A14]], [[VEC_B7]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A15]], [[VEC_B7]]
// CHECK-NEXT:      vmovapd [[VEC_A16:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A17:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B8:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A16]], [[VEC_B8]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A17]], [[VEC_B8]]
// CHECK-NEXT:      vmovapd [[VEC_A18:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A19:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B9:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A18]], [[VEC_B9]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A19]], [[VEC_B9]]
// CHECK-NEXT:      vmovapd [[VEC_A20:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A21:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B10:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A20]], [[VEC_B10]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A21]], [[VEC_B10]]
// CHECK-NEXT:      vmovapd [[VEC_A22:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A23:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B11:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A22]], [[VEC_B11]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A23]], [[VEC_B11]]
// CHECK-NEXT:      vmovapd [[VEC_A24:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A25:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B12:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A24]], [[VEC_B12]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A25]], [[VEC_B12]]
// CHECK-NEXT:      vmovapd [[VEC_A26:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A27:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B13:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A26]], [[VEC_B13]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A27]], [[VEC_B13]]
// CHECK-NEXT:      vmovapd [[VEC_A28:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A29:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B14:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A28]], [[VEC_B14]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A29]], [[VEC_B14]]
// CHECK-NEXT:      vmovapd [[VEC_A30:\S+]], [rdi]
// CHECK-NEXT:      vmovapd [[VEC_A31:\S+]], [rdi+64]
// CHECK-NEXT:      vbroadcastsd [[VEC_B15:\S+]], [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_A30]], [[VEC_B15]]
// CHECK-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_A31]], [[VEC_B15]]
// CHECK-NEXT:      vmovapd [rdx], [[VEC_ACC0]]
// CHECK-NEXT:      vmovapd [rdx+64], [[VEC_ACC1]]
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      sub rdi, 1920
// CHECK-NEXT:      cmp [[M]], 16
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      add rdx, 0
// CHECK-NEXT:      add rsi, 128
// CHECK-NEXT:      sub rdi, 128
// CHECK-NEXT:      cmp [[N]], 1
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
