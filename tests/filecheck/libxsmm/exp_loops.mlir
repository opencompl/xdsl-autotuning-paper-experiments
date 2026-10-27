// RUN: libxsmm-exp %t exp_kernel 512 --degree 3 && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-exp %t exp_kernel 1024 --degree 3 && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-exp %t exp_kernel 512 --degree 5 && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-exp %t exp_kernel 1024 --degree 5 && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// LIBXSMM's loop header is MOV / label / ADD: entering the first iteration
// must not execute an extra branch. The conditional backedge must remain.
// CHECK:      mov r10, 0
// CHECK-NEXT: [[LOOP:[a-zA-Z_][a-zA-Z_0-9]*]]:
// CHECK-NEXT: add r10, 16
// CHECK:      cmp r10,
// CHECK-NEXT: jl [[LOOP]]
// CHECK-NEXT: sub r8,
// CHECK-NEXT: sub r9,
// CHECK-NEXT: ret
