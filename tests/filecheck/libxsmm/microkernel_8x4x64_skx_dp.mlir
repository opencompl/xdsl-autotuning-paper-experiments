// Microkernel-only generation: a single register tile (load C, K-loop, store C) with
// NO outer M/N tiling loops. Contrast with the full-kernel tests in this directory,
// which emit a nested n-loop / m-loop / k-loop. The distinguishing property is that the
// only loop here is the K-loop, so there is exactly one backward branch.
// RUN: libxsmm-gemm --microkernel dense %t matmul 8 4 64 8 64 8 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .globl matmul
// CHECK:       matmul:
// The single loop is the K-loop over K=64 (unrolled by 4). Its footer is the only
// backward branch; there must be no further loop (no m-loop / n-loop footer after it).
// CHECK:           cmp r12, 64
// CHECK-NEXT:      jl l{{[0-9]+}}
// CHECK-NOT:       jl l
