// RUN: xdsl-opt %s --verify-diagnostics -p 'xsmm-apply-schedule{strategy=libxsmm-spr-bf16-amx}' | filecheck %s

// AMX tiles read A in K pairs, so a flat A is rejected rather than reshuffled.

x86_func.func public @matmul(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
  %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 32 : i64, n = 32 : i64, k = 32 : i64, lda = 32 : i64, ldb = 32 : i64, ldc = 32 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", ab_datatype = bf16} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}

// CHECK: nano-kernel does not support the GEMM descriptor
