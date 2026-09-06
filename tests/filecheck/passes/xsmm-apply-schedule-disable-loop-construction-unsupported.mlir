// RUN: xdsl-opt %s --verify-diagnostics -p 'xsmm-apply-schedule{disable-loop-construction=true}' | filecheck %s

// CHECK: unsupported SKX nano-kernel tile
x86_func.func @unsupported(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 40 : i64, n = 1 : i64, k = 1 : i64, lda = 40 : i64, ldb = 1 : i64, ldc = 40 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}
