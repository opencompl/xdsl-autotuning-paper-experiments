// RUN: xdsl-opt %s -p 'xsmm-tile-n-m{strategy=zen5-kdot-greedy}' | filecheck %s

x86_func.func @kdot_greedy(
  %a: !x86.reg64,
  %b: !x86.reg64,
  %c: !x86.reg64,
  %rbp: !x86.reg64,
  %rsp: !x86.reg64
) {
  %0, %1, %2, %3, %4 = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 1 : i64, n_start = 0 : i64, n = 25 : i64, k = 64 : i64, lda = 1 : i64, ldb = 64 : i64, ldc = 1 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @kdot_greedy
// CHECK: x86_scf.for {{.*}} to 24 : si32 step 8 : si32
// CHECK: "xsmm.matmul"{{.*}}m = 1 : i64{{.*}}n_start = 0 : i64{{.*}}n = 8 : i64
// CHECK: x86_scf.for {{.*}} = {{.*}} to 25 : si32 step 1 : si32
// CHECK: "xsmm.matmul"{{.*}}m = 1 : i64{{.*}}n_start = 24 : i64{{.*}}n = 1 : i64
