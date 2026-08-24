// RUN: xdsl-opt %s -p 'xsmm-tile-n{disable-regalloc=true}' | filecheck %s

x86_func.func @unallocated(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n_start = 14 : i64, n_blocking = 52 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @unallocated(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
// CHECK-NEXT:      %0 = x86.di.mov 14 : () -> !x86.reg64
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64  = %0 to 66 : si32 step 13 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_n"(%8, %9, %10, %11, %12) <{m = 16 : i64, n_start = 14 : i64, n_blocking = 13 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
