// RUN: xdsl-opt %s --split-input-file -p 'xsmm-tile-m{disable-regalloc=true}' | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @unallocated(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64  = %0 to 32 : si32 step 32 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_m"(%8, %9, %10, %11, %12) <{m_blocking = 32 : i64, n_blocking = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      %18 = x86.di.mov 3 : () -> !x86.reg64
// CHECK-NEXT:      %19 = x86.ks.kmovb %18 : (!x86.reg64) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %20 = x86.di.mov 32 : () -> !x86.reg64
// CHECK-NEXT:      %21, %22, %23, %24, %25, %26, %27 = x86_scf.for %28 : !x86.reg64  = %20 to 34 : si32 step 2 : si32 iter_args(%29 = %2, %30 = %3, %31 = %4, %32 = %5, %33 = %6, %34 = %19) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %35, %36, %37, %38, %39, %40 = "xsmm.matmul_m"(%29, %30, %31, %32, %33, %34) <{m_blocking = 2 : i64, n_blocking = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>)
// CHECK-NEXT:        x86_scf.yield %35, %36, %37, %38, %39, %40 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @unallocated(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 34 : i64, n_blocking = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
  x86_func.ret
}
