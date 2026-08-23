// RUN: xdsl-opt %s --split-input-file -p 'xsmm-tile-m' | filecheck %s

x86_func.func @single_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 16 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 16 : i64, ldb = 5 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @single_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r10>  = %0 to 16 : si32 step 16 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_m"(%8, %9, %10, %11, %12) <{m_blocking = 16 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 16 : i64, ldb = 5 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @multiple_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 64 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 64 : i64, ldb = 5 : i64, ldc = 64 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @multiple_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r10>  = %0 to 64 : si32 step 32 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_m"(%8, %9, %10, %11, %12) <{m_blocking = 32 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 64 : i64, ldb = 5 : i64, ldc = 64 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @vector_remainder_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 40 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 40 : i64, ldb = 5 : i64, ldc = 40 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @vector_remainder_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r10>  = %0 to 32 : si32 step 32 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_m"(%8, %9, %10, %11, %12) <{m_blocking = 32 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 40 : i64, ldb = 5 : i64, ldc = 40 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %18 = x86.di.mov 32 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %19, %20, %21, %22, %23, %24 = x86_scf.for %25 : !x86.reg64<r10>  = %18 to 40 : si32 step 8 : si32 iter_args(%26 = %2, %27 = %3, %28 = %4, %29 = %5, %30 = %6) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %31, %32, %33, %34, %35 = "xsmm.matmul_m"(%26, %27, %28, %29, %30) <{m_blocking = 8 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 40 : i64, ldb = 5 : i64, ldc = 40 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %31, %32, %33, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @masked_remainder_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 34 : i64, n_blocking = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @masked_remainder_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r10>  = %0 to 32 : si32 step 32 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_m"(%8, %9, %10, %11, %12) <{m_blocking = 32 : i64, n_blocking = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %18 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %19 = x86.ks.kmovb %18 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %20 = x86.di.mov 32 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %21, %22, %23, %24, %25, %26, %27 = x86_scf.for %28 : !x86.reg64<r10>  = %20 to 34 : si32 step 2 : si32 iter_args(%29 = %2, %30 = %3, %31 = %4, %32 = %5, %33 = %6, %34 = %19) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %35, %36, %37, %38, %39, %40 = "xsmm.matmul_m"(%29, %30, %31, %32, %33, %34) <{m_blocking = 2 : i64, n_blocking = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>)
// CHECK-NEXT:        x86_scf.yield %35, %36, %37, %38, %39, %40 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @partial_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 50 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 50 : i64, ldb = 5 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @partial_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %1 = x86.ks.kmovw %0 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %2 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %3, %4, %5, %6, %7, %8, %9 = x86_scf.for %10 : !x86.reg64<r10>  = %2 to 50 : si32 step 50 : si32 iter_args(%11 = %a, %12 = %b, %13 = %c, %14 = %rbp, %15 = %rsp, %16 = %1) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %17, %18, %19, %20, %21, %22 = "xsmm.matmul_m"(%11, %12, %13, %14, %15, %16) <{m_blocking = 50 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 50 : i64, ldb = 5 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>)
// CHECK-NEXT:        x86_scf.yield %17, %18, %19, %20, %21, %22 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
