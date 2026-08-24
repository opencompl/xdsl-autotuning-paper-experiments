// RUN: xdsl-opt %s --split-input-file -p 'xsmm-tile-n' | filecheck %s

x86_func.func @first_range_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n_start = 0 : i64, n_blocking = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @first_range_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 14 : si32 step 14 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_n"(%8, %9, %10, %11, %12) <{m = 16 : i64, n_start = 0 : i64, n_blocking = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @second_range_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n_start = 14 : i64, n_blocking = 52 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @second_range_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 14 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 66 : si32 step 13 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_n"(%8, %9, %10, %11, %12) <{m = 16 : i64, n_start = 14 : i64, n_blocking = 13 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @first_range_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 70 : i64, n_start = 0 : i64, n_blocking = 18 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @first_range_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 18 : si32 step 6 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_n"(%8, %9, %10, %11, %12) <{m = 70 : i64, n_start = 0 : i64, n_blocking = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @second_range_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 70 : i64, n_start = 18 : i64, n_blocking = 20 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @second_range_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 38 : si32 step 5 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_n"(%8, %9, %10, %11, %12) <{m = 70 : i64, n_start = 18 : i64, n_blocking = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
