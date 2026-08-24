// RUN: xdsl-opt %s --split-input-file -p xsmm-tile-k | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @below_threshold(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 23 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @below_threshold(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 23 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @exact_tiles(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %1, %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = x86_scf.for %2 : !x86.reg64<r12>  = %0 to 24 : si32 step 4 : si32 iter_args(%3 = %a, %4 = %b, %5 = %c, %6 = %rbp, %7 = %rsp, %8 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %9, %10, %11, %12, %13, %14 = "xsmm.matmul_k"(%3, %4, %5, %6, %7, %8) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        x86_scf.yield %9, %10, %11, %12, %13, %14 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @exact_tiles(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 24 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @tiles_and_remainder(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6, %7 = x86_scf.for %8 : !x86.reg64<r12>  = %0 to 24 : si32 step 4 : si32 iter_args(%9 = %a, %10 = %b, %11 = %c, %12 = %rbp, %13 = %rsp, %14 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %15, %16, %17, %18, %19, %20 = "xsmm.matmul_k"(%9, %10, %11, %12, %13, %14) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        x86_scf.yield %15, %16, %17, %18, %19, %20 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%2, %3, %4, %5, %6, %7) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 1 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @tiles_and_remainder(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 25 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}
