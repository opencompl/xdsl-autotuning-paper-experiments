// RUN: xdsl-opt %s --split-input-file -p xsmm-tile-k | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @below_threshold(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 23 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
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
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 23 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @exact_tiles(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %1, %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = x86_scf.for %2 : !x86.reg64<r12>  = %0 to 24 : si32 step 4 : si32 iter_args(%3 = %a, %4 = %b, %5 = %rbp, %6 = %rsp, %7 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %8, %9, %10, %11, %12 = "xsmm.matmul_reg"(%3, %4, %5, %6, %7) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        x86_scf.yield %8, %9, %10, %11, %12 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>
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
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 24 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @tiles_and_remainder(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r12>  = %0 to 24 : si32 step 4 : si32 iter_args(%8 = %a, %9 = %b, %10 = %rbp, %11 = %rsp, %12 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %13, %14, %15, %16, %17 = "xsmm.matmul_reg"(%8, %9, %10, %11, %12) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        x86_scf.yield %13, %14, %15, %16, %17 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%2, %3, %4, %5, %6) <{m = 8 : i64, n = 1 : i64, k = 1 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
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
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 25 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @read_only_mask(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %mask: !x86.avx512maskreg<k1>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %1, %2, %3, %rbp_out, %rsp_out, %acc_out = x86_scf.for %4 : !x86.reg64<r12>  = %0 to 24 : si32 step 4 : si32 iter_args(%5 = %a, %6 = %b, %7 = %rbp, %8 = %rsp, %9 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %10, %11, %12, %13, %14 = "xsmm.matmul_reg"(%5, %6, %7, %8, %mask, %9) <{m = 7 : i64, n = 1 : i64, k = 4 : i64, lda = 7 : i64, ldb = 32 : i64, datatype = f64, aligned_a = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        x86_scf.yield %10, %11, %12, %13, %14 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %b_out = x86.ri.sub %3, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %a_out = x86.ri.sub %2, 1288 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @read_only_mask(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %mask: !x86.avx512maskreg<k1>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %mask, %acc) <{m = 7 : i64, n = 1 : i64, k = 24 : i64, lda = 7 : i64, ldb = 32 : i64, datatype = f64, aligned_a = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}
