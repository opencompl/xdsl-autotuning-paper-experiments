// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'convert-xsmm-to-x86{strategy=libxsmm-skx-fsdbcst}' | filecheck %s

x86_func.func @multiple_m_vectors_are_not_fsdbcst(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %mask: !x86.avx512maskreg<k1>,
  %acc0: !x86.avx512reg<zmm30>,
  %acc1: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc0_out, %acc1_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %mask, %acc0, %acc1) <{m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, datatype = f32, aligned_a = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// CHECK: unsupported SKX fsdbcst nano-kernel tile

// -----

x86_func.func @wrong_accumulator_count(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 2 : i64, k = 2 : i64, lda = 8 : i64, ldb = 16 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// CHECK: SKX matmul_reg expected 2 accumulator outs, got 1

// -----

x86_func.func @missing_mask(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 7 : i64, n = 1 : i64, k = 2 : i64, lda = 7 : i64, ldb = 16 : i64, datatype = f64, aligned_a = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// CHECK: SKX matmul_reg expects one mask in exactly when M has a partial vector

// -----

x86_func.func @extraneous_mask(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %mask: !x86.avx512maskreg<k1>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %mask, %acc) <{m = 8 : i64, n = 1 : i64, k = 2 : i64, lda = 8 : i64, ldb = 16 : i64, datatype = f64, aligned_a = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// CHECK: SKX matmul_reg expects one mask in exactly when M has a partial vector
