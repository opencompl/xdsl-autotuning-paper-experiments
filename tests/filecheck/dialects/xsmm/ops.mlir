// RUN: xdsl-opt %s --split-input-file | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %rbp = "test.op"() : () -> !x86.reg64<rbp>
// CHECK-NEXT:    %rsp = "test.op"() : () -> !x86.reg64<rsp>
// CHECK-NEXT:    %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 17 : i64, n_blocking = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %rbp = "test.op"() : () -> !x86.reg64<rbp>
  %rsp = "test.op"() : () -> !x86.reg64<rsp>
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 17 : i64, n_blocking = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %rbp = "test.op"() : () -> !x86.reg64<rbp>
// CHECK-NEXT:    %rsp = "test.op"() : () -> !x86.reg64<rsp>
// CHECK-NEXT:    %acc0 = "test.op"() : () -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %acc1 = "test.op"() : () -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc0_out, %acc1_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc0, %acc1) <{m_blocking = 8 : i64, n_blocking = 2 : i64, k_blocking = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %rbp = "test.op"() : () -> !x86.reg64<rbp>
  %rsp = "test.op"() : () -> !x86.reg64<rsp>
  %acc0 = "test.op"() : () -> !x86.avx512reg<zmm30>
  %acc1 = "test.op"() : () -> !x86.avx512reg<zmm31>
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc0_out, %acc1_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc0, %acc1) <{m_blocking = 8 : i64, n_blocking = 2 : i64, k_blocking = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %rbp = "test.op"() : () -> !x86.reg64<rbp>
// CHECK-NEXT:    %rsp = "test.op"() : () -> !x86.reg64<rsp>
// CHECK-NEXT:    %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 8 : i64, n_blocking = 2 : i64, k = 5 : i64, lda = 8 : i64, ldb = 5 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %rbp = "test.op"() : () -> !x86.reg64<rbp>
  %rsp = "test.op"() : () -> !x86.reg64<rsp>
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 8 : i64, n_blocking = 2 : i64, k = 5 : i64, lda = 8 : i64, ldb = 5 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %rbp = "test.op"() : () -> !x86.reg64<rbp>
// CHECK-NEXT:    %rsp = "test.op"() : () -> !x86.reg64<rsp>
// CHECK-NEXT:    %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %mask_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp, %mask) <{m_blocking = 17 : i64, n_blocking = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %rbp = "test.op"() : () -> !x86.reg64<rbp>
  %rsp = "test.op"() : () -> !x86.reg64<rsp>
  %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %mask_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp, %mask) <{m_blocking = 17 : i64, n_blocking = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %rbp = "test.op"() : () -> !x86.reg64<rbp>
// CHECK-NEXT:    %rsp = "test.op"() : () -> !x86.reg64<rsp>
// CHECK-NEXT:    %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %acc0 = "test.op"() : () -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %acc1 = "test.op"() : () -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %acc2 = "test.op"() : () -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %acc3 = "test.op"() : () -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %mask_out, %acc0_out, %acc1_out, %acc2_out, %acc3_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %mask, %acc0, %acc1, %acc2, %acc3) <{m_blocking = 17 : i64, n_blocking = 2 : i64, k_blocking = 3 : i64, lda = 17 : i64, ldb = 24 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 4>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 4>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %rbp = "test.op"() : () -> !x86.reg64<rbp>
  %rsp = "test.op"() : () -> !x86.reg64<rsp>
  %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
  %acc0 = "test.op"() : () -> !x86.avx512reg<zmm28>
  %acc1 = "test.op"() : () -> !x86.avx512reg<zmm29>
  %acc2 = "test.op"() : () -> !x86.avx512reg<zmm30>
  %acc3 = "test.op"() : () -> !x86.avx512reg<zmm31>
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %mask_out, %acc0_out, %acc1_out, %acc2_out, %acc3_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %mask, %acc0, %acc1, %acc2, %acc3) <{m_blocking = 17 : i64, n_blocking = 2 : i64, k_blocking = 3 : i64, lda = 17 : i64, ldb = 24 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 4>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 4>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
}
