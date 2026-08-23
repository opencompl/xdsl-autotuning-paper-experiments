// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%a, %b, %c, %rbp, %rsp, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 0 : i64, lda = 8 : i64, ldb = 8 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: k_blocking must be positive, got 0

// -----

%a, %b, %c, %rbp, %rsp, %acc0, %acc1 = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc0_out, %acc1_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc0, %acc1) <{m_blocking = 16 : i64, n_blocking = 2 : i64, k_blocking = 4 : i64, lda = 16 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: expected 4 accumulators for 16x2 blocking, got 2

// -----

%a, %b, %c, %rbp, %rsp, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 7 : i64, n_blocking = 1 : i64, k_blocking = 4 : i64, lda = 7 : i64, ldb = 8 : i64, datatype = f64, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: expected a mask exactly when m_blocking is not a multiple of the vector length

// -----

%a, %b, %c, %rbp, %rsp, %mask, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out, %mask_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %mask, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 4 : i64, lda = 8 : i64, ldb = 8 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: expected a mask exactly when m_blocking is not a multiple of the vector length

// -----

%a, %b, %c, %rbp, %rsp, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k_blocking = 4 : i64, lda = 7 : i64, ldb = 8 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: aligned A requires lda to be a multiple of the vector length

// -----

%a, %b, %c, %rbp, %rsp = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k = 0 : i64, lda = 8 : i64, ldb = 8 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)

// CHECK: Operation does not verify: k must be positive, got 0

// -----

%a, %b, %c, %rbp, %rsp = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 7 : i64, n_blocking = 1 : i64, k = 4 : i64, lda = 7 : i64, ldb = 8 : i64, ldc = 7 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)

// CHECK: Operation does not verify: expected a mask exactly when m_blocking is not a multiple of the vector length

// -----

%a, %b, %c, %rbp, %rsp = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
%a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 8 : i64, n_blocking = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 8 : i64, ldc = 7 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)

// CHECK: Operation does not verify: aligned C requires ldc to be a multiple of the vector length
