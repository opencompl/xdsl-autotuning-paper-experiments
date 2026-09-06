// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%a, %b, %c = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
%a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 0 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 8 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)

// CHECK: Operation does not verify: m must be positive, got 0

// -----

%a, %b, %c, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 0 : i64, lda = 8 : i64, ldb = 8 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: k must be positive, got 0

// -----

%a, %b, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 8 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512maskreg<k1>)

// CHECK: Operation does not verify: operand and result types must match pairwise

// -----

%a, %b, %c, %acc = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.avx512reg<zmm31>)
%a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 7 : i64, ldb = 8 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)

// CHECK: Operation does not verify: aligned A requires lda to be a multiple of the vector length

// -----

%a, %b, %c = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
%a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 8 : i64, n = 1 : i64, k = 0 : i64, lda = 8 : i64, ldb = 8 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)

// CHECK: Operation does not verify: k must be positive, got 0

// -----

%a, %b, %c = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
%a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 8 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "x", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)

// CHECK: Operation does not verify: iterator must be one of none, m, n, k, got x

// -----

%a, %b, %c = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
%a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 8 : i64, ldc = 7 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)

// CHECK: Operation does not verify: aligned C requires ldc to be a multiple of the vector length
