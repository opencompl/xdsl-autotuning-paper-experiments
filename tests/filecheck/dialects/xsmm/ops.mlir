// RUN: xdsl-opt %s --split-input-file | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %a_out, %b_out, %c_out = xsmm.matmul %a, %b, %c {m = 17 : i64, n = 3 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 17 : i64, n = 3 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %a_out, %b_out, %c_out = xsmm.matmul %a, %b, %c {m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %a_out, %b_out, %c_out = xsmm.matmul %a, %b, %c {m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %acc0 = "test.op"() : () -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %acc1 = "test.op"() : () -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %a_out, %b_out, %acc0_out, %acc1_out = xsmm.matmul_reg %a, %b {m = 8 : i64, n = 2 : i64, k = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true} outs(%acc0, %acc1 : !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) : (!x86.reg64<rdi>, !x86.reg64<rsi>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %acc0 = "test.op"() : () -> !x86.avx512reg<zmm30>
  %acc1 = "test.op"() : () -> !x86.avx512reg<zmm31>
  %a_out, %b_out, %acc0_out, %acc1_out = xsmm.matmul_reg %a, %b {m = 8 : i64, n = 2 : i64, k = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true} outs(%acc0, %acc1 : !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) : (!x86.reg64<rdi>, !x86.reg64<rsi>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %a_out, %b_out, %c_out = xsmm.matmul %a, %b, %c {m = 8 : i64, n = 2 : i64, k = 5 : i64, lda = 8 : i64, ldb = 5 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 8 : i64, n = 2 : i64, k = 5 : i64, lda = 8 : i64, ldb = 5 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %carried = "test.op"() : () -> !x86.reg64<rax>
// CHECK-NEXT:    %a_out, %b_out, %c_out, %carried_out = xsmm.matmul %a, %b, %c {m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m"} ins(%mask : !x86.avx512maskreg<k1>) outs(%carried : !x86.reg64<rax>) : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
  %carried = "test.op"() : () -> !x86.reg64<rax>
  %a_out, %b_out, %c_out, %carried_out = xsmm.matmul %a, %b, %c {m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m"} ins(%mask : !x86.avx512maskreg<k1>) outs(%carried : !x86.reg64<rax>) : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %a = "test.op"() : () -> !x86.reg64<rdi>
// CHECK-NEXT:    %b = "test.op"() : () -> !x86.reg64<rsi>
// CHECK-NEXT:    %c = "test.op"() : () -> !x86.reg64<rdx>
// CHECK-NEXT:    %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %acc0 = "test.op"() : () -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %acc1 = "test.op"() : () -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %acc2 = "test.op"() : () -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %acc3 = "test.op"() : () -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %a_out, %b_out, %acc0_out, %acc1_out, %acc2_out, %acc3_out = xsmm.matmul_reg %a, %b {m = 17 : i64, n = 2 : i64, k = 3 : i64, lda = 17 : i64, ldb = 24 : i64, datatype = f32, aligned_a = false} ins(%mask : !x86.avx512maskreg<k1>) outs(%acc0, %acc1, %acc2, %acc3 : !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) : (!x86.reg64<rdi>, !x86.reg64<rsi>)
// CHECK-NEXT:  }

builtin.module {
  %a = "test.op"() : () -> !x86.reg64<rdi>
  %b = "test.op"() : () -> !x86.reg64<rsi>
  %c = "test.op"() : () -> !x86.reg64<rdx>
  %mask = "test.op"() : () -> !x86.avx512maskreg<k1>
  %acc0 = "test.op"() : () -> !x86.avx512reg<zmm28>
  %acc1 = "test.op"() : () -> !x86.avx512reg<zmm29>
  %acc2 = "test.op"() : () -> !x86.avx512reg<zmm30>
  %acc3 = "test.op"() : () -> !x86.avx512reg<zmm31>
  %a_out, %b_out, %acc0_out, %acc1_out, %acc2_out, %acc3_out = xsmm.matmul_reg %a, %b {m = 17 : i64, n = 2 : i64, k = 3 : i64, lda = 17 : i64, ldb = 24 : i64, datatype = f32, aligned_a = false} ins(%mask : !x86.avx512maskreg<k1>) outs(%acc0, %acc1, %acc2, %acc3 : !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) : (!x86.reg64<rdi>, !x86.reg64<rsi>)
}
