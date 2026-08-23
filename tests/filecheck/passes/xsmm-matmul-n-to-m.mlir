// RUN: xdsl-opt %s --split-input-file -p xsmm-matmul-n-to-m | filecheck %s

x86_func.func @single_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n_start = 0 : i64, n_blocking = 1 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @single_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 16 : i64, n_blocking = 1 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:      %5 = x86.ri.add %2, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %6 = x86.ri.add %1, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %7 = x86.ri.sub %0, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @blocked_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n_start = 4 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 16 : i64, ldb = 5 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @blocked_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 16 : i64, n_blocking = 3 : i64, k = 5 : i64, lda = 16 : i64, ldb = 5 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:      %5 = x86.ri.add %2, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %6 = x86.ri.add %1, 120 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %7 = x86.ri.sub %0, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @unallocated_f32(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 50 : i64, n_start = 18 : i64, n_blocking = 6 : i64, k = 128 : i64, lda = 50 : i64, ldb = 128 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @unallocated_f32(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
// CHECK-NEXT:      %0, %1, %2, %3, %4 = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 50 : i64, n_blocking = 6 : i64, k = 128 : i64, lda = 50 : i64, ldb = 128 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:      %5 = x86.ri.add %2, 1000 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %6 = x86.ri.add %1, 3072 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      %7 = x86.ri.sub %0, 200 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
