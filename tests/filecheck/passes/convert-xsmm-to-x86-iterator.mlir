// RUN: xdsl-opt %s --split-input-file -p 'convert-xsmm-to-x86{strategy=libxsmm-skx-fsdbcst}' | filecheck %s

// CHECK-LABEL: x86_func.func @iterator_m
// CHECK:         %[[A_K:.*]] = x86.ri.add %a, 128
// CHECK:         %[[B_K:.*]] = x86.ri.add %b, 16
// CHECK:         %b_out = x86.ri.sub %[[B_K]], 16
// CHECK-NEXT:    %a_out = x86.ri.sub %[[A_K]], 64

x86_func.func @iterator_m(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 2 : i64, lda = 8 : i64, ldb = 16 : i64, datatype = f64, aligned_a = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK-LABEL: x86_func.func @iterator_n
// CHECK:         %[[A_K:.*]] = x86.ri.add %a, 128
// CHECK:         %[[B_K:.*]] = x86.ri.add %b, 16
// CHECK:         %b_out = x86.ri.add %[[B_K]], 112
// CHECK-NEXT:    %a_out = x86.ri.sub %[[A_K]], 128

x86_func.func @iterator_n(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 2 : i64, lda = 8 : i64, ldb = 16 : i64, datatype = f64, aligned_a = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK-LABEL: x86_func.func @iterator_none
// CHECK:         %[[A_K:.*]] = x86.ri.add %a, 128
// CHECK:         %[[B_K:.*]] = x86.ri.add %b, 16
// CHECK:         %b_out = x86.ri.sub %[[B_K]], 16
// CHECK-NEXT:    %a_out = x86.ri.sub %[[A_K]], 128

x86_func.func @iterator_none(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %acc) <{m = 8 : i64, n = 1 : i64, k = 2 : i64, lda = 8 : i64, ldb = 16 : i64, datatype = f64, aligned_a = true, iterator = "none", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm31>)
  x86_func.ret
}
