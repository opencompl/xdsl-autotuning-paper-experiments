// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{disable-loop-construction=true}' | filecheck %s

// CHECK-LABEL: x86_func.func @unmasked
// CHECK-NOT:     x86_scf.for
// CHECK-NOT:     x86.ks.kmov
// CHECK:         x86_func.ret
x86_func.func @unmasked(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 8 : i64, n = 1 : i64, k = 24 : i64, lda = 8 : i64, ldb = 24 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK-LABEL: x86_func.func @masked
// CHECK-NOT:     x86_scf.for
// CHECK:         %[[MASK_TMP:.*]] = x86.di.mov 127 : () -> !x86.reg64<r15>
// CHECK-NEXT:    %[[MASK:.*]] = x86.ks.kmovb %[[MASK_TMP]] : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK:         x86_func.ret
x86_func.func @masked(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 7 : i64, n = 1 : i64, k = 1 : i64, lda = 7 : i64, ldb = 1 : i64, ldc = 7 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}
