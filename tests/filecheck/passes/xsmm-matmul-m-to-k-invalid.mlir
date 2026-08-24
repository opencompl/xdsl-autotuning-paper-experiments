// RUN: not xdsl-opt %s -p xsmm-matmul-m-to-k 2>&1 | filecheck %s

x86_func.func @missing_mask(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul_m"(%a, %b, %c, %rbp, %rsp) <{m_blocking = 7 : i64, n_blocking = 1 : i64, k = 4 : i64, lda = 7 : i64, ldb = 8 : i64, ldc = 7 : i64, datatype = f64, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK: Exception: xsmm-matmul-m-to-k requires a mask for a partial M vector; run xsmm-tile-m first
