// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'convert-xsmm-to-x86{nano-kernel=skx-fsdbcst}' | filecheck %s

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
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %mask_out, %acc0_out, %acc1_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %mask, %acc0, %acc1) <{m_blocking = 17 : i64, n_blocking = 1 : i64, k_blocking = 2 : i64, lda = 17 : i64, ldb = 16 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// CHECK: unsupported SKX fsdbcst nano-kernel tile
