// RUN: xdsl-opt %s -p 'convert-xsmm-to-x86{nano-kernel=libxsmm-skx-nofsdbcst}' | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @matmul_k_nofsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %mask: !x86.avx512maskreg<k1>, %acc0: !x86.avx512reg<zmm30>, %acc1: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.dm.vmovups [%a] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %1 = x86.dmk.vmovups[%a + 64], %mask {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %2 = x86.dm.vbroadcastss [%b] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %3 = x86.ri.add %b, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %4 = x86.ri.add %a, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %5 = x86.rss.vfmadd231ps %acc0, %0, %2 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %6 = x86.rss.vfmadd231ps %acc1, %1, %2 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %7 = x86.dm.vmovups [%4] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %8 = x86.dmk.vmovups[%4 + 64], %mask {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %9 = x86.dm.vbroadcastss [%3] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %b_out = x86.ri.add %3, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %a_out = x86.ri.add %4, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %acc0_out = x86.rss.vfmadd231ps %5, %7, %9 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %acc1_out = x86.rss.vfmadd231ps %6, %8, %9 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @matmul_k_nofsdbcst(
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
