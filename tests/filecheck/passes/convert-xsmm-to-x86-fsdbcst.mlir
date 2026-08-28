// RUN: xdsl-opt %s -p 'convert-xsmm-to-x86{nano-kernel=libxsmm-skx-fsdbcst}' | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @matmul_k_fsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %acc0: !x86.avx512reg<zmm30>, %acc1: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %0 = x86.get_avx_register : !x86.avx512reg<zmm28>
// CHECK-NEXT:      %1 = x86.dss.vpxord %0, %0 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %2 = x86.get_avx_register : !x86.avx512reg<zmm29>
// CHECK-NEXT:      %3 = x86.dss.vpxord %2, %2 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %4 = x86.dm.vmovapd [%a] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %5 = x86.dm.vmovapd [%a + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %6 = x86.rsm.vfmadd231pd %acc0, %4, [%b] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %7 = x86.rsm.vfmadd231pd %acc1, %4, [%b + 128] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %a_out = x86.ri.add %a, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %8 = x86.rsm.vfmadd231pd %1, %5, [%b + 8] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %9 = x86.rsm.vfmadd231pd %3, %5, [%b + 136] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %b_out = x86.ri.add %b, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %acc0_out = x86.dss.vaddpd %8, %6 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %acc1_out = x86.dss.vaddpd %9, %7 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @matmul_k_fsdbcst(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %acc0: !x86.avx512reg<zmm30>,
  %acc1: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out, %acc0_out, %acc1_out = "xsmm.matmul_k"(%a, %b, %c, %rbp, %rsp, %acc0, %acc1) <{m_blocking = 8 : i64, n_blocking = 2 : i64, k_blocking = 2 : i64, lda = 8 : i64, ldb = 16 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
  x86_func.ret
}
