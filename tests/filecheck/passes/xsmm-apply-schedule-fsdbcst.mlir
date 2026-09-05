// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-fsdbcst}' | filecheck %s
// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-fsdbcst disable-regalloc=true}' | filecheck %s --check-prefix AUTO

// AUTO-LABEL:   x86_func.func @fsdbcst
// AUTO-NOT:     !x86.avx512reg<zmm
// AUTO:         x86_func.ret

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @fsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.dm.vmovapd [%c] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1 = x86.dm.vmovapd [%c + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %2 = x86.get_avx_register : !x86.avx512reg<zmm28>
// CHECK-NEXT:      %3 = x86.dss.vpxord %2, %2 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %4 = x86.get_avx_register : !x86.avx512reg<zmm29>
// CHECK-NEXT:      %5 = x86.dss.vpxord %4, %4 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %6 = x86.dm.vmovapd [%a] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %7 = x86.dm.vmovapd [%a + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %8 = x86.rsm.vfmadd231pd %0, %6, [%b] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %9 = x86.rsm.vfmadd231pd %1, %6, [%b + 16] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %10 = x86.ri.add %a, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %11 = x86.rsm.vfmadd231pd %3, %7, [%b + 8] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %12 = x86.rsm.vfmadd231pd %5, %7, [%b + 24] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %13 = x86.ri.add %b, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %14 = x86.dss.vaddpd %11, %8 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %15 = x86.dss.vaddpd %12, %9 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      x86.ms.vmovapd [%c], %14 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%c + 64], %15 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %16 = x86.ri.sub %10, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %17 = x86.ri.sub %13, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %18 = x86.ri.add %c, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %a_out = x86.ri.sub %16, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %b_out = x86.ri.add %17, 32 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %c_out = x86.ri.add %18, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @fsdbcst(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 8 : i64, n = 2 : i64, k = 2 : i64, lda = 8 : i64, ldb = 2 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}
