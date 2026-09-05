// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-nofsdbcst}' | filecheck %s
// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-nofsdbcst disable-regalloc=true}' | filecheck %s --check-prefix AUTO

// AUTO-LABEL:   x86_func.func @nofsdbcst
// AUTO-NOT:     !x86.avx512reg<zmm
// AUTO:         x86_func.ret

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @nofsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 1 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %1 = x86.ks.kmovw %0 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %2 = x86.dm.vmovups [%c] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %3 = x86.dmk.vmovups[%c + 64], %1 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %4 = x86.dm.vmovups [%a] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %5 = x86.dmk.vmovups[%a + 64], %1 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %6 = x86.dm.vbroadcastss [%b] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %7 = x86.ri.add %b, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %8 = x86.ri.add %a, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %9 = x86.rss.vfmadd231ps %2, %4, %6 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %10 = x86.rss.vfmadd231ps %3, %5, %6 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %11 = x86.dm.vmovups [%8] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %12 = x86.dmk.vmovups[%8 + 64], %1 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %13 = x86.dm.vbroadcastss [%7] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %14 = x86.ri.add %7, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %15 = x86.ri.add %8, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %16 = x86.rss.vfmadd231ps %9, %11, %13 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %17 = x86.rss.vfmadd231ps %10, %12, %13 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      x86.ms.vmovups [%c], %16 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%c + 64], %17, %1 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %18 = x86.ri.sub %15, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %19 = x86.ri.sub %14, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %20 = x86.ri.add %c, 68 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %a_out = x86.ri.sub %18, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %b_out = x86.ri.add %19, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %c_out = x86.ri.add %20, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @nofsdbcst(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 2 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}
