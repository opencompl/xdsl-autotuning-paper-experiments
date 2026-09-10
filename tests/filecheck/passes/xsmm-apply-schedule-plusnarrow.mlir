// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-plusnarrow}' | filecheck %s
// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx}' | filecheck %s --check-prefix WIDE

// Three f64 lanes of M.  The LIBXSMM heuristic sends that to `fsdbcst`, which
// masks off five lanes of a zmm and, at this N, keeps four accumulator sets to
// shorten the cross-K chain; `libxsmm-skx-plusnarrow` sends it to the narrow
// kernel instead, which masks one lane of a ymm and keeps one set.  The tiling
// either way is the same: one 3-by-2 tile with K fully unrolled.

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @plusnarrow(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>) {
// CHECK-NEXT:      %0 = x86.di.mov 7 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %1 = x86.ks.kmovb %0 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %2 = x86.dmk.vmovupd[%c], %1 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm30>
// CHECK-NEXT:      %3 = x86.dmk.vmovupd[%c + 24], %1 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:      %4 = x86.dmk.vmovupd[%a], %1 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT:      %5 = x86.dmk.vmovupd[%a + 24], %1 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm1>
// CHECK-NEXT:      %6 = x86.rsm.vfmadd231pd %2, %4, [%b] {broadcast} : (!x86.avx2reg<ymm30>, !x86.avx2reg<ymm0>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm30>
// CHECK-NEXT:      %7 = x86.rsm.vfmadd231pd %3, %4, [%b + 16] {broadcast} : (!x86.avx2reg<ymm31>, !x86.avx2reg<ymm0>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:      %8 = x86.ri.add %a, 48 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %9 = x86.rsm.vfmadd231pd %6, %5, [%b + 8] {broadcast} : (!x86.avx2reg<ymm30>, !x86.avx2reg<ymm1>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm30>
// CHECK-NEXT:      %10 = x86.rsm.vfmadd231pd %7, %5, [%b + 24] {broadcast} : (!x86.avx2reg<ymm31>, !x86.avx2reg<ymm1>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:      %11 = x86.ri.add %b, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.msk.vmovupd[%c], %9, %1 : (!x86.reg64<rdx>, !x86.avx2reg<ymm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovupd[%c + 24], %10, %1 : (!x86.reg64<rdx>, !x86.avx2reg<ymm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %12 = x86.ri.sub %8, 24 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %13 = x86.ri.sub %11, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %14 = x86.ri.add %c, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %a_out = x86.ri.sub %12, 24 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %b_out = x86.ri.add %13, 32 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %c_out = x86.ri.add %14, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// The same three lanes under the unmodified heuristic: the same mask, over a
// zmm, and two more accumulators to zero and add back in at the end.
// WIDE:        %1 = x86.ks.kmovb %0 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// WIDE-NEXT:   %2 = x86.dmk.vmovupd[%c], %1 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// WIDE-NEXT:   %3 = x86.dmk.vmovupd[%c + 24], %1 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// WIDE:        x86.dss.vpxord
// WIDE:        x86.dss.vpxord
// WIDE:        x86.dss.vaddpd
// WIDE:        x86.dss.vaddpd
// WIDE-NOT:    !x86.avx2reg

x86_func.func @plusnarrow(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 3 : i64, n = 2 : i64, k = 2 : i64, lda = 3 : i64, ldb = 2 : i64, ldc = 3 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}
