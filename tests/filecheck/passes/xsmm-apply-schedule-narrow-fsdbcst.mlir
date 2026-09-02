// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-narrow-fsdbcst}' | filecheck %s

// The same one-M-vector kernel as xsmm-apply-schedule-fsdbcst.mlir, with M=3
// instead of a full vector: every vector register is a ymm rather than a masked
// zmm, the FMAs broadcast 1to4, and the mask enables the ymm's three low lanes
// (0b111 = 7) rather than three of a zmm's eight.

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @narrow_fsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %a_out, %b_out, %c_out, %rbp_out, %rsp_out = x86_scf.for %2 : !x86.reg64<r11>  = %0 to 2 : si32 step 2 : si32 iter_args(%3 = %a, %4 = %b, %5 = %c, %6 = %rbp, %7 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %8 = x86.di.mov 7 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %9 = x86.ks.kmovb %8 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %10 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r10>  = %10 to 3 : si32 step 3 : si32 iter_args(%18 = %3, %19 = %4, %20 = %5, %21 = %6, %22 = %7) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %23 = x86.dmk.vmovupd[%20], %9 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm30>
// CHECK-NEXT:          %24 = x86.dmk.vmovupd[%20 + 24], %9 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:          %25 = x86.get_avx_register : !x86.avx2reg<ymm28>
// CHECK-NEXT:          %26 = x86.dss.vpxord %25, %25 : (!x86.avx2reg<ymm28>, !x86.avx2reg<ymm28>) -> !x86.avx2reg<ymm28>
// CHECK-NEXT:          %27 = x86.get_avx_register : !x86.avx2reg<ymm29>
// CHECK-NEXT:          %28 = x86.dss.vpxord %27, %27 : (!x86.avx2reg<ymm29>, !x86.avx2reg<ymm29>) -> !x86.avx2reg<ymm29>
// CHECK-NEXT:          %29 = x86.dmk.vmovupd[%18], %9 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT:          %30 = x86.dmk.vmovupd[%18 + 24], %9 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx2reg<ymm1>
// CHECK-NEXT:          %31 = x86.rsm.vfmadd231pd %23, %29, [%19] {broadcast} : (!x86.avx2reg<ymm30>, !x86.avx2reg<ymm0>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm30>
// CHECK-NEXT:          %32 = x86.rsm.vfmadd231pd %24, %29, [%19 + 16] {broadcast} : (!x86.avx2reg<ymm31>, !x86.avx2reg<ymm0>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:          %33 = x86.ri.add %18, 48 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %34 = x86.rsm.vfmadd231pd %26, %30, [%19 + 8] {broadcast} : (!x86.avx2reg<ymm28>, !x86.avx2reg<ymm1>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm28>
// CHECK-NEXT:          %35 = x86.rsm.vfmadd231pd %28, %30, [%19 + 24] {broadcast} : (!x86.avx2reg<ymm29>, !x86.avx2reg<ymm1>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm29>
// CHECK-NEXT:          %36 = x86.ri.add %19, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %37 = x86.dss.vaddpd %34, %31 : (!x86.avx2reg<ymm28>, !x86.avx2reg<ymm30>) -> !x86.avx2reg<ymm30>
// CHECK-NEXT:          %38 = x86.dss.vaddpd %35, %32 : (!x86.avx2reg<ymm29>, !x86.avx2reg<ymm31>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:          x86.msk.vmovupd[%20], %37, %9 : (!x86.reg64<rdx>, !x86.avx2reg<ymm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:          x86.msk.vmovupd[%20 + 24], %38, %9 : (!x86.reg64<rdx>, !x86.avx2reg<ymm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:          %39 = x86.ri.sub %33, 24 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %40 = x86.ri.sub %36, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %41 = x86.ri.add %20, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %39, %40, %41, %21, %22 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %42 = x86.ri.sub %12, 24 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %43 = x86.ri.add %13, 32 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %44 = x86.ri.add %14, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %42, %43, %44, %15, %16 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @narrow_fsdbcst(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 3 : i64, n = 2 : i64, k = 2 : i64, lda = 3 : i64, ldb = 2 : i64, ldc = 3 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}
