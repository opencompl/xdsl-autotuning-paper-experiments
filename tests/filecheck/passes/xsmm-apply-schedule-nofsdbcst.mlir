// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-nofsdbcst}' | filecheck %s
// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-nofsdbcst disable-regalloc=true}' | filecheck %s --check-prefix AUTO

// AUTO-LABEL:   x86_func.func @nofsdbcst
// AUTO-NOT:     !x86.avx512reg<zmm
// AUTO:         x86_func.ret

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @nofsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %a_out, %b_out, %c_out, %rbp_out, %rsp_out = x86_scf.for %2 : !x86.reg64<r11>  = %0 to 1 : si32 step 1 : si32 iter_args(%3 = %a, %4 = %b, %5 = %c, %6 = %rbp, %7 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %8 = x86.di.mov 1 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %9 = x86.ks.kmovw %8 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %10 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r10>  = %10 to 17 : si32 step 17 : si32 iter_args(%18 = %3, %19 = %4, %20 = %5, %21 = %6, %22 = %7) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %23 = x86.dm.vmovups [%20] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %24 = x86.dmk.vmovups[%20 + 64], %9 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %25 = x86.dm.vmovups [%18] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %26 = x86.dmk.vmovups[%18 + 64], %9 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %27 = x86.dm.vbroadcastss [%19] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %28 = x86.ri.add %19, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %29 = x86.ri.add %18, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %30 = x86.rss.vfmadd231ps %23, %25, %27 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %31 = x86.rss.vfmadd231ps %24, %26, %27 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %32 = x86.dm.vmovups [%29] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %33 = x86.dmk.vmovups[%29 + 64], %9 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %34 = x86.dm.vbroadcastss [%28] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %35 = x86.ri.add %28, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %36 = x86.ri.add %29, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %37 = x86.rss.vfmadd231ps %30, %32, %34 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %38 = x86.rss.vfmadd231ps %31, %33, %34 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86.ms.vmovups [%20], %37 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:          x86.msk.vmovups[%20 + 64], %38, %9 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:          %39 = x86.ri.sub %36, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %40 = x86.ri.sub %35, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %41 = x86.ri.add %20, 68 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %39, %40, %41, %21, %22 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %42 = x86.ri.sub %12, 68 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %43 = x86.ri.add %13, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %44 = x86.ri.add %14, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %42, %43, %44, %15, %16 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
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
