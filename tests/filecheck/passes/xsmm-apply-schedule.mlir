// RUN: xdsl-opt %s -p xsmm-apply-schedule | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @matmul(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %a_out, %b_out, %c_out, %rbp_out, %rsp_out = x86_scf.for %2 : !x86.reg64<r11>  = %0 to 1 : si32 step 1 : si32 iter_args(%3 = %a, %4 = %b, %5 = %c, %6 = %rbp, %7 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %8 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %9, %10, %11, %12, %13, %14 = x86_scf.for %15 : !x86.reg64<r10>  = %8 to 8 : si32 step 8 : si32 iter_args(%16 = %3, %17 = %4, %18 = %5, %19 = %6, %20 = %7) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %21 = x86.dm.vmovapd [%18] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %22 = x86.dm.vmovapd [%16] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %23 = x86.ri.add %16, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %24 = x86.rsm.vfmadd231pd %21, %22, [%17] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %25 = x86.ri.add %17, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          x86.ms.vmovapd [%18], %24 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:          %26 = x86.ri.add %23, 0 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %27 = x86.ri.sub %25, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %28 = x86.ri.add %18, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %26, %27, %28, %19, %20 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %29 = x86.ri.sub %10, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %30 = x86.ri.add %11, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %31 = x86.ri.add %12, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %29, %30, %31, %13, %14 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @matmul(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 8 : i64, n = 1 : i64, k = 1 : i64, lda = 8 : i64, ldb = 1 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}
