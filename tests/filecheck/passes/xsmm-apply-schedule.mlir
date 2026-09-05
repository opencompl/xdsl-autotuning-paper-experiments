// RUN: xdsl-opt %s -p xsmm-apply-schedule | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @matmul(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.dm.vmovapd [%c] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %1 = x86.dm.vmovapd [%a] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %2 = x86.ri.add %a, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %3 = x86.rsm.vfmadd231pd %0, %1, [%b] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %4 = x86.ri.add %b, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%c], %3 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %5 = x86.ri.add %2, 0 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %6 = x86.ri.sub %4, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %7 = x86.ri.add %c, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %a_out = x86.ri.sub %5, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %b_out = x86.ri.add %6, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %c_out = x86.ri.add %7, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
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
