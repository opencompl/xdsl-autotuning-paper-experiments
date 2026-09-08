// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=llvm-skx-narrow-fsdbcst}' | filecheck %s
// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=llvm-skx-narrow-fsdbcst disable-regalloc=true}' | filecheck %s --check-prefix AUTO

// An M of two f64 lanes fits an xmm exactly, so neither the accumulators nor
// the A column touch a wider bank and no tail mask is needed.

// AUTO-LABEL:   x86_func.func @narrow_fsdbcst
// AUTO-NOT:     !x86.avx512reg
// AUTO-NOT:     !x86.avx2reg
// AUTO-NOT:     !x86.avx512maskreg
// AUTO:         x86_func.ret

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @narrow_fsdbcst(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>) {
// CHECK-NEXT:      %0 = x86.dm.vmovupd [%c] : (!x86.reg64<rdx>) -> !x86.ssereg<xmm30>
// CHECK-NEXT:      %1 = x86.dm.vmovupd [%c + 16] : (!x86.reg64<rdx>) -> !x86.ssereg<xmm31>
// CHECK-NEXT:      %2 = x86.dm.vmovupd [%a] : (!x86.reg64<rdi>) -> !x86.ssereg<xmm0>
// CHECK-NEXT:      %3 = x86.dm.vmovupd [%a + 16] : (!x86.reg64<rdi>) -> !x86.ssereg<xmm1>
// CHECK-NEXT:      %4 = x86.rsm.vfmadd231pd %0, %2, [%b] {broadcast} : (!x86.ssereg<xmm30>, !x86.ssereg<xmm0>, !x86.reg64<rsi>) -> !x86.ssereg<xmm30>
// CHECK-NEXT:      %5 = x86.rsm.vfmadd231pd %1, %2, [%b + 16] {broadcast} : (!x86.ssereg<xmm31>, !x86.ssereg<xmm0>, !x86.reg64<rsi>) -> !x86.ssereg<xmm31>
// CHECK-NEXT:      %6 = x86.ri.add %a, 32 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %7 = x86.rsm.vfmadd231pd %4, %3, [%b + 8] {broadcast} : (!x86.ssereg<xmm30>, !x86.ssereg<xmm1>, !x86.reg64<rsi>) -> !x86.ssereg<xmm30>
// CHECK-NEXT:      %8 = x86.rsm.vfmadd231pd %5, %3, [%b + 24] {broadcast} : (!x86.ssereg<xmm31>, !x86.ssereg<xmm1>, !x86.reg64<rsi>) -> !x86.ssereg<xmm31>
// CHECK-NEXT:      %9 = x86.ri.add %b, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovupd [%c], %7 : (!x86.reg64<rdx>, !x86.ssereg<xmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovupd [%c + 16], %8 : (!x86.reg64<rdx>, !x86.ssereg<xmm31>) -> ()
// CHECK-NEXT:      %10 = x86.ri.sub %6, 16 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %11 = x86.ri.sub %9, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %12 = x86.ri.add %c, 16 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %a_out = x86.ri.sub %10, 16 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %b_out = x86.ri.add %11, 32 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %c_out = x86.ri.add %12, 16 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @narrow_fsdbcst(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 2 : i64, n = 2 : i64, k = 2 : i64, lda = 2 : i64, ldb = 2 : i64, ldc = 2 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}
