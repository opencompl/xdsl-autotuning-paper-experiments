// RUN: xdsl-opt %s -p 'convert-xsmm-to-x86{strategy=zen5-kdot}' | filecheck %s

// CHECK-LABEL: x86_func.func @matmul_reg_kdot
// CHECK:         %[[A:.*]] = x86.dm.vmovupd [%a] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %[[ACC0:.*]] = x86.rsm.vfmadd231pd %acc0, %[[A]], [%b] : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rdi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %[[ACC1:.*]] = x86.rsm.vfmadd231pd %acc1, %[[A]], [%b + 64] : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rdi>) -> !x86.avx512reg<zmm31>
// CHECK:         x86.dsi.vextractf64x4 %[[ACC0]], 1
// CHECK:         x86.dssi.vshufpd
// CHECK:         x86.dss.vaddsd
// CHECK:         %acc0_out = x86.get_avx_register : !x86.avx512reg<zmm30>
// CHECK:         x86.dsi.vextractf64x4 %[[ACC1]], 1
// CHECK:         x86.dssi.vshufpd
// CHECK:         x86.dss.vaddsd
// CHECK:         %acc1_out = x86.get_avx_register : !x86.avx512reg<zmm31>
// CHECK:         %a_out = x86.ri.add %a, 8
// CHECK-NOT:     xsmm.matmul_reg

x86_func.func @matmul_reg_kdot(
  %a: !x86.reg64<rsi>,
  %b: !x86.reg64<rdi>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %mask: !x86.avx512maskreg<k1>,
  %acc0: !x86.avx512reg<zmm30>,
  %acc1: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %rbp_out, %rsp_out, %acc0_out, %acc1_out = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %mask, %acc0, %acc1) <{m = 1 : i64, n = 2 : i64, k = 8 : i64, lda = 1 : i64, ldb = 8 : i64, datatype = f64, aligned_a = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 2>}> : (!x86.reg64<rsi>, !x86.reg64<rdi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rsi>, !x86.reg64<rdi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
  x86_func.ret
}
