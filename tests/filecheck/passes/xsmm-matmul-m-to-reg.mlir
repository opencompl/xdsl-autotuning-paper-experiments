// RUN: xdsl-opt %s --split-input-file -p xsmm-matmul-m-to-reg | filecheck %s

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @aligned_f64(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.dm.vmovapd [%c] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1 = x86.dm.vmovapd [%c + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %a_out, %b_out, %rbp_out, %rsp_out, %2, %3 = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %0, %1) <{m = 8 : i64, n = 2 : i64, k = 5 : i64, lda = 8 : i64, ldb = 5 : i64, datatype = f64, aligned_a = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:      x86.ms.vmovapd [%c], %2 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%c + 64], %3 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %c_out = x86.ri.add %c, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @aligned_f64(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 8 : i64, n_start = 0 : i64, n = 2 : i64, k = 5 : i64, lda = 8 : i64, ldb = 5 : i64, ldc = 8 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @masked_f32(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %mask: !x86.avx512maskreg<k1>) {
// CHECK-NEXT:      %0 = x86.dm.vmovups [%c] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %1 = x86.dmk.vmovups[%c + 64], %mask {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %a_out, %b_out, %rbp_out, %rsp_out, %2, %3 = "xsmm.matmul_reg"(%a, %b, %rbp, %rsp, %mask, %0, %1) <{m = 17 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, datatype = f32, aligned_a = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 2>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 2>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:      x86.ms.vmovups [%c], %2 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%c + 64], %3, %mask : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %c_out = x86.ri.add %c, 68 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %mask_copy = x86.dk.kmovw %mask : (!x86.avx512maskreg<k1>) -> !x86.reg64<rax>
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

x86_func.func @masked_f32(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %mask: !x86.avx512maskreg<k1>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp, %mask) <{m = 17 : i64, n_start = 0 : i64, n = 1 : i64, k = 2 : i64, lda = 17 : i64, ldb = 16 : i64, ldc = 17 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  %mask_copy = x86.dk.kmovw %mask : (!x86.avx512maskreg<k1>) -> !x86.reg64<rax>
  x86_func.ret
}
