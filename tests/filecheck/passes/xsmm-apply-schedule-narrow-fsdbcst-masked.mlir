// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=llvm-skx-narrow-fsdbcst},x86-regalloc-verify-liveness' -t x86-asm | filecheck %s
// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-fsdbcst},x86-regalloc-verify-liveness' -t x86-asm | filecheck %s --check-prefix WIDE

// An M of three f64 lanes has no exact bank, so it masks -- but on the ymm
// that covers it rather than on a zmm, and the mask covers that bank's four
// lanes instead of a full vector's eight.  The libxsmm nano-kernel run below
// is the same tile on the wide bank, for contrast.

// CHECK:       matmul_masked:
// CHECK-NEXT:      mov r15, 7
// CHECK-NEXT:      kmovb k1, r15d
// CHECK-NEXT:      vmovupd ymm30 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovupd ymm31 {k1}{z}, [rdx+24]
// CHECK-NEXT:      vmovupd ymm0 {k1}{z}, [rdi]
// CHECK-NEXT:      vmovupd ymm1 {k1}{z}, [rdi+24]
// CHECK-NEXT:      vfmadd231pd ymm30, ymm0, [rsi]{1to4}
// CHECK-NEXT:      vfmadd231pd ymm31, ymm0, [rsi+16]{1to4}
// CHECK-NEXT:      add rdi, 48
// CHECK-NEXT:      vfmadd231pd ymm30, ymm1, [rsi+8]{1to4}
// CHECK-NEXT:      vfmadd231pd ymm31, ymm1, [rsi+24]{1to4}
// CHECK-NEXT:      add rsi, 16
// CHECK-NEXT:      vmovupd [rdx] {k1}, ymm30
// CHECK-NEXT:      vmovupd [rdx+24] {k1}, ymm31

// WIDE:        matmul_masked:
// WIDE-NEXT:       mov r15, 7
// WIDE-NEXT:       kmovb k1, r15d
// WIDE-NEXT:       vmovupd zmm30 {k1}{z}, [rdx]
// WIDE-NEXT:       vmovupd zmm31 {k1}{z}, [rdx+24]
// WIDE:            vfmadd231pd zmm30, zmm0, [rsi]{1to8}
// WIDE-NOT:        ymm

x86_func.func @matmul_masked(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 3 : i64, n = 2 : i64, k = 2 : i64, lda = 3 : i64, ldb = 2 : i64, ldc = 3 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}
