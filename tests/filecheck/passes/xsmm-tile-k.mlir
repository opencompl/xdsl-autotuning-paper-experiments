// RUN: xdsl-opt %s --split-input-file -p test-xsmm-tiling | filecheck %s

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @below_threshold(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:     %a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 23 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> {test_tile_size = 23 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @below_threshold(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 23 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> {test_tile_size = 23 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @exact_tiles(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:     %0 = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:     %1, %a_out, %b_out, %acc_out = x86_scf.for %2 : !x86.reg64  = %0 to 24 : si32 step 4 : si32 iter_args(%3 = %a, %4 = %b, %5 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:       %6, %7, %8 = "xsmm.matmul_reg"(%3, %4, %5) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:       x86_scf.yield %6, %7, %8 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>
// CHECK-NEXT:     }
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @exact_tiles(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 24 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> {test_tile_size = 4 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @tiles_and_remainder(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:     %0 = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:     %1, %2, %3, %4 = x86_scf.for %5 : !x86.reg64  = %0 to 24 : si32 step 4 : si32 iter_args(%6 = %a, %7 = %b, %8 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:       %9, %10, %11 = "xsmm.matmul_reg"(%6, %7, %8) <{m = 8 : i64, n = 1 : i64, k = 4 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:       x86_scf.yield %9, %10, %11 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>
// CHECK-NEXT:     }
// CHECK-NEXT:     %a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%2, %3, %4) <{m = 8 : i64, n = 1 : i64, k = 1 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @tiles_and_remainder(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %acc) <{m = 8 : i64, n = 1 : i64, k = 25 : i64, lda = 8 : i64, ldb = 32 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> {test_tile_size = 4 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
  x86_func.ret
}

// -----

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @read_only_mask(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %mask: !x86.avx512maskreg<k1>, %acc: !x86.avx512reg<zmm31>) {
// CHECK-NEXT:     %0 = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:     %1, %a_out, %b_out, %acc_out = x86_scf.for %2 : !x86.reg64  = %0 to 24 : si32 step 4 : si32 iter_args(%3 = %a, %4 = %b, %5 = %acc) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:       %6, %7, %8 = "xsmm.matmul_reg"(%3, %4, %mask, %5) <{m = 7 : i64, n = 1 : i64, k = 4 : i64, lda = 7 : i64, ldb = 32 : i64, datatype = f64, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:       x86_scf.yield %6, %7, %8 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>
// CHECK-NEXT:     }
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @read_only_mask(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %mask: !x86.avx512maskreg<k1>,
  %acc: !x86.avx512reg<zmm31>
) {
  %a_out, %b_out, %acc_out = "xsmm.matmul_reg"(%a, %b, %mask, %acc) <{m = 7 : i64, n = 1 : i64, k = 24 : i64, lda = 7 : i64, ldb = 32 : i64, datatype = f64, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1>}> {test_tile_size = 4 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.avx512reg<zmm31>)
  x86_func.ret
}
