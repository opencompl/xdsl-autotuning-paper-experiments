// RUN: xdsl-opt %s --split-input-file -p test-xsmm-tiling | filecheck %s

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @tile_n(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>) {
// CHECK-NEXT:     %0 = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:     %1, %a_out, %b_out, %c_out = x86_scf.for %2 : !x86.reg64  = %0 to 28 : si32 step 14 : si32 iter_args(%3 = %a, %4 = %b, %5 = %c) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) {
// CHECK-NEXT:       %6, %7, %8 = xsmm.matmul %3, %4, %5 {m = 16 : i64, n = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:       x86_scf.yield %6, %7, %8 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>
// CHECK-NEXT:     }
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @tile_n(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 16 : i64, n = 28 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> {test_tile_size = 14 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}

// -----

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @split_n(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>) {
// CHECK-NEXT:     %0, %1, %2 = xsmm.matmul %a, %b, %c {m = 16 : i64, n = 12 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:     %a_out, %b_out, %c_out = xsmm.matmul %0, %1, %2 {m = 16 : i64, n = 10 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @split_n(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 16 : i64, n = 22 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> {test_tile_size = 12 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}

// -----

// CHECK:      builtin.module {
// CHECK-NEXT:   x86_func.func @tile_m_with_remainder(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>) {
// CHECK-NEXT:     %0 = x86.di.mov 0 : () -> !x86.reg64
// CHECK-NEXT:     %1, %2, %3, %4 = x86_scf.for %5 : !x86.reg64  = %0 to 16 : si32 step 8 : si32 iter_args(%6 = %a, %7 = %b, %8 = %c) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) {
// CHECK-NEXT:       %9, %10, %11 = xsmm.matmul %6, %7, %8 {m = 8 : i64, n = 2 : i64, k = 5 : i64, lda = 17 : i64, ldb = 5 : i64, ldc = 17 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "m"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:       x86_scf.yield %9, %10, %11 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>
// CHECK-NEXT:     }
// CHECK-NEXT:     %a_out, %b_out, %c_out = xsmm.matmul %2, %3, %4 {m = 1 : i64, n = 2 : i64, k = 5 : i64, lda = 17 : i64, ldb = 5 : i64, ldc = 17 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "m"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:     x86_func.ret
// CHECK-NEXT:   }
// CHECK-NEXT: }

x86_func.func @tile_m_with_remainder(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>
) {
  %a_out, %b_out, %c_out = "xsmm.matmul"(%a, %b, %c) <{m = 17 : i64, n = 2 : i64, k = 5 : i64, lda = 17 : i64, ldb = 5 : i64, ldc = 17 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> {test_tile_size = 8 : i64} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
  x86_func.ret
}
