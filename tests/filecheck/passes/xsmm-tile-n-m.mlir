// RUN: xdsl-opt %s --split-input-file -p xsmm-tile-n-m | filecheck %s

x86_func.func @single_n_range(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %0, %1, %2, %3, %4 = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n = 28 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @single_n_range(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 28 : si32 step 14 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %14, %15, %16, %17, %18, %19 = x86_scf.for %20 : !x86.reg64<r10>  = %13 to 16 : si32 step 16 : si32 iter_args(%21 = %8, %22 = %9, %23 = %10, %24 = %11, %25 = %12) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %26, %27, %28, %29, %30 = "xsmm.matmul"(%21, %22, %23, %24, %25) <{m = 16 : i64, n = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:          x86_scf.yield %26, %27, %28, %29, %30 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %31 = x86.ri.sub %15, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %32 = x86.ri.add %16, 1792 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %33 = x86.ri.add %17, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %31, %32, %33, %18, %19 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @loop_carried_out(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>,
  %state: !x86.reg64<r8>
) {
  %0, %1, %2, %3, %4, %5 = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp, %state) <{m = 16 : i64, n = 28 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @loop_carried_out(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>, %state: !x86.reg64<r8>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6, %7 = x86_scf.for %8 : !x86.reg64<r11>  = %0 to 28 : si32 step 14 : si32 iter_args(%9 = %a, %10 = %b, %11 = %c, %12 = %rbp, %13 = %rsp, %14 = %state) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>) {
// CHECK-NEXT:        %15 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %16, %17, %18, %19, %20, %21, %22 = x86_scf.for %23 : !x86.reg64<r10>  = %15 to 16 : si32 step 16 : si32 iter_args(%24 = %9, %25 = %10, %26 = %11, %27 = %12, %28 = %13, %29 = %14) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>) {
// CHECK-NEXT:          %30, %31, %32, %33, %34, %35 = "xsmm.matmul"(%24, %25, %26, %27, %28, %29) <{m = 16 : i64, n = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>)
// CHECK-NEXT:          x86_scf.yield %30, %31, %32, %33, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>
// CHECK-NEXT:        }
// CHECK-NEXT:        %36 = x86.ri.sub %17, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %37 = x86.ri.add %18, 1792 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %38 = x86.ri.add %19, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %36, %37, %38, %20, %21, %22 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

x86_func.func @two_n_ranges_with_m_remainder(
  %a: !x86.reg64,
  %b: !x86.reg64,
  %c: !x86.reg64,
  %rbp: !x86.reg64,
  %rsp: !x86.reg64
) {
  %0, %1, %2, %3, %4 = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 70 : i64, n = 38 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @two_n_ranges_with_m_remainder(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 18 : si32 step 6 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %13 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %14, %15, %16, %17, %18, %19 = x86_scf.for %20 : !x86.reg64<r10>  = %13 to 64 : si32 step 64 : si32 iter_args(%21 = %8, %22 = %9, %23 = %10, %24 = %11, %25 = %12) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %26, %27, %28, %29, %30 = "xsmm.matmul"(%21, %22, %23, %24, %25) <{m = 64 : i64, n = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          x86_scf.yield %26, %27, %28, %29, %30 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %31 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %32 = x86.ks.kmovw %31 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %33 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %34, %35, %36, %37, %38, %39 = x86_scf.for %40 : !x86.reg64<r10>  = %33 to 70 : si32 step 6 : si32 iter_args(%41 = %15, %42 = %16, %43 = %17, %44 = %18, %45 = %19) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %46, %47, %48, %49, %50 = "xsmm.matmul"(%41, %42, %43, %44, %45, %32) <{m = 6 : i64, n = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          x86_scf.yield %46, %47, %48, %49, %50 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %51 = x86.ri.sub %35, 280 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %52 = x86.ri.add %36, 3072 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %53 = x86.ri.add %37, 1400 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        x86_scf.yield %51, %52, %53, %38, %39 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      %54 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %55, %56, %57, %58, %59, %60 = x86_scf.for %61 : !x86.reg64<r11>  = %54 to 20 : si32 step 5 : si32 iter_args(%62 = %2, %63 = %3, %64 = %4, %65 = %5, %66 = %6) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %67 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %68, %69, %70, %71, %72, %73 = x86_scf.for %74 : !x86.reg64<r10>  = %67 to 64 : si32 step 64 : si32 iter_args(%75 = %62, %76 = %63, %77 = %64, %78 = %65, %79 = %66) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %80, %81, %82, %83, %84 = "xsmm.matmul"(%75, %76, %77, %78, %79) <{m = 64 : i64, n = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          x86_scf.yield %80, %81, %82, %83, %84 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %85 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %86 = x86.ks.kmovw %85 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %87 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %88, %89, %90, %91, %92, %93 = x86_scf.for %94 : !x86.reg64<r10>  = %87 to 70 : si32 step 6 : si32 iter_args(%95 = %69, %96 = %70, %97 = %71, %98 = %72, %99 = %73) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %100, %101, %102, %103, %104 = "xsmm.matmul"(%95, %96, %97, %98, %99, %86) <{m = 6 : i64, n = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "m", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          x86_scf.yield %100, %101, %102, %103, %104 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %105 = x86.ri.sub %89, 280 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %106 = x86.ri.add %90, 2560 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %107 = x86.ri.add %91, 1120 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        x86_scf.yield %105, %106, %107, %92, %93 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
