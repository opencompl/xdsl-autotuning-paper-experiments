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
// CHECK-NEXT:          %26, %27, %28, %29, %30 = "xsmm.matmul"(%21, %22, %23, %24, %25) <{m = 16 : i64, n = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:          %31 = x86.ri.sub %26, 1920 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %32 = x86.ri.sub %27, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %33 = x86.ri.add %28, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %31, %32, %33, %29, %30 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %34 = x86.ri.sub %15, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %35 = x86.ri.add %16, 1792 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %36 = x86.ri.add %17, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %34, %35, %36, %18, %19 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
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
// CHECK-NEXT:          %30, %31, %32, %33, %34, %35 = "xsmm.matmul"(%24, %25, %26, %27, %28, %29) <{m = 16 : i64, n = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>)
// CHECK-NEXT:          %36 = x86.ri.sub %30, 1920 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %37 = x86.ri.sub %31, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %38 = x86.ri.add %32, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %36, %37, %38, %33, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>
// CHECK-NEXT:        }
// CHECK-NEXT:        %39 = x86.ri.sub %17, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %40 = x86.ri.add %18, 1792 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %41 = x86.ri.add %19, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %39, %40, %41, %20, %21, %22 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r8>
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
// CHECK-NEXT:          %26, %27, %28, %29, %30 = "xsmm.matmul"(%21, %22, %23, %24, %25) <{m = 64 : i64, n = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          %31 = x86.ri.sub %26, 35584 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %32 = x86.ri.sub %27, 512 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %33 = x86.ri.add %28, 256 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          x86_scf.yield %31, %32, %33, %29, %30 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %34 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %35 = x86.ks.kmovw %34 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %36 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %37, %38, %39, %40, %41, %42 = x86_scf.for %43 : !x86.reg64<r10>  = %36 to 70 : si32 step 6 : si32 iter_args(%44 = %15, %45 = %16, %46 = %17, %47 = %18, %48 = %19) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %49, %50, %51, %52, %53 = "xsmm.matmul"(%44, %45, %46, %47, %48, %35) <{m = 6 : i64, n = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          %54 = x86.ri.sub %49, 35816 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %55 = x86.ri.sub %50, 512 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %56 = x86.ri.add %51, 24 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          x86_scf.yield %54, %55, %56, %52, %53 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %57 = x86.ri.sub %38, 280 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %58 = x86.ri.add %39, 3072 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %59 = x86.ri.add %40, 1400 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        x86_scf.yield %57, %58, %59, %41, %42 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      %60 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %61, %62, %63, %64, %65, %66 = x86_scf.for %67 : !x86.reg64<r11>  = %60 to 20 : si32 step 5 : si32 iter_args(%68 = %2, %69 = %3, %70 = %4, %71 = %5, %72 = %6) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %73 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %74, %75, %76, %77, %78, %79 = x86_scf.for %80 : !x86.reg64<r10>  = %73 to 64 : si32 step 64 : si32 iter_args(%81 = %68, %82 = %69, %83 = %70, %84 = %71, %85 = %72) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %86, %87, %88, %89, %90 = "xsmm.matmul"(%81, %82, %83, %84, %85) <{m = 64 : i64, n = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          %91 = x86.ri.sub %86, 35584 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %92 = x86.ri.sub %87, 512 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %93 = x86.ri.add %88, 256 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          x86_scf.yield %91, %92, %93, %89, %90 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %94 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %95 = x86.ks.kmovw %94 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %96 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %97, %98, %99, %100, %101, %102 = x86_scf.for %103 : !x86.reg64<r10>  = %96 to 70 : si32 step 6 : si32 iter_args(%104 = %75, %105 = %76, %106 = %77, %107 = %78, %108 = %79) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %109, %110, %111, %112, %113 = "xsmm.matmul"(%104, %105, %106, %107, %108, %95) <{m = 6 : i64, n = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "k", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          %114 = x86.ri.sub %109, 35816 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %115 = x86.ri.sub %110, 512 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          %116 = x86.ri.add %111, 24 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:          x86_scf.yield %114, %115, %116, %112, %113 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %117 = x86.ri.sub %98, 280 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %118 = x86.ri.add %99, 2560 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %119 = x86.ri.add %100, 1120 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        x86_scf.yield %117, %118, %119, %101, %102 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
