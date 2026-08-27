// RUN: xdsl-opt %s --split-input-file -p xsmm-tile-n-m | filecheck %s

x86_func.func @single_n_range(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 16 : i64, n_start = 4 : i64, n_blocking = 28 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @single_n_range(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 4 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 32 : si32 step 14 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %13 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %14, %15, %16, %17, %18, %19 = x86_scf.for %20 : !x86.reg64<r10>  = %13 to 16 : si32 step 16 : si32 iter_args(%21 = %8, %22 = %9, %23 = %10, %24 = %11, %25 = %12) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %26, %27, %28, %29, %30 = "xsmm.matmul_m"(%21, %22, %23, %24, %25) <{m_blocking = 16 : i64, n_blocking = 14 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:          x86_scf.yield %26, %27, %28, %29, %30 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %31 = x86.ri.add %17, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %32 = x86.ri.add %16, 1792 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %33 = x86.ri.sub %15, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %33, %32, %31, %18, %19 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
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
  %0, %1, %2, %3, %4 = "xsmm.matmul_n"(%a, %b, %c, %rbp, %rsp) <{m = 70 : i64, n_start = 0 : i64, n_blocking = 38 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
  x86_func.ret
}

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @two_n_ranges_with_m_remainder(%a: !x86.reg64, %b: !x86.reg64, %c: !x86.reg64, %rbp: !x86.reg64, %rsp: !x86.reg64) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %2, %3, %4, %5, %6 = x86_scf.for %7 : !x86.reg64<r11>  = %0 to 18 : si32 step 6 : si32 iter_args(%8 = %a, %9 = %b, %10 = %c, %11 = %rbp, %12 = %rsp) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %13 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %14, %15, %16, %17, %18, %19 = x86_scf.for %20 : !x86.reg64<r10>  = %13 to 64 : si32 step 64 : si32 iter_args(%21 = %8, %22 = %9, %23 = %10, %24 = %11, %25 = %12) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %26, %27, %28, %29, %30 = "xsmm.matmul_m"(%21, %22, %23, %24, %25) <{m_blocking = 64 : i64, n_blocking = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          x86_scf.yield %26, %27, %28, %29, %30 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %31 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %32 = x86.ks.kmovw %31 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %33 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %34, %35, %36, %37, %38, %39, %40 = x86_scf.for %41 : !x86.reg64<r10>  = %33 to 70 : si32 step 6 : si32 iter_args(%42 = %15, %43 = %16, %44 = %17, %45 = %18, %46 = %19, %47 = %32) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:          %48, %49, %50, %51, %52, %53 = "xsmm.matmul_m"(%42, %43, %44, %45, %46, %47) <{m_blocking = 6 : i64, n_blocking = 6 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>)
// CHECK-NEXT:          x86_scf.yield %48, %49, %50, %51, %52, %53 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>
// CHECK-NEXT:        }
// CHECK-NEXT:        %54 = x86.ri.add %37, 1400 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %55 = x86.ri.add %36, 3072 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %56 = x86.ri.sub %35, 280 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        x86_scf.yield %56, %55, %54, %38, %39 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      %57 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %58, %59, %60, %61, %62, %63 = x86_scf.for %64 : !x86.reg64<r11>  = %57 to 38 : si32 step 5 : si32 iter_args(%65 = %2, %66 = %3, %67 = %4, %68 = %5, %69 = %6) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:        %70 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %71, %72, %73, %74, %75, %76 = x86_scf.for %77 : !x86.reg64<r10>  = %70 to 64 : si32 step 64 : si32 iter_args(%78 = %65, %79 = %66, %80 = %67, %81 = %68, %82 = %69) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) {
// CHECK-NEXT:          %83, %84, %85, %86, %87 = "xsmm.matmul_m"(%78, %79, %80, %81, %82) <{m_blocking = 64 : i64, n_blocking = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64)
// CHECK-NEXT:          x86_scf.yield %83, %84, %85, %86, %87 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:        }
// CHECK-NEXT:        %88 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:        %89 = x86.ks.kmovw %88 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:        %90 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %91, %92, %93, %94, %95, %96, %97 = x86_scf.for %98 : !x86.reg64<r10>  = %90 to 70 : si32 step 6 : si32 iter_args(%99 = %72, %100 = %73, %101 = %74, %102 = %75, %103 = %76, %104 = %89) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:          %105, %106, %107, %108, %109, %110 = "xsmm.matmul_m"(%99, %100, %101, %102, %103, %104) <{m_blocking = 6 : i64, n_blocking = 5 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>) -> (!x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>)
// CHECK-NEXT:          x86_scf.yield %105, %106, %107, %108, %109, %110 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.avx512maskreg<k1>
// CHECK-NEXT:        }
// CHECK-NEXT:        %111 = x86.ri.add %94, 1120 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %112 = x86.ri.add %93, 2560 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        %113 = x86.ri.sub %92, 280 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:        x86_scf.yield %113, %112, %111, %95, %96 : !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64, !x86.reg64
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
