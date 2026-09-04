// RUN: xdsl-opt %s -p 'xsmm-apply-schedule{strategy=libxsmm-skx-narrow-fsdbcst}' | filecheck %s

// M=12 splits into an eight-row blocked range and a four-row remainder, and each
// block fills its own bank exactly: the blocked one a zmm, the remainder a ymm.
// Neither needs a mask, where the full-width kernel masks four of the
// remainder's eight zmm lanes (k1 = 0b1111). This is the shape that pins the
// mask to the tile rather than the block: the blocked range spans 8 rows here
// but 16 or 24 for a larger M, and no bank holds that many f64 elements at all.
// xsmm-apply-schedule-narrow-fsdbcst.mlir covers a narrowed bank that does still
// need a mask.

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func @narrow_fsdbcst_remainder(%a: !x86.reg64<rdi>, %b: !x86.reg64<rsi>, %c: !x86.reg64<rdx>, %rbp: !x86.reg64<rbp>, %rsp: !x86.reg64<rsp>) {
// CHECK-NEXT:      %0 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      %1, %a_out, %b_out, %c_out, %rbp_out, %rsp_out = x86_scf.for %2 : !x86.reg64<r11>  = %0 to 1 : si32 step 1 : si32 iter_args(%3 = %a, %4 = %b, %5 = %c, %6 = %rbp, %7 = %rsp) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %8 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %9, %10, %11, %12, %13, %14 = x86_scf.for %15 : !x86.reg64<r10>  = %8 to 8 : si32 step 8 : si32 iter_args(%16 = %3, %17 = %4, %18 = %5, %19 = %6, %20 = %7) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %21 = x86.dm.vmovupd [%18] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %22 = x86.dm.vmovupd [%16] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %23 = x86.ri.add %16, 96 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %24 = x86.rsm.vfmadd231pd %21, %22, [%17] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %25 = x86.ri.add %17, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          x86.ms.vmovupd [%18], %24 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:          %26 = x86.ri.sub %23, 32 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %27 = x86.ri.sub %25, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %28 = x86.ri.add %18, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %26, %27, %28, %19, %20 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %29 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:        %30, %31, %32, %33, %34, %35 = x86_scf.for %36 : !x86.reg64<r10>  = %29 to 4 : si32 step 4 : si32 iter_args(%37 = %10, %38 = %11, %39 = %12, %40 = %13, %41 = %14) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:          %42 = x86.dm.vmovupd [%39] : (!x86.reg64<rdx>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:          %43 = x86.dm.vmovupd [%37] : (!x86.reg64<rdi>) -> !x86.avx2reg<ymm0>
// CHECK-NEXT:          %44 = x86.ri.add %37, 96 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %45 = x86.rsm.vfmadd231pd %42, %43, [%38] {broadcast} : (!x86.avx2reg<ymm31>, !x86.avx2reg<ymm0>, !x86.reg64<rsi>) -> !x86.avx2reg<ymm31>
// CHECK-NEXT:          %46 = x86.ri.add %38, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          x86.ms.vmovupd [%39], %45 : (!x86.reg64<rdx>, !x86.avx2reg<ymm31>) -> ()
// CHECK-NEXT:          %47 = x86.ri.sub %44, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %48 = x86.ri.sub %46, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %49 = x86.ri.add %39, 32 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:          x86_scf.yield %47, %48, %49, %40, %41 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:        }
// CHECK-NEXT:        %50 = x86.ri.sub %31, 96 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %51 = x86.ri.add %32, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %52 = x86.ri.add %33, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        x86_scf.yield %50, %51, %52, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  

x86_func.func @narrow_fsdbcst_remainder(
  %a: !x86.reg64<rdi>,
  %b: !x86.reg64<rsi>,
  %c: !x86.reg64<rdx>,
  %rbp: !x86.reg64<rbp>,
  %rsp: !x86.reg64<rsp>
) {
  %a_out, %b_out, %c_out, %rbp_out, %rsp_out = "xsmm.matmul"(%a, %b, %c, %rbp, %rsp) <{m = 12 : i64, n = 1 : i64, k = 1 : i64, lda = 12 : i64, ldb = 1 : i64, ldc = 12 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
  x86_func.ret
}
