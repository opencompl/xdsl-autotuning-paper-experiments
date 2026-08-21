// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 5 : si32 step 5 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 32 : si32 step 32 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovupd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %37 = x86.dm.vmovupd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %38 = x86.dm.vmovupd [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %39 = x86.dm.vmovupd [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %40 = x86.dm.vmovupd [%33 + 272] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %41 = x86.dm.vmovupd [%33 + 336] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %42 = x86.dm.vmovupd [%33 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %43 = x86.dm.vmovupd [%33 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %44 = x86.dm.vmovupd [%33 + 544] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %45 = x86.dm.vmovupd [%33 + 608] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %46 = x86.dm.vmovupd [%33 + 672] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %47 = x86.dm.vmovupd [%33 + 736] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %48 = x86.dm.vmovupd [%33 + 816] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %49 = x86.dm.vmovupd [%33 + 880] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %50 = x86.dm.vmovupd [%33 + 944] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %51 = x86.dm.vmovupd [%33 + 1008] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %52 = x86.dm.vmovupd [%33 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %53 = x86.dm.vmovupd [%33 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %54 = x86.dm.vmovupd [%33 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %55 = x86.dm.vmovupd [%33 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80 = "xsmm.matmul_k"(%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55) <{m_blocking = 32 : i64, n_blocking = 5 : i64, k_blocking = 16 : i64, lda = 34 : i64, ldb = 16 : i64, datatype = f64, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %81 = x86.ri.sub %57, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovupd [%58], %61 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 64], %62 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 128], %63 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 192], %64 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 272], %65 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 336], %66 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 400], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 464], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 544], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 608], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 672], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 736], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 816], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 880], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 944], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 1008], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 1088], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 1152], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 1216], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%58 + 1280], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %82 = x86.ri.add %58, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %83 = x86.ri.sub %56, 4096 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %83, %81, %82, %59, %60 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %84 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %85 = x86.ks.kmovb %84 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %86 = x86.di.mov 32 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %87, %88, %89, %90, %91, %92, %93 = x86_scf.for %94 : !x86.reg64<r10>  = %86 to 34 : si32 step 2 : si32 iter_args(%95 = %25, %96 = %26, %97 = %27, %98 = %28, %99 = %29, %100 = %85) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %101 = x86.dmk.vmovupd[%97], %100 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %102 = x86.dmk.vmovupd[%97 + 272], %100 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %103 = x86.dmk.vmovupd[%97 + 544], %100 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %104 = x86.dmk.vmovupd[%97 + 816], %100 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %105 = x86.dmk.vmovupd[%97 + 1088], %100 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116 = "xsmm.matmul_k"(%95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105) <{m_blocking = 2 : i64, n_blocking = 5 : i64, k_blocking = 16 : i64, lda = 34 : i64, ldb = 16 : i64, datatype = f64, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %117 = x86.ri.sub %107, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovupd[%108], %112, %111 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%108 + 272], %113, %111 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%108 + 544], %114, %111 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%108 + 816], %115, %111 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%108 + 1088], %116, %111 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %118 = x86.ri.add %108, 16 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %119 = x86.ri.sub %106, 4336 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %119, %117, %118, %109, %110, %111 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %120 = x86.ri.add %90, 1088 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %121 = x86.ri.add %89, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %122 = x86.ri.sub %88, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %122, %121, %120, %91, %92 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %123 = x86.ds.mov %15 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %124, %125 = x86.d.pop %123 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-NEXT:      sub rsp, 192
// CHECK-REGALLOC-NEXT:      mov r10, -64
// CHECK-REGALLOC-NEXT:      and rsp, r10
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 32
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm15, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovupd zmm16, [rdx+272]
// CHECK-REGALLOC-NEXT:      vmovupd zmm17, [rdx+336]
// CHECK-REGALLOC-NEXT:      vmovupd zmm18, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovupd zmm19, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdx+544]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdx+608]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdx+672]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdx+736]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdx+816]
// CHECK-REGALLOC-NEXT:      vmovupd zmm25, [rdx+880]
// CHECK-REGALLOC-NEXT:      vmovupd zmm26, [rdx+944]
// CHECK-REGALLOC-NEXT:      vmovupd zmm27, [rdx+1008]
// CHECK-REGALLOC-NEXT:      vmovupd zmm28, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovupd zmm29, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovupd zmm30, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovupd zmm31, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovupd [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+192], zmm15
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+272], zmm16
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+336], zmm17
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+400], zmm18
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+464], zmm19
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+544], zmm20
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+608], zmm21
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+672], zmm22
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+736], zmm23
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+816], zmm24
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+880], zmm25
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+944], zmm26
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1008], zmm27
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1088], zmm28
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1152], zmm29
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1216], zmm30
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1280], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 4096
// CHECK-REGALLOC-NEXT:      cmp rcx, 32
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovb k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 32
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 2
// CHECK-REGALLOC-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+272]
// CHECK-REGALLOC-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+544]
// CHECK-REGALLOC-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+816]
// CHECK-REGALLOC-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-REGALLOC-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-REGALLOC-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-REGALLOC-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-REGALLOC-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-REGALLOC-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-REGALLOC-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-REGALLOC-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-REGALLOC-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-REGALLOC-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-REGALLOC-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-REGALLOC-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-REGALLOC-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-REGALLOC-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-REGALLOC-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+272]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+128]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+256]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+384]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+512]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+544]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+8]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+136]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+264]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+392]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+520]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+816]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+16]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+144]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+272]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+400]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+528]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+1088]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+24]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+152]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+280]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+408]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+536]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+1360]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+32]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+160]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+288]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+416]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+544]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+1632]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+40]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+168]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+296]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+424]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+552]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+1904]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+48]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+176]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+304]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+432]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+560]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+2176]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+56]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+184]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+312]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+440]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+568]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+2448]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+64]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+192]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+320]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+448]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+576]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+2720]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+72]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+200]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+328]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+456]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+584]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+2992]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+80]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+208]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+336]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+464]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+592]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+3264]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+88]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+216]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+344]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+472]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+600]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+3536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+96]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+224]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+352]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+480]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+608]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+3808]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+104]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+232]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+360]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+488]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+616]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+4080]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+112]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+240]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+368]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+496]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+624]{1to8}
// CHECK-REGALLOC-NEXT:      add rdi, 4352
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+120]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+248]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+376]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+504]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+632]{1to8}
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm22, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm23, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm24, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm25, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm26, zmm31
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm17, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm18, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm19, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm20, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm21, zmm31
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm12, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm13, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm14, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm15, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm16, zmm31
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+272] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+544] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+816] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1088] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 16
// CHECK-REGALLOC-NEXT:      sub rdi, 4336
// CHECK-REGALLOC-NEXT:      cmp rcx, 34
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 1088
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      sub rdi, 272
// CHECK-REGALLOC-NEXT:      cmp rax, 5
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
