// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 14 : si32 step 14 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 16 : si32 step 16 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovapd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %37 = x86.dm.vmovapd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:        %38 = x86.dm.vmovapd [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %39 = x86.dm.vmovapd [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %40 = x86.dm.vmovapd [%33 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %41 = x86.dm.vmovapd [%33 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %42 = x86.dm.vmovapd [%33 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %43 = x86.dm.vmovapd [%33 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %44 = x86.dm.vmovapd [%33 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %45 = x86.dm.vmovapd [%33 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %46 = x86.dm.vmovapd [%33 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %47 = x86.dm.vmovapd [%33 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %48 = x86.dm.vmovapd [%33 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %49 = x86.dm.vmovapd [%33 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %50 = x86.dm.vmovapd [%33 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %51 = x86.dm.vmovapd [%33 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %52 = x86.dm.vmovapd [%33 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %53 = x86.dm.vmovapd [%33 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %54 = x86.dm.vmovapd [%33 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %55 = x86.dm.vmovapd [%33 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %56 = x86.dm.vmovapd [%33 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %57 = x86.dm.vmovapd [%33 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %58 = x86.dm.vmovapd [%33 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %59 = x86.dm.vmovapd [%33 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %60 = x86.dm.vmovapd [%33 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %61 = x86.dm.vmovapd [%33 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %62 = x86.dm.vmovapd [%33 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %63 = x86.dm.vmovapd [%33 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96 = "xsmm.matmul_k"(%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63) <{m_blocking = 16 : i64, n_blocking = 14 : i64, k_blocking = 24 : i64, lda = 16 : i64, ldb = 24 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 28>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 28>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %97 = x86.ri.sub %65, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%66], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm4>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 64], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm5>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 128], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 192], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 256], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 320], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 384], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 448], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 512], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 576], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 640], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 704], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 768], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 832], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 896], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 960], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1024], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1088], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1152], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1216], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1280], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1344], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1408], %91 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1472], %92 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1536], %93 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1600], %94 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1664], %95 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%66 + 1728], %96 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %98 = x86.ri.add %66, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %99 = x86.ri.sub %64, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %99, %97, %98, %67, %68 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %100 = x86.ri.add %27, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %101 = x86.ri.add %26, 2688 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %102 = x86.ri.sub %25, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %102, %101, %100, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %103 = x86.di.mov 14 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %104, %105, %106, %107, %108, %109 = x86_scf.for %110 : !x86.reg64<r11>  = %103 to 66 : si32 step 13 : si32 iter_args(%111 = %12, %112 = %13, %113 = %14, %114 = %15, %115 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %116 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %117, %118, %119, %120, %121, %122 = x86_scf.for %123 : !x86.reg64<r10>  = %116 to 16 : si32 step 16 : si32 iter_args(%124 = %111, %125 = %112, %126 = %113, %127 = %114, %128 = %115) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %129 = x86.dm.vmovapd [%126] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %130 = x86.dm.vmovapd [%126 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %131 = x86.dm.vmovapd [%126 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %132 = x86.dm.vmovapd [%126 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %133 = x86.dm.vmovapd [%126 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %134 = x86.dm.vmovapd [%126 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %135 = x86.dm.vmovapd [%126 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %136 = x86.dm.vmovapd [%126 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %137 = x86.dm.vmovapd [%126 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %138 = x86.dm.vmovapd [%126 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %139 = x86.dm.vmovapd [%126 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %140 = x86.dm.vmovapd [%126 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %141 = x86.dm.vmovapd [%126 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %142 = x86.dm.vmovapd [%126 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %143 = x86.dm.vmovapd [%126 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %144 = x86.dm.vmovapd [%126 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %145 = x86.dm.vmovapd [%126 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %146 = x86.dm.vmovapd [%126 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %147 = x86.dm.vmovapd [%126 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %148 = x86.dm.vmovapd [%126 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %149 = x86.dm.vmovapd [%126 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %150 = x86.dm.vmovapd [%126 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %151 = x86.dm.vmovapd [%126 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %152 = x86.dm.vmovapd [%126 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %153 = x86.dm.vmovapd [%126 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %154 = x86.dm.vmovapd [%126 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %155, %156, %157, %158, %159, %160, %161, %162, %163, %164, %165, %166, %167, %168, %169, %170, %171, %172, %173, %174, %175, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185 = "xsmm.matmul_k"(%124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150, %151, %152, %153, %154) <{m_blocking = 16 : i64, n_blocking = 13 : i64, k_blocking = 24 : i64, lda = 16 : i64, ldb = 24 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 26>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 26>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %186 = x86.ri.sub %156, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%157], %160 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 64], %161 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 128], %162 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 192], %163 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 256], %164 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 320], %165 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 384], %166 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 448], %167 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 512], %168 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 576], %169 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 640], %170 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 704], %171 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 768], %172 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 832], %173 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 896], %174 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 960], %175 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1024], %176 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1088], %177 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1152], %178 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1216], %179 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1280], %180 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1344], %181 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1408], %182 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1472], %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1536], %184 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%157 + 1600], %185 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %187 = x86.ri.add %157, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %188 = x86.ri.sub %155, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %188, %186, %187, %158, %159 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %189 = x86.ri.add %120, 1536 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %190 = x86.ri.add %119, 2496 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %191 = x86.ri.sub %118, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %191, %190, %189, %121, %122 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %192 = x86.ds.mov %108 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %193, %194 = x86.d.pop %192 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      push rbx
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-NEXT:      sub rsp, 192
// CHECK-REGALLOC-NEXT:      mov r10, -64
// CHECK-REGALLOC-NEXT:      and rsp, r10
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 14
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovapd zmm23, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovapd zmm24, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovapd zmm25, [rdx+1344]
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx+1408]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdx+1472]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+1728]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      cmp rbx, 24
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 192
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm17
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1152], zmm22
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1216], zmm23
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1280], zmm24
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1344], zmm25
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1408], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1472], zmm27
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1536], zmm28
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1600], zmm29
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1664], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1728], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 2944
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 1664
// CHECK-REGALLOC-NEXT:      add rsi, 2688
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp rax, 14
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 14
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 13
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm23, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm24, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovapd zmm25, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdx+1344]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+1408]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdx+1472]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+1600]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 24
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      sub rsi, 192
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm17
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm22
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm23
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1152], zmm24
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1216], zmm25
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1280], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1344], zmm27
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1408], zmm28
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1472], zmm29
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1536], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1600], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 2944
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      add rdx, 1536
// CHECK-REGALLOC-NEXT:      add rsi, 2496
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp rax, 66
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
