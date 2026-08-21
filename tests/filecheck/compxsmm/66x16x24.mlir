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
// CHECK-NEXT:        %64 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98 = x86_scf.for %99 : !x86.reg64<r12>  = %64 to 24 : si32 step 4 : si32 iter_args(%100 = %31, %101 = %32, %102 = %33, %103 = %34, %104 = %35, %105 = %36, %106 = %37, %107 = %38, %108 = %39, %109 = %40, %110 = %41, %111 = %42, %112 = %43, %113 = %44, %114 = %45, %115 = %46, %116 = %47, %117 = %48, %118 = %49, %119 = %50, %120 = %51, %121 = %52, %122 = %53, %123 = %54, %124 = %55, %125 = %56, %126 = %57, %127 = %58, %128 = %59, %129 = %60, %130 = %61, %131 = %62, %132 = %63) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150, %151, %152, %153, %154, %155, %156, %157, %158, %159, %160, %161, %162, %163, %164, %165 = "xsmm.matmul_k"(%100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132) <{m_blocking = 16 : i64, n_blocking = 14 : i64, k_blocking = 4 : i64, lda = 16 : i64, ldb = 24 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 28>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 28>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149, %150, %151, %152, %153, %154, %155, %156, %157, %158, %159, %160, %161, %162, %163, %164, %165 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %166 = x86.ri.sub %67, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%68], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm4>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 64], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm5>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 128], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 192], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 256], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 320], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 384], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 448], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 512], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 576], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 640], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 704], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 768], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 832], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 896], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 960], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1024], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1088], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1152], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1216], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1280], %91 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1344], %92 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1408], %93 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1472], %94 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1536], %95 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1600], %96 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1664], %97 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%68 + 1728], %98 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %167 = x86.ri.add %68, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %168 = x86.ri.sub %66, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %168, %166, %167, %69, %70 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %169 = x86.ri.add %27, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %170 = x86.ri.add %26, 2688 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %171 = x86.ri.sub %25, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %171, %170, %169, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %172 = x86.di.mov 14 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %173, %174, %175, %176, %177, %178 = x86_scf.for %179 : !x86.reg64<r11>  = %172 to 66 : si32 step 13 : si32 iter_args(%180 = %12, %181 = %13, %182 = %14, %183 = %15, %184 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %185 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %186, %187, %188, %189, %190, %191 = x86_scf.for %192 : !x86.reg64<r10>  = %185 to 16 : si32 step 16 : si32 iter_args(%193 = %180, %194 = %181, %195 = %182, %196 = %183, %197 = %184) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %198 = x86.dm.vmovapd [%195] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %199 = x86.dm.vmovapd [%195 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %200 = x86.dm.vmovapd [%195 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %201 = x86.dm.vmovapd [%195 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %202 = x86.dm.vmovapd [%195 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %203 = x86.dm.vmovapd [%195 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %204 = x86.dm.vmovapd [%195 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %205 = x86.dm.vmovapd [%195 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %206 = x86.dm.vmovapd [%195 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %207 = x86.dm.vmovapd [%195 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %208 = x86.dm.vmovapd [%195 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %209 = x86.dm.vmovapd [%195 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %210 = x86.dm.vmovapd [%195 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %211 = x86.dm.vmovapd [%195 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %212 = x86.dm.vmovapd [%195 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %213 = x86.dm.vmovapd [%195 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %214 = x86.dm.vmovapd [%195 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %215 = x86.dm.vmovapd [%195 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %216 = x86.dm.vmovapd [%195 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %217 = x86.dm.vmovapd [%195 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %218 = x86.dm.vmovapd [%195 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %219 = x86.dm.vmovapd [%195 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %220 = x86.dm.vmovapd [%195 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %221 = x86.dm.vmovapd [%195 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %222 = x86.dm.vmovapd [%195 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %223 = x86.dm.vmovapd [%195 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %224 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %225, %226, %227, %228, %229, %230, %231, %232, %233, %234, %235, %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, %247, %248, %249, %250, %251, %252, %253, %254, %255, %256 = x86_scf.for %257 : !x86.reg64<r12>  = %224 to 24 : si32 step 4 : si32 iter_args(%258 = %193, %259 = %194, %260 = %195, %261 = %196, %262 = %197, %263 = %198, %264 = %199, %265 = %200, %266 = %201, %267 = %202, %268 = %203, %269 = %204, %270 = %205, %271 = %206, %272 = %207, %273 = %208, %274 = %209, %275 = %210, %276 = %211, %277 = %212, %278 = %213, %279 = %214, %280 = %215, %281 = %216, %282 = %217, %283 = %218, %284 = %219, %285 = %220, %286 = %221, %287 = %222, %288 = %223) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %307, %308, %309, %310, %311, %312, %313, %314, %315, %316, %317, %318, %319 = "xsmm.matmul_k"(%258, %259, %260, %261, %262, %263, %264, %265, %266, %267, %268, %269, %270, %271, %272, %273, %274, %275, %276, %277, %278, %279, %280, %281, %282, %283, %284, %285, %286, %287, %288) <{m_blocking = 16 : i64, n_blocking = 13 : i64, k_blocking = 4 : i64, lda = 16 : i64, ldb = 24 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 26>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 26>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %307, %308, %309, %310, %311, %312, %313, %314, %315, %316, %317, %318, %319 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %320 = x86.ri.sub %227, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%228], %231 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 64], %232 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 128], %233 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 192], %234 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 256], %235 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 320], %236 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 384], %237 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 448], %238 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 512], %239 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 576], %240 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 640], %241 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 704], %242 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 768], %243 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 832], %244 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 896], %245 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 960], %246 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1024], %247 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1088], %248 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1152], %249 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1216], %250 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1280], %251 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1344], %252 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1408], %253 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1472], %254 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1536], %255 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%228 + 1600], %256 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %321 = x86.ri.add %228, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %322 = x86.ri.sub %226, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %322, %320, %321, %229, %230 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %323 = x86.ri.add %189, 1536 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %324 = x86.ri.add %188, 2496 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %325 = x86.ri.sub %187, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %325, %324, %323, %190, %191 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %326 = x86.ds.mov %177 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %327, %328 = x86.d.pop %326 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
