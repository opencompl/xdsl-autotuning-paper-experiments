// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128 128 128 128 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128 128 128 128 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 18 : si32 step 6 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 128 : si32 step 64 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovaps [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %37 = x86.dm.vmovaps [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %38 = x86.dm.vmovaps [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %39 = x86.dm.vmovaps [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %40 = x86.dm.vmovaps [%33 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %41 = x86.dm.vmovaps [%33 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %42 = x86.dm.vmovaps [%33 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %43 = x86.dm.vmovaps [%33 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %44 = x86.dm.vmovaps [%33 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %45 = x86.dm.vmovaps [%33 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %46 = x86.dm.vmovaps [%33 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %47 = x86.dm.vmovaps [%33 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %48 = x86.dm.vmovaps [%33 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %49 = x86.dm.vmovaps [%33 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %50 = x86.dm.vmovaps [%33 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %51 = x86.dm.vmovaps [%33 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %52 = x86.dm.vmovaps [%33 + 2048] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %53 = x86.dm.vmovaps [%33 + 2112] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %54 = x86.dm.vmovaps [%33 + 2176] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %55 = x86.dm.vmovaps [%33 + 2240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %56 = x86.dm.vmovaps [%33 + 2560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %57 = x86.dm.vmovaps [%33 + 2624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %58 = x86.dm.vmovaps [%33 + 2688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %59 = x86.dm.vmovaps [%33 + 2752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %60 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90 = x86_scf.for %91 : !x86.reg64<r12>  = %60 to 128 : si32 step 4 : si32 iter_args(%92 = %31, %93 = %32, %94 = %33, %95 = %34, %96 = %35, %97 = %36, %98 = %37, %99 = %38, %100 = %39, %101 = %40, %102 = %41, %103 = %42, %104 = %43, %105 = %44, %106 = %45, %107 = %46, %108 = %47, %109 = %48, %110 = %49, %111 = %50, %112 = %51, %113 = %52, %114 = %53, %115 = %54, %116 = %55, %117 = %56, %118 = %57, %119 = %58, %120 = %59) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %121, %122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149 = "xsmm.matmul_k"(%92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120) <{m_blocking = 64 : i64, n_blocking = 6 : i64, k_blocking = 4 : i64, lda = 128 : i64, ldb = 128 : i64, datatype = f32, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 24>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 24>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %121, %122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %150 = x86.ri.sub %63, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovaps [%64], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 64], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 128], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 192], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 512], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 576], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 640], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 704], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1024], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1088], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1152], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1216], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1536], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1600], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1664], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 1728], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2048], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2112], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2176], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2240], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2560], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2624], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2688], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%64 + 2752], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %151 = x86.ri.add %64, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %152 = x86.ri.sub %62, 65280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %152, %150, %151, %65, %66 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %153 = x86.ri.add %27, 2560 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %154 = x86.ri.add %26, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %155 = x86.ri.sub %25, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %155, %154, %153, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %156 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %157, %158, %159, %160, %161, %162 = x86_scf.for %163 : !x86.reg64<r11>  = %156 to 38 : si32 step 5 : si32 iter_args(%164 = %12, %165 = %13, %166 = %14, %167 = %15, %168 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %169 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %170, %171, %172, %173, %174, %175 = x86_scf.for %176 : !x86.reg64<r10>  = %169 to 128 : si32 step 64 : si32 iter_args(%177 = %164, %178 = %165, %179 = %166, %180 = %167, %181 = %168) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %182 = x86.dm.vmovaps [%179] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %183 = x86.dm.vmovaps [%179 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %184 = x86.dm.vmovaps [%179 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %185 = x86.dm.vmovaps [%179 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %186 = x86.dm.vmovaps [%179 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %187 = x86.dm.vmovaps [%179 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %188 = x86.dm.vmovaps [%179 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %189 = x86.dm.vmovaps [%179 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %190 = x86.dm.vmovaps [%179 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %191 = x86.dm.vmovaps [%179 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %192 = x86.dm.vmovaps [%179 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %193 = x86.dm.vmovaps [%179 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %194 = x86.dm.vmovaps [%179 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %195 = x86.dm.vmovaps [%179 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %196 = x86.dm.vmovaps [%179 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %197 = x86.dm.vmovaps [%179 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %198 = x86.dm.vmovaps [%179 + 2048] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %199 = x86.dm.vmovaps [%179 + 2112] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %200 = x86.dm.vmovaps [%179 + 2176] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %201 = x86.dm.vmovaps [%179 + 2240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %202 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %203, %204, %205, %206, %207, %208, %209, %210, %211, %212, %213, %214, %215, %216, %217, %218, %219, %220, %221, %222, %223, %224, %225, %226, %227, %228 = x86_scf.for %229 : !x86.reg64<r12>  = %202 to 128 : si32 step 4 : si32 iter_args(%230 = %177, %231 = %178, %232 = %179, %233 = %180, %234 = %181, %235 = %182, %236 = %183, %237 = %184, %238 = %185, %239 = %186, %240 = %187, %241 = %188, %242 = %189, %243 = %190, %244 = %191, %245 = %192, %246 = %193, %247 = %194, %248 = %195, %249 = %196, %250 = %197, %251 = %198, %252 = %199, %253 = %200, %254 = %201) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %255, %256, %257, %258, %259, %260, %261, %262, %263, %264, %265, %266, %267, %268, %269, %270, %271, %272, %273, %274, %275, %276, %277, %278, %279 = "xsmm.matmul_k"(%230, %231, %232, %233, %234, %235, %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, %247, %248, %249, %250, %251, %252, %253, %254) <{m_blocking = 64 : i64, n_blocking = 5 : i64, k_blocking = 4 : i64, lda = 128 : i64, ldb = 128 : i64, datatype = f32, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %255, %256, %257, %258, %259, %260, %261, %262, %263, %264, %265, %266, %267, %268, %269, %270, %271, %272, %273, %274, %275, %276, %277, %278, %279 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %280 = x86.ri.sub %205, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovaps [%206], %209 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 64], %210 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 128], %211 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 192], %212 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 512], %213 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 576], %214 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 640], %215 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 704], %216 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1024], %217 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1088], %218 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1152], %219 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1216], %220 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1536], %221 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1600], %222 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1664], %223 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 1728], %224 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 2048], %225 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 2112], %226 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 2176], %227 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovaps [%206 + 2240], %228 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %281 = x86.ri.add %206, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %282 = x86.ri.sub %204, 65280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %282, %280, %281, %207, %208 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %283 = x86.ri.add %173, 2048 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %284 = x86.ri.add %172, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %285 = x86.ri.sub %171, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %285, %284, %283, %174, %175 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %286 = x86.ds.mov %161 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %287, %288 = x86.d.pop %286 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovaps zmm8, [rdx]
// CHECK-REGALLOC-NEXT:      vmovaps zmm9, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm10, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm11, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovaps zmm12, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovaps zmm13, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovaps zmm14, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovaps zmm15, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovaps zmm16, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovaps zmm17, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovaps zmm18, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovaps zmm19, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovaps zmm20, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovaps zmm21, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovaps zmm23, [rdx+1728]
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdx+2048]
// CHECK-REGALLOC-NEXT:      vmovaps zmm25, [rdx+2112]
// CHECK-REGALLOC-NEXT:      vmovaps zmm26, [rdx+2176]
// CHECK-REGALLOC-NEXT:      vmovaps zmm27, [rdx+2240]
// CHECK-REGALLOC-NEXT:      vmovaps zmm28, [rdx+2560]
// CHECK-REGALLOC-NEXT:      vmovaps zmm29, [rdx+2624]
// CHECK-REGALLOC-NEXT:      vmovaps zmm30, [rdx+2688]
// CHECK-REGALLOC-NEXT:      vmovaps zmm31, [rdx+2752]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovaps [rdx], zmm8
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+64], zmm9
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+128], zmm10
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+192], zmm11
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+512], zmm12
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+576], zmm13
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+640], zmm14
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+704], zmm15
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1024], zmm16
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1088], zmm17
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1152], zmm18
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1216], zmm19
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1536], zmm20
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1600], zmm21
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1664], zmm22
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1728], zmm23
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2048], zmm24
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2112], zmm25
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2176], zmm26
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2240], zmm27
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2560], zmm28
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2624], zmm29
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2688], zmm30
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2752], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 65280
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 2560
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      sub rdi, 512
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 18
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovaps zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovaps zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm15, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovaps zmm16, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovaps zmm17, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovaps zmm18, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovaps zmm19, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovaps zmm20, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovaps zmm21, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovaps zmm22, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovaps zmm23, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovaps zmm24, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovaps zmm25, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovaps zmm26, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovaps zmm27, [rdx+1728]
// CHECK-REGALLOC-NEXT:      vmovaps zmm28, [rdx+2048]
// CHECK-REGALLOC-NEXT:      vmovaps zmm29, [rdx+2112]
// CHECK-REGALLOC-NEXT:      vmovaps zmm30, [rdx+2176]
// CHECK-REGALLOC-NEXT:      vmovaps zmm31, [rdx+2240]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovaps zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovaps zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovaps zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovaps zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovaps zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 512
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovaps [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+192], zmm15
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+512], zmm16
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+576], zmm17
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+640], zmm18
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+704], zmm19
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1024], zmm20
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1088], zmm21
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1152], zmm22
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1216], zmm23
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1536], zmm24
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1600], zmm25
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1664], zmm26
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+1728], zmm27
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2048], zmm28
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2112], zmm29
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2176], zmm30
// CHECK-REGALLOC-NEXT:      vmovaps [rdx+2240], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 65280
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      add rdx, 2048
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      sub rdi, 512
// CHECK-REGALLOC-NEXT:      cmp rax, 38
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
