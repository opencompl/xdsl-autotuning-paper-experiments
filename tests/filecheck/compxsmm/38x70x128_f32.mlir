// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

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
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 64 : si32 step 64 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovups [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %37 = x86.dm.vmovups [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %38 = x86.dm.vmovups [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %39 = x86.dm.vmovups [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %40 = x86.dm.vmovups [%33 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %41 = x86.dm.vmovups [%33 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %42 = x86.dm.vmovups [%33 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %43 = x86.dm.vmovups [%33 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %44 = x86.dm.vmovups [%33 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %45 = x86.dm.vmovups [%33 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %46 = x86.dm.vmovups [%33 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %47 = x86.dm.vmovups [%33 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %48 = x86.dm.vmovups [%33 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %49 = x86.dm.vmovups [%33 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %50 = x86.dm.vmovups [%33 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %51 = x86.dm.vmovups [%33 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %52 = x86.dm.vmovups [%33 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %53 = x86.dm.vmovups [%33 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %54 = x86.dm.vmovups [%33 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %55 = x86.dm.vmovups [%33 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %56 = x86.dm.vmovups [%33 + 1400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %57 = x86.dm.vmovups [%33 + 1464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %58 = x86.dm.vmovups [%33 + 1528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %59 = x86.dm.vmovups [%33 + 1592] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %60 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90 = x86_scf.for %91 : !x86.reg64<r12>  = %60 to 128 : si32 step 4 : si32 iter_args(%92 = %31, %93 = %32, %94 = %33, %95 = %34, %96 = %35, %97 = %36, %98 = %37, %99 = %38, %100 = %39, %101 = %40, %102 = %41, %103 = %42, %104 = %43, %105 = %44, %106 = %45, %107 = %46, %108 = %47, %109 = %48, %110 = %49, %111 = %50, %112 = %51, %113 = %52, %114 = %53, %115 = %54, %116 = %55, %117 = %56, %118 = %57, %119 = %58, %120 = %59) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %121, %122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149 = "xsmm.matmul_k"(%92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120) <{m_blocking = 64 : i64, n_blocking = 6 : i64, k_blocking = 4 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 24>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 24>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %121, %122, %123, %124, %125, %126, %127, %128, %129, %130, %131, %132, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148, %149 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %150 = x86.ri.sub %63, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%64], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 64], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 128], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 192], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 280], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 344], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 408], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 472], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 560], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 624], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 688], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 752], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 840], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 904], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 968], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1032], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1120], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1184], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1248], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1312], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1400], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1464], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1528], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%64 + 1592], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %151 = x86.ri.add %64, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %152 = x86.ri.sub %62, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %152, %150, %151, %65, %66 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %153 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %154 = x86.ks.kmovw %153 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %155 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %156, %157, %158, %159, %160, %161, %162 = x86_scf.for %163 : !x86.reg64<r10>  = %155 to 70 : si32 step 6 : si32 iter_args(%164 = %25, %165 = %26, %166 = %27, %167 = %28, %168 = %29, %169 = %154) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %170 = x86.dmk.vmovups[%166], %169 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %171 = x86.dmk.vmovups[%166 + 280], %169 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %172 = x86.dmk.vmovups[%166 + 560], %169 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %173 = x86.dmk.vmovups[%166 + 840], %169 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %174 = x86.dmk.vmovups[%166 + 1120], %169 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %175 = x86.dmk.vmovups[%166 + 1400], %169 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %176 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %177, %178, %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189 = x86_scf.for %190 : !x86.reg64<r12>  = %176 to 128 : si32 step 4 : si32 iter_args(%191 = %164, %192 = %165, %193 = %166, %194 = %167, %195 = %168, %196 = %169, %197 = %170, %198 = %171, %199 = %172, %200 = %173, %201 = %174, %202 = %175) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %203, %204, %205, %206, %207, %208, %209, %210, %211, %212, %213, %214 = "xsmm.matmul_k"(%191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202) <{m_blocking = 6 : i64, n_blocking = 6 : i64, k_blocking = 4 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 6>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 6>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %203, %204, %205, %206, %207, %208, %209, %210, %211, %212, %213, %214 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %215 = x86.ri.sub %179, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovups[%180], %184, %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%180 + 280], %185, %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%180 + 560], %186, %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%180 + 840], %187, %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%180 + 1120], %188, %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%180 + 1400], %189, %183 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %216 = x86.ri.add %180, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %217 = x86.ri.sub %178, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %217, %215, %216, %181, %182, %183 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %218 = x86.ri.add %159, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %219 = x86.ri.add %158, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %220 = x86.ri.sub %157, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %220, %219, %218, %160, %161 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %221 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %222, %223, %224, %225, %226, %227 = x86_scf.for %228 : !x86.reg64<r11>  = %221 to 38 : si32 step 5 : si32 iter_args(%229 = %12, %230 = %13, %231 = %14, %232 = %15, %233 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %234 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %235, %236, %237, %238, %239, %240 = x86_scf.for %241 : !x86.reg64<r10>  = %234 to 64 : si32 step 64 : si32 iter_args(%242 = %229, %243 = %230, %244 = %231, %245 = %232, %246 = %233) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %247 = x86.dm.vmovups [%244] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %248 = x86.dm.vmovups [%244 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %249 = x86.dm.vmovups [%244 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %250 = x86.dm.vmovups [%244 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %251 = x86.dm.vmovups [%244 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %252 = x86.dm.vmovups [%244 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %253 = x86.dm.vmovups [%244 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %254 = x86.dm.vmovups [%244 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %255 = x86.dm.vmovups [%244 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %256 = x86.dm.vmovups [%244 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %257 = x86.dm.vmovups [%244 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %258 = x86.dm.vmovups [%244 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %259 = x86.dm.vmovups [%244 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %260 = x86.dm.vmovups [%244 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %261 = x86.dm.vmovups [%244 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %262 = x86.dm.vmovups [%244 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %263 = x86.dm.vmovups [%244 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %264 = x86.dm.vmovups [%244 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %265 = x86.dm.vmovups [%244 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %266 = x86.dm.vmovups [%244 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %267 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %268, %269, %270, %271, %272, %273, %274, %275, %276, %277, %278, %279, %280, %281, %282, %283, %284, %285, %286, %287, %288, %289, %290, %291, %292, %293 = x86_scf.for %294 : !x86.reg64<r12>  = %267 to 128 : si32 step 4 : si32 iter_args(%295 = %242, %296 = %243, %297 = %244, %298 = %245, %299 = %246, %300 = %247, %301 = %248, %302 = %249, %303 = %250, %304 = %251, %305 = %252, %306 = %253, %307 = %254, %308 = %255, %309 = %256, %310 = %257, %311 = %258, %312 = %259, %313 = %260, %314 = %261, %315 = %262, %316 = %263, %317 = %264, %318 = %265, %319 = %266) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %320, %321, %322, %323, %324, %325, %326, %327, %328, %329, %330, %331, %332, %333, %334, %335, %336, %337, %338, %339, %340, %341, %342, %343, %344 = "xsmm.matmul_k"(%295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %307, %308, %309, %310, %311, %312, %313, %314, %315, %316, %317, %318, %319) <{m_blocking = 64 : i64, n_blocking = 5 : i64, k_blocking = 4 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %320, %321, %322, %323, %324, %325, %326, %327, %328, %329, %330, %331, %332, %333, %334, %335, %336, %337, %338, %339, %340, %341, %342, %343, %344 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %345 = x86.ri.sub %270, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%271], %274 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 64], %275 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 128], %276 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 192], %277 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 280], %278 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 344], %279 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 408], %280 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 472], %281 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 560], %282 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 624], %283 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 688], %284 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 752], %285 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 840], %286 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 904], %287 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 968], %288 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 1032], %289 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 1120], %290 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 1184], %291 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 1248], %292 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%271 + 1312], %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %346 = x86.ri.add %271, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %347 = x86.ri.sub %269, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %347, %345, %346, %272, %273 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %348 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %349 = x86.ks.kmovw %348 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %350 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %351, %352, %353, %354, %355, %356, %357 = x86_scf.for %358 : !x86.reg64<r10>  = %350 to 70 : si32 step 6 : si32 iter_args(%359 = %236, %360 = %237, %361 = %238, %362 = %239, %363 = %240, %364 = %349) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %365 = x86.dmk.vmovups[%361], %364 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %366 = x86.dmk.vmovups[%361 + 280], %364 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %367 = x86.dmk.vmovups[%361 + 560], %364 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %368 = x86.dmk.vmovups[%361 + 840], %364 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %369 = x86.dmk.vmovups[%361 + 1120], %364 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %370 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %371, %372, %373, %374, %375, %376, %377, %378, %379, %380, %381, %382 = x86_scf.for %383 : !x86.reg64<r12>  = %370 to 128 : si32 step 4 : si32 iter_args(%384 = %359, %385 = %360, %386 = %361, %387 = %362, %388 = %363, %389 = %364, %390 = %365, %391 = %366, %392 = %367, %393 = %368, %394 = %369) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %395, %396, %397, %398, %399, %400, %401, %402, %403, %404, %405 = "xsmm.matmul_k"(%384, %385, %386, %387, %388, %389, %390, %391, %392, %393, %394) <{m_blocking = 6 : i64, n_blocking = 5 : i64, k_blocking = 4 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:          x86_scf.yield %395, %396, %397, %398, %399, %400, %401, %402, %403, %404, %405 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %406 = x86.ri.sub %373, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovups[%374], %378, %377 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%374 + 280], %379, %377 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%374 + 560], %380, %377 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%374 + 840], %381, %377 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%374 + 1120], %382, %377 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %407 = x86.ri.add %374, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %408 = x86.ri.sub %372, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %408, %406, %407, %375, %376, %377 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %409 = x86.ri.add %354, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %410 = x86.ri.add %353, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %411 = x86.ri.sub %352, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %411, %410, %409, %355, %356 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %412 = x86.ds.mov %226 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %413, %414 = x86.d.pop %412 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm11, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+344]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+408]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+472]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+624]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+688]
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx+752]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+904]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+968]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdx+1032]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+1184]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+1248]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdx+1312]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+1400]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+1464]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+1528]
// CHECK-REGALLOC-NEXT:      vmovups zmm31, [rdx+1592]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+344], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+408], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+472], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+624], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+688], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+752], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+904], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+968], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1032], zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1184], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1248], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1312], zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1400], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1464], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1528], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1592], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 35584
// CHECK-REGALLOC-NEXT:      cmp rcx, 64
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rcx, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 64
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rcx, 6
// CHECK-REGALLOC-NEXT:      vmovups zmm26 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm28 {k1}{z}, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm29 {k1}{z}, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm30 {k1}{z}, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1400]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-REGALLOC-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-REGALLOC-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-REGALLOC-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-REGALLOC-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-REGALLOC-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm0, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, [rsi+512]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+1024]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1536]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+2048]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2560]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+4]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+516]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+1028]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+1540]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+2052]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+2564]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm0, [rsi+8]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, [rsi+520]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+1032]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1544]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+2056]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2568]{1to16}
// CHECK-REGALLOC-NEXT:      add rdi, 1120
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+12]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+524]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+1036]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+1548]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+2060]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+2572]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 16
// CHECK-REGALLOC-NEXT:      vaddps zmm26, zmm20, zmm26
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm21, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm22, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm23, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm24, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm25, zmm31
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1400] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 35816
// CHECK-REGALLOC-NEXT:      cmp rcx, 70
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      add rdx, 1400
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      mov rax, 18
// CHECK-REGALLOC-NEXT:  scf_body_9_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_6_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+344]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+408]
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx+472]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+624]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+688]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdx+752]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+904]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+968]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdx+1032]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+1184]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+1248]
// CHECK-REGALLOC-NEXT:      vmovups zmm31, [rdx+1312]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+344], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+408], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+472], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+624], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+688], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+752], zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+904], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+968], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1032], zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1184], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1248], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1312], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 35584
// CHECK-REGALLOC-NEXT:      cmp rcx, 64
// CHECK-REGALLOC-NEXT:      jl scf_body_6_for
// CHECK-REGALLOC-NEXT:      mov rcx, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 64
// CHECK-REGALLOC-NEXT:  scf_body_8_for:
// CHECK-REGALLOC-NEXT:      add rcx, 6
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm28 {k1}{z}, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm29 {k1}{z}, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm30 {k1}{z}, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1120]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_7_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
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
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, [rsi+512]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, [rsi+1024]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm1, [rsi+1536]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, [rsi+2048]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm0, [rsi+4]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, [rsi+516]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, [rsi+1028]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, [rsi+1540]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm0, [rsi+2052]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, [rsi+8]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm1, [rsi+520]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, [rsi+1032]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+1544]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+2056]{1to16}
// CHECK-REGALLOC-NEXT:      add rdi, 1120
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, [rsi+12]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, [rsi+524]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm0, [rsi+1036]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, [rsi+1548]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, [rsi+2060]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 16
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm22, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm23, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm24, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm25, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm26, zmm31
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm17, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm18, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm19, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm20, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm21, zmm31
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm12, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm13, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm14, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm15, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm16, zmm31
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_7_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 35816
// CHECK-REGALLOC-NEXT:      cmp rcx, 70
// CHECK-REGALLOC-NEXT:      jl scf_body_8_for
// CHECK-REGALLOC-NEXT:      add rdx, 1120
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      cmp rax, 38
// CHECK-REGALLOC-NEXT:      jl scf_body_9_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
