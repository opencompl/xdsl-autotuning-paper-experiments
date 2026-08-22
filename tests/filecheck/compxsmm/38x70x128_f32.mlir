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
// CHECK-NEXT:        %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88 = "xsmm.matmul_k"(%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59) <{m_blocking = 64 : i64, n_blocking = 6 : i64, k_blocking = 128 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 24>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 24>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %89 = x86.ri.sub %61, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%62], %65 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 64], %66 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 128], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 192], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 280], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 344], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 408], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 472], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 560], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 624], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 688], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 752], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 840], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 904], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 968], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1032], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1120], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1184], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1248], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1312], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1400], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1464], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1528], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%62 + 1592], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %90 = x86.ri.add %62, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %91 = x86.ri.sub %60, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %91, %89, %90, %63, %64 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %92 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %93 = x86.ks.kmovw %92 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %94 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %95, %96, %97, %98, %99, %100, %101 = x86_scf.for %102 : !x86.reg64<r10>  = %94 to 70 : si32 step 6 : si32 iter_args(%103 = %25, %104 = %26, %105 = %27, %106 = %28, %107 = %29, %108 = %93) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %109 = x86.dmk.vmovups[%105], %108 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %110 = x86.dmk.vmovups[%105 + 280], %108 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %111 = x86.dmk.vmovups[%105 + 560], %108 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %112 = x86.dmk.vmovups[%105 + 840], %108 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %113 = x86.dmk.vmovups[%105 + 1120], %108 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %114 = x86.dmk.vmovups[%105 + 1400], %108 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126 = "xsmm.matmul_k"(%103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114) <{m_blocking = 6 : i64, n_blocking = 6 : i64, k_blocking = 128 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 6>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 6>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %127 = x86.ri.sub %116, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovups[%117], %121, %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%117 + 280], %122, %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%117 + 560], %123, %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%117 + 840], %124, %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%117 + 1120], %125, %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%117 + 1400], %126, %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %128 = x86.ri.add %117, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %129 = x86.ri.sub %115, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %129, %127, %128, %118, %119, %120 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %130 = x86.ri.add %98, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %131 = x86.ri.add %97, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %132 = x86.ri.sub %96, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %132, %131, %130, %99, %100 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %133 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %134, %135, %136, %137, %138, %139 = x86_scf.for %140 : !x86.reg64<r11>  = %133 to 38 : si32 step 5 : si32 iter_args(%141 = %12, %142 = %13, %143 = %14, %144 = %15, %145 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %146 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %147, %148, %149, %150, %151, %152 = x86_scf.for %153 : !x86.reg64<r10>  = %146 to 64 : si32 step 64 : si32 iter_args(%154 = %141, %155 = %142, %156 = %143, %157 = %144, %158 = %145) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %159 = x86.dm.vmovups [%156] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %160 = x86.dm.vmovups [%156 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %161 = x86.dm.vmovups [%156 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %162 = x86.dm.vmovups [%156 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %163 = x86.dm.vmovups [%156 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %164 = x86.dm.vmovups [%156 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %165 = x86.dm.vmovups [%156 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %166 = x86.dm.vmovups [%156 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %167 = x86.dm.vmovups [%156 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %168 = x86.dm.vmovups [%156 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %169 = x86.dm.vmovups [%156 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %170 = x86.dm.vmovups [%156 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %171 = x86.dm.vmovups [%156 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %172 = x86.dm.vmovups [%156 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %173 = x86.dm.vmovups [%156 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %174 = x86.dm.vmovups [%156 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %175 = x86.dm.vmovups [%156 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %176 = x86.dm.vmovups [%156 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %177 = x86.dm.vmovups [%156 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %178 = x86.dm.vmovups [%156 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189, %190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203 = "xsmm.matmul_k"(%154, %155, %156, %157, %158, %159, %160, %161, %162, %163, %164, %165, %166, %167, %168, %169, %170, %171, %172, %173, %174, %175, %176, %177, %178) <{m_blocking = 64 : i64, n_blocking = 5 : i64, k_blocking = 128 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 20>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %204 = x86.ri.sub %180, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%181], %184 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 64], %185 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 128], %186 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 192], %187 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 280], %188 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 344], %189 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 408], %190 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 472], %191 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 560], %192 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 624], %193 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 688], %194 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 752], %195 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 840], %196 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 904], %197 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 968], %198 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 1032], %199 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 1120], %200 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 1184], %201 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 1248], %202 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%181 + 1312], %203 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %205 = x86.ri.add %181, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %206 = x86.ri.sub %179, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %206, %204, %205, %182, %183 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %207 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %208 = x86.ks.kmovw %207 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %209 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %210, %211, %212, %213, %214, %215, %216 = x86_scf.for %217 : !x86.reg64<r10>  = %209 to 70 : si32 step 6 : si32 iter_args(%218 = %148, %219 = %149, %220 = %150, %221 = %151, %222 = %152, %223 = %208) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %224 = x86.dmk.vmovups[%220], %223 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %225 = x86.dmk.vmovups[%220 + 280], %223 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %226 = x86.dmk.vmovups[%220 + 560], %223 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %227 = x86.dmk.vmovups[%220 + 840], %223 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %228 = x86.dmk.vmovups[%220 + 1120], %223 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %229, %230, %231, %232, %233, %234, %235, %236, %237, %238, %239 = "xsmm.matmul_k"(%218, %219, %220, %221, %222, %223, %224, %225, %226, %227, %228) <{m_blocking = 6 : i64, n_blocking = 5 : i64, k_blocking = 128 : i64, lda = 70 : i64, ldb = 128 : i64, datatype = f32, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %240 = x86.ri.sub %230, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovups[%231], %235, %234 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%231 + 280], %236, %234 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%231 + 560], %237, %234 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%231 + 840], %238, %234 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%231 + 1120], %239, %234 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %241 = x86.ri.add %231, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %242 = x86.ri.sub %229, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %242, %240, %241, %232, %233, %234 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %243 = x86.ri.add %213, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %244 = x86.ri.add %212, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %245 = x86.ri.sub %211, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %245, %244, %243, %214, %215 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %246 = x86.ds.mov %138 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %247, %248 = x86.d.pop %246 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
