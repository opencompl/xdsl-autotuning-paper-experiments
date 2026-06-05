// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && cat %t | filecheck %s

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15 = x86_scf.for %16 : !x86.reg64<r11>  = %10 to 18 : si32 step 6 : si32 iter_args(%17 = %0, %18 = %1, %19 = %2, %20 = %6, %21 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %22 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %23 = x86.ks.kmovw %22 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %24 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %25, %26, %27, %28, %29, %30 = x86_scf.for %31 : !x86.reg64<r10>  = %24 to 50 : si32 step 50 : si32 iter_args(%32 = %17, %33 = %18, %34 = %19, %35 = %20, %36 = %21, %37 = %23) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %38 = x86.dm.vmovups [%34] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %39 = x86.dm.vmovups [%34 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %40 = x86.dm.vmovups [%34 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %41 = x86.dmk.vmovups[%34 + 192], %37 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %42 = x86.dm.vmovups [%34 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %43 = x86.dm.vmovups [%34 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %44 = x86.dm.vmovups [%34 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %45 = x86.dmk.vmovups[%34 + 392], %37 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %46 = x86.dm.vmovups [%34 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %47 = x86.dm.vmovups [%34 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %48 = x86.dm.vmovups [%34 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %49 = x86.dmk.vmovups[%34 + 592], %37 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %50 = x86.dm.vmovups [%34 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %51 = x86.dm.vmovups [%34 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %52 = x86.dm.vmovups [%34 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %53 = x86.dmk.vmovups[%34 + 792], %37 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %54 = x86.dm.vmovups [%34 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %55 = x86.dm.vmovups [%34 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %56 = x86.dm.vmovups [%34 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %57 = x86.dmk.vmovups[%34 + 992], %37 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %58 = x86.dm.vmovups [%34 + 1000] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %59 = x86.dm.vmovups [%34 + 1064] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %60 = x86.dm.vmovups [%34 + 1128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %61 = x86.dmk.vmovups[%34 + 1192], %37 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %62 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92 = x86_scf.for %93 : !x86.reg64<r12>  = %62 to 128 : si32 step 4 : si32 iter_args(%94 = %32, %95 = %33, %96 = %34, %97 = %35, %98 = %36, %99 = %37, %100 = %38, %101 = %39, %102 = %40, %103 = %41, %104 = %42, %105 = %43, %106 = %44, %107 = %45, %108 = %46, %109 = %47, %110 = %48, %111 = %49, %112 = %50, %113 = %51, %114 = %52, %115 = %53, %116 = %54, %117 = %55, %118 = %56, %119 = %57, %120 = %58, %121 = %59, %122 = %60, %123 = %61) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %124 = x86.dm.vmovups [%94] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %125 = x86.dm.vmovups [%94 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %126 = x86.dm.vmovups [%94 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %127 = x86.dmk.vmovups[%94 + 192], %99 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %128 = x86.dm.vbroadcastss [%95] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %129 = x86.rss.vfmadd231ps %100, %124, %128 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %130 = x86.rss.vfmadd231ps %101, %125, %128 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %131 = x86.rss.vfmadd231ps %102, %126, %128 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %132 = x86.rss.vfmadd231ps %103, %127, %128 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %133 = x86.dm.vbroadcastss [%95 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %134 = x86.rss.vfmadd231ps %104, %124, %133 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %135 = x86.rss.vfmadd231ps %105, %125, %133 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %136 = x86.rss.vfmadd231ps %106, %126, %133 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %137 = x86.rss.vfmadd231ps %107, %127, %133 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %138 = x86.dm.vbroadcastss [%95 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %139 = x86.rss.vfmadd231ps %108, %124, %138 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %140 = x86.rss.vfmadd231ps %109, %125, %138 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %141 = x86.rss.vfmadd231ps %110, %126, %138 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %142 = x86.rss.vfmadd231ps %111, %127, %138 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %143 = x86.dm.vbroadcastss [%95 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %144 = x86.rss.vfmadd231ps %112, %124, %143 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %145 = x86.rss.vfmadd231ps %113, %125, %143 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %146 = x86.rss.vfmadd231ps %114, %126, %143 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %147 = x86.rss.vfmadd231ps %115, %127, %143 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %148 = x86.dm.vbroadcastss [%95 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %149 = x86.rss.vfmadd231ps %116, %124, %148 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %150 = x86.rss.vfmadd231ps %117, %125, %148 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %151 = x86.rss.vfmadd231ps %118, %126, %148 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %152 = x86.rss.vfmadd231ps %119, %127, %148 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %153 = x86.dm.vbroadcastss [%95 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %154 = x86.ri.add %95, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %155 = x86.ri.add %94, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %156 = x86.rss.vfmadd231ps %120, %124, %153 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %157 = x86.rss.vfmadd231ps %121, %125, %153 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %158 = x86.rss.vfmadd231ps %122, %126, %153 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %159 = x86.rss.vfmadd231ps %123, %127, %153 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %160 = x86.dm.vmovups [%155] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %161 = x86.dm.vmovups [%155 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %162 = x86.dm.vmovups [%155 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %163 = x86.dmk.vmovups[%155 + 192], %99 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %164 = x86.dm.vbroadcastss [%154] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %165 = x86.rss.vfmadd231ps %129, %160, %164 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %166 = x86.rss.vfmadd231ps %130, %161, %164 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %167 = x86.rss.vfmadd231ps %131, %162, %164 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %168 = x86.rss.vfmadd231ps %132, %163, %164 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %169 = x86.dm.vbroadcastss [%154 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %170 = x86.rss.vfmadd231ps %134, %160, %169 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %171 = x86.rss.vfmadd231ps %135, %161, %169 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %172 = x86.rss.vfmadd231ps %136, %162, %169 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %173 = x86.rss.vfmadd231ps %137, %163, %169 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %174 = x86.dm.vbroadcastss [%154 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %175 = x86.rss.vfmadd231ps %139, %160, %174 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %176 = x86.rss.vfmadd231ps %140, %161, %174 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %177 = x86.rss.vfmadd231ps %141, %162, %174 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %178 = x86.rss.vfmadd231ps %142, %163, %174 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %179 = x86.dm.vbroadcastss [%154 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %180 = x86.rss.vfmadd231ps %144, %160, %179 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %181 = x86.rss.vfmadd231ps %145, %161, %179 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %182 = x86.rss.vfmadd231ps %146, %162, %179 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %183 = x86.rss.vfmadd231ps %147, %163, %179 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %184 = x86.dm.vbroadcastss [%154 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %185 = x86.rss.vfmadd231ps %149, %160, %184 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %186 = x86.rss.vfmadd231ps %150, %161, %184 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %187 = x86.rss.vfmadd231ps %151, %162, %184 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %188 = x86.rss.vfmadd231ps %152, %163, %184 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %189 = x86.dm.vbroadcastss [%154 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %190 = x86.ri.add %154, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %191 = x86.ri.add %155, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %192 = x86.rss.vfmadd231ps %156, %160, %189 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %193 = x86.rss.vfmadd231ps %157, %161, %189 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %194 = x86.rss.vfmadd231ps %158, %162, %189 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %195 = x86.rss.vfmadd231ps %159, %163, %189 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %196 = x86.dm.vmovups [%191] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %197 = x86.dm.vmovups [%191 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %198 = x86.dm.vmovups [%191 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %199 = x86.dmk.vmovups[%191 + 192], %99 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %200 = x86.dm.vbroadcastss [%190] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %201 = x86.rss.vfmadd231ps %165, %196, %200 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %202 = x86.rss.vfmadd231ps %166, %197, %200 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %203 = x86.rss.vfmadd231ps %167, %198, %200 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %204 = x86.rss.vfmadd231ps %168, %199, %200 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %205 = x86.dm.vbroadcastss [%190 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %206 = x86.rss.vfmadd231ps %170, %196, %205 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %207 = x86.rss.vfmadd231ps %171, %197, %205 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %208 = x86.rss.vfmadd231ps %172, %198, %205 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %209 = x86.rss.vfmadd231ps %173, %199, %205 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %210 = x86.dm.vbroadcastss [%190 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %211 = x86.rss.vfmadd231ps %175, %196, %210 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %212 = x86.rss.vfmadd231ps %176, %197, %210 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %213 = x86.rss.vfmadd231ps %177, %198, %210 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %214 = x86.rss.vfmadd231ps %178, %199, %210 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %215 = x86.dm.vbroadcastss [%190 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %216 = x86.rss.vfmadd231ps %180, %196, %215 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %217 = x86.rss.vfmadd231ps %181, %197, %215 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %218 = x86.rss.vfmadd231ps %182, %198, %215 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %219 = x86.rss.vfmadd231ps %183, %199, %215 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %220 = x86.dm.vbroadcastss [%190 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %221 = x86.rss.vfmadd231ps %185, %196, %220 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %222 = x86.rss.vfmadd231ps %186, %197, %220 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %223 = x86.rss.vfmadd231ps %187, %198, %220 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %224 = x86.rss.vfmadd231ps %188, %199, %220 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %225 = x86.dm.vbroadcastss [%190 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %226 = x86.ri.add %190, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %227 = x86.ri.add %191, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %228 = x86.rss.vfmadd231ps %192, %196, %225 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %229 = x86.rss.vfmadd231ps %193, %197, %225 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %230 = x86.rss.vfmadd231ps %194, %198, %225 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %231 = x86.rss.vfmadd231ps %195, %199, %225 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %232 = x86.dm.vmovups [%227] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %233 = x86.dm.vmovups [%227 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %234 = x86.dm.vmovups [%227 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %235 = x86.dmk.vmovups[%227 + 192], %99 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %236 = x86.dm.vbroadcastss [%226] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %237 = x86.rss.vfmadd231ps %201, %232, %236 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %238 = x86.rss.vfmadd231ps %202, %233, %236 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %239 = x86.rss.vfmadd231ps %203, %234, %236 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %240 = x86.rss.vfmadd231ps %204, %235, %236 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %241 = x86.dm.vbroadcastss [%226 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %242 = x86.rss.vfmadd231ps %206, %232, %241 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %243 = x86.rss.vfmadd231ps %207, %233, %241 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %244 = x86.rss.vfmadd231ps %208, %234, %241 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %245 = x86.rss.vfmadd231ps %209, %235, %241 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %246 = x86.dm.vbroadcastss [%226 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %247 = x86.rss.vfmadd231ps %211, %232, %246 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %248 = x86.rss.vfmadd231ps %212, %233, %246 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %249 = x86.rss.vfmadd231ps %213, %234, %246 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %250 = x86.rss.vfmadd231ps %214, %235, %246 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %251 = x86.dm.vbroadcastss [%226 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %252 = x86.rss.vfmadd231ps %216, %232, %251 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %253 = x86.rss.vfmadd231ps %217, %233, %251 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %254 = x86.rss.vfmadd231ps %218, %234, %251 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %255 = x86.rss.vfmadd231ps %219, %235, %251 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %256 = x86.dm.vbroadcastss [%226 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %257 = x86.rss.vfmadd231ps %221, %232, %256 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %258 = x86.rss.vfmadd231ps %222, %233, %256 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %259 = x86.rss.vfmadd231ps %223, %234, %256 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %260 = x86.rss.vfmadd231ps %224, %235, %256 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %261 = x86.dm.vbroadcastss [%226 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %262 = x86.ri.add %226, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %263 = x86.ri.add %227, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %264 = x86.rss.vfmadd231ps %228, %232, %261 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %265 = x86.rss.vfmadd231ps %229, %233, %261 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %266 = x86.rss.vfmadd231ps %230, %234, %261 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %267 = x86.rss.vfmadd231ps %231, %235, %261 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %263, %262, %96, %97, %98, %99, %237, %238, %239, %240, %242, %243, %244, %245, %247, %248, %249, %250, %252, %253, %254, %255, %257, %258, %259, %260, %264, %265, %266, %267 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %268 = x86.ri.sub %64, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%65], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 64], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 128], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%65 + 192], %72, %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 200], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 264], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 328], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%65 + 392], %76, %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 400], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 464], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 528], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%65 + 592], %80, %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 600], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 664], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 728], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%65 + 792], %84, %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 800], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 864], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 928], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%65 + 992], %88, %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 1000], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 1064], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%65 + 1128], %91 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%65 + 1192], %92, %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %269 = x86.ri.add %65, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %270 = x86.ri.sub %63, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %270, %268, %269, %66, %67, %68 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %271 = x86.ri.add %27, 1000 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %272 = x86.ri.add %26, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %273 = x86.ri.sub %25, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %273, %272, %271, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %274 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %275, %276, %277, %278, %279 = x86_scf.for %280 : !x86.reg64<r11>  = %274 to 38 : si32 step 5 : si32 iter_args(%281 = %11, %282 = %12, %283 = %13, %284 = %14, %285 = %15) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %286 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %287 = x86.ks.kmovw %286 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %288 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %289, %290, %291, %292, %293, %294 = x86_scf.for %295 : !x86.reg64<r10>  = %288 to 50 : si32 step 50 : si32 iter_args(%296 = %281, %297 = %282, %298 = %283, %299 = %284, %300 = %285, %301 = %287) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %302 = x86.dm.vmovups [%298] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %303 = x86.dm.vmovups [%298 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %304 = x86.dm.vmovups [%298 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %305 = x86.dmk.vmovups[%298 + 192], %301 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %306 = x86.dm.vmovups [%298 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %307 = x86.dm.vmovups [%298 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %308 = x86.dm.vmovups [%298 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %309 = x86.dmk.vmovups[%298 + 392], %301 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %310 = x86.dm.vmovups [%298 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %311 = x86.dm.vmovups [%298 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %312 = x86.dm.vmovups [%298 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %313 = x86.dmk.vmovups[%298 + 592], %301 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %314 = x86.dm.vmovups [%298 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %315 = x86.dm.vmovups [%298 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %316 = x86.dm.vmovups [%298 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %317 = x86.dmk.vmovups[%298 + 792], %301 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %318 = x86.dm.vmovups [%298 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %319 = x86.dm.vmovups [%298 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %320 = x86.dm.vmovups [%298 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %321 = x86.dmk.vmovups[%298 + 992], %301 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %322 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %323, %324, %325, %326, %327, %328, %329, %330, %331, %332, %333, %334, %335, %336, %337, %338, %339, %340, %341, %342, %343, %344, %345, %346, %347, %348 = x86_scf.for %349 : !x86.reg64<r12>  = %322 to 128 : si32 step 4 : si32 iter_args(%350 = %296, %351 = %297, %352 = %298, %353 = %299, %354 = %300, %355 = %301, %356 = %302, %357 = %303, %358 = %304, %359 = %305, %360 = %306, %361 = %307, %362 = %308, %363 = %309, %364 = %310, %365 = %311, %366 = %312, %367 = %313, %368 = %314, %369 = %315, %370 = %316, %371 = %317, %372 = %318, %373 = %319, %374 = %320, %375 = %321) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %376 = x86.dm.vmovups [%350] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %377 = x86.dm.vmovups [%350 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %378 = x86.dm.vmovups [%350 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %379 = x86.dmk.vmovups[%350 + 192], %355 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %380 = x86.dm.vbroadcastss [%351] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %381 = x86.rss.vfmadd231ps %356, %376, %380 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %382 = x86.rss.vfmadd231ps %357, %377, %380 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %383 = x86.rss.vfmadd231ps %358, %378, %380 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %384 = x86.rss.vfmadd231ps %359, %379, %380 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %385 = x86.dm.vbroadcastss [%351 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %386 = x86.rss.vfmadd231ps %360, %376, %385 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %387 = x86.rss.vfmadd231ps %361, %377, %385 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %388 = x86.rss.vfmadd231ps %362, %378, %385 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %389 = x86.rss.vfmadd231ps %363, %379, %385 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %390 = x86.dm.vbroadcastss [%351 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %391 = x86.rss.vfmadd231ps %364, %376, %390 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %392 = x86.rss.vfmadd231ps %365, %377, %390 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %393 = x86.rss.vfmadd231ps %366, %378, %390 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %394 = x86.rss.vfmadd231ps %367, %379, %390 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %395 = x86.dm.vbroadcastss [%351 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %396 = x86.rss.vfmadd231ps %368, %376, %395 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %397 = x86.rss.vfmadd231ps %369, %377, %395 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %398 = x86.rss.vfmadd231ps %370, %378, %395 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %399 = x86.rss.vfmadd231ps %371, %379, %395 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %400 = x86.dm.vbroadcastss [%351 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %401 = x86.ri.add %351, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %402 = x86.ri.add %350, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %403 = x86.rss.vfmadd231ps %372, %376, %400 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %404 = x86.rss.vfmadd231ps %373, %377, %400 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %405 = x86.rss.vfmadd231ps %374, %378, %400 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %406 = x86.rss.vfmadd231ps %375, %379, %400 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %407 = x86.dm.vmovups [%402] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %408 = x86.dm.vmovups [%402 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %409 = x86.dm.vmovups [%402 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %410 = x86.dmk.vmovups[%402 + 192], %355 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %411 = x86.dm.vbroadcastss [%401] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %412 = x86.rss.vfmadd231ps %381, %407, %411 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %413 = x86.rss.vfmadd231ps %382, %408, %411 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %414 = x86.rss.vfmadd231ps %383, %409, %411 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %415 = x86.rss.vfmadd231ps %384, %410, %411 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %416 = x86.dm.vbroadcastss [%401 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %417 = x86.rss.vfmadd231ps %386, %407, %416 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %418 = x86.rss.vfmadd231ps %387, %408, %416 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %419 = x86.rss.vfmadd231ps %388, %409, %416 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %420 = x86.rss.vfmadd231ps %389, %410, %416 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %421 = x86.dm.vbroadcastss [%401 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %422 = x86.rss.vfmadd231ps %391, %407, %421 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %423 = x86.rss.vfmadd231ps %392, %408, %421 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %424 = x86.rss.vfmadd231ps %393, %409, %421 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %425 = x86.rss.vfmadd231ps %394, %410, %421 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %426 = x86.dm.vbroadcastss [%401 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %427 = x86.rss.vfmadd231ps %396, %407, %426 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %428 = x86.rss.vfmadd231ps %397, %408, %426 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %429 = x86.rss.vfmadd231ps %398, %409, %426 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %430 = x86.rss.vfmadd231ps %399, %410, %426 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %431 = x86.dm.vbroadcastss [%401 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %432 = x86.ri.add %401, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %433 = x86.ri.add %402, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %434 = x86.rss.vfmadd231ps %403, %407, %431 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %435 = x86.rss.vfmadd231ps %404, %408, %431 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %436 = x86.rss.vfmadd231ps %405, %409, %431 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %437 = x86.rss.vfmadd231ps %406, %410, %431 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %438 = x86.dm.vmovups [%433] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %439 = x86.dm.vmovups [%433 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %440 = x86.dm.vmovups [%433 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %441 = x86.dmk.vmovups[%433 + 192], %355 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %442 = x86.dm.vbroadcastss [%432] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %443 = x86.rss.vfmadd231ps %412, %438, %442 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %444 = x86.rss.vfmadd231ps %413, %439, %442 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %445 = x86.rss.vfmadd231ps %414, %440, %442 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %446 = x86.rss.vfmadd231ps %415, %441, %442 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %447 = x86.dm.vbroadcastss [%432 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %448 = x86.rss.vfmadd231ps %417, %438, %447 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %449 = x86.rss.vfmadd231ps %418, %439, %447 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %450 = x86.rss.vfmadd231ps %419, %440, %447 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %451 = x86.rss.vfmadd231ps %420, %441, %447 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %452 = x86.dm.vbroadcastss [%432 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %453 = x86.rss.vfmadd231ps %422, %438, %452 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %454 = x86.rss.vfmadd231ps %423, %439, %452 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %455 = x86.rss.vfmadd231ps %424, %440, %452 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %456 = x86.rss.vfmadd231ps %425, %441, %452 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %457 = x86.dm.vbroadcastss [%432 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %458 = x86.rss.vfmadd231ps %427, %438, %457 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %459 = x86.rss.vfmadd231ps %428, %439, %457 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %460 = x86.rss.vfmadd231ps %429, %440, %457 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %461 = x86.rss.vfmadd231ps %430, %441, %457 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %462 = x86.dm.vbroadcastss [%432 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %463 = x86.ri.add %432, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %464 = x86.ri.add %433, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %465 = x86.rss.vfmadd231ps %434, %438, %462 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %466 = x86.rss.vfmadd231ps %435, %439, %462 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %467 = x86.rss.vfmadd231ps %436, %440, %462 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %468 = x86.rss.vfmadd231ps %437, %441, %462 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %469 = x86.dm.vmovups [%464] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %470 = x86.dm.vmovups [%464 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %471 = x86.dm.vmovups [%464 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %472 = x86.dmk.vmovups[%464 + 192], %355 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %473 = x86.dm.vbroadcastss [%463] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %474 = x86.rss.vfmadd231ps %443, %469, %473 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %475 = x86.rss.vfmadd231ps %444, %470, %473 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %476 = x86.rss.vfmadd231ps %445, %471, %473 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %477 = x86.rss.vfmadd231ps %446, %472, %473 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %478 = x86.dm.vbroadcastss [%463 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %479 = x86.rss.vfmadd231ps %448, %469, %478 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %480 = x86.rss.vfmadd231ps %449, %470, %478 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %481 = x86.rss.vfmadd231ps %450, %471, %478 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %482 = x86.rss.vfmadd231ps %451, %472, %478 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %483 = x86.dm.vbroadcastss [%463 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %484 = x86.rss.vfmadd231ps %453, %469, %483 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %485 = x86.rss.vfmadd231ps %454, %470, %483 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %486 = x86.rss.vfmadd231ps %455, %471, %483 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %487 = x86.rss.vfmadd231ps %456, %472, %483 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %488 = x86.dm.vbroadcastss [%463 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %489 = x86.rss.vfmadd231ps %458, %469, %488 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %490 = x86.rss.vfmadd231ps %459, %470, %488 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %491 = x86.rss.vfmadd231ps %460, %471, %488 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %492 = x86.rss.vfmadd231ps %461, %472, %488 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %493 = x86.dm.vbroadcastss [%463 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %494 = x86.ri.add %463, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %495 = x86.ri.add %464, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %496 = x86.rss.vfmadd231ps %465, %469, %493 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %497 = x86.rss.vfmadd231ps %466, %470, %493 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %498 = x86.rss.vfmadd231ps %467, %471, %493 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %499 = x86.rss.vfmadd231ps %468, %472, %493 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %495, %494, %352, %353, %354, %355, %474, %475, %476, %477, %479, %480, %481, %482, %484, %485, %486, %487, %489, %490, %491, %492, %496, %497, %498, %499 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %500 = x86.ri.sub %324, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%325], %329 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 64], %330 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 128], %331 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%325 + 192], %332, %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 200], %333 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 264], %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 328], %335 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%325 + 392], %336, %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 400], %337 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 464], %338 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 528], %339 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%325 + 592], %340, %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 600], %341 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 664], %342 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 728], %343 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%325 + 792], %344, %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 800], %345 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 864], %346 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%325 + 928], %347 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%325 + 992], %348, %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %501 = x86.ri.add %325, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %502 = x86.ri.sub %323, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %502, %500, %501, %326, %327, %328 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %503 = x86.ri.add %291, 800 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %504 = x86.ri.add %290, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %505 = x86.ri.sub %289, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %505, %504, %503, %292, %293 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %506 = x86.ds.mov %278 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %507, %508 = x86.d.pop %506 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
