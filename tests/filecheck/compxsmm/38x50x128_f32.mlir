// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:     x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:  %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:  %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:  %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:  %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:  %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:  %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 18 : si32 step 6 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:    %23 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:    %24 = x86.ks.kmovw %23 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %25 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %26, %27, %28, %29, %30, %31, %32 = x86_scf.for %33 : !x86.reg64<r10>  = %25 to 50 : si32 step 50 : si32 iter_args(%34 = %18, %35 = %19, %36 = %20, %37 = %21, %38 = %22, %39 = %24) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:      %40 = x86.dm.vmovups [%36] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %41 = x86.dm.vmovups [%36 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %42 = x86.dm.vmovups [%36 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %43 = x86.dmk.vmovups[%36 + 192], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %44 = x86.dm.vmovups [%36 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %45 = x86.dm.vmovups [%36 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %46 = x86.dm.vmovups [%36 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %47 = x86.dmk.vmovups[%36 + 392], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %48 = x86.dm.vmovups [%36 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %49 = x86.dm.vmovups [%36 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %50 = x86.dm.vmovups [%36 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %51 = x86.dmk.vmovups[%36 + 592], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %52 = x86.dm.vmovups [%36 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %53 = x86.dm.vmovups [%36 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %54 = x86.dm.vmovups [%36 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %55 = x86.dmk.vmovups[%36 + 792], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %56 = x86.dm.vmovups [%36 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %57 = x86.dm.vmovups [%36 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %58 = x86.dm.vmovups [%36 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %59 = x86.dmk.vmovups[%36 + 992], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %60 = x86.dm.vmovups [%36 + 1000] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %61 = x86.dm.vmovups [%36 + 1064] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %62 = x86.dm.vmovups [%36 + 1128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %63 = x86.dmk.vmovups[%36 + 1192], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %64 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95 = x86_scf.for %96 : !x86.reg64<r12>  = %64 to 128 : si32 step 4 : si32 iter_args(%97 = %34, %98 = %35, %99 = %36, %100 = %37, %101 = %38, %102 = %39, %103 = %40, %104 = %41, %105 = %42, %106 = %43, %107 = %44, %108 = %45, %109 = %46, %110 = %47, %111 = %48, %112 = %49, %113 = %50, %114 = %51, %115 = %52, %116 = %53, %117 = %54, %118 = %55, %119 = %56, %120 = %57, %121 = %58, %122 = %59, %123 = %60, %124 = %61, %125 = %62, %126 = %63) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %127 = x86.dm.vmovups [%97] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %128 = x86.dm.vmovups [%97 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %129 = x86.dm.vmovups [%97 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %130 = x86.dmk.vmovups[%97 + 192], %102 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %131 = x86.dm.vbroadcastss [%98] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %132 = x86.rss.vfmadd231ps %103, %127, %131 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %133 = x86.rss.vfmadd231ps %104, %128, %131 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %134 = x86.rss.vfmadd231ps %105, %129, %131 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %135 = x86.rss.vfmadd231ps %106, %130, %131 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %136 = x86.dm.vbroadcastss [%98 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %137 = x86.rss.vfmadd231ps %107, %127, %136 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %138 = x86.rss.vfmadd231ps %108, %128, %136 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %139 = x86.rss.vfmadd231ps %109, %129, %136 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %140 = x86.rss.vfmadd231ps %110, %130, %136 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %141 = x86.dm.vbroadcastss [%98 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %142 = x86.rss.vfmadd231ps %111, %127, %141 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %143 = x86.rss.vfmadd231ps %112, %128, %141 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %144 = x86.rss.vfmadd231ps %113, %129, %141 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %145 = x86.rss.vfmadd231ps %114, %130, %141 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %146 = x86.dm.vbroadcastss [%98 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %147 = x86.rss.vfmadd231ps %115, %127, %146 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %148 = x86.rss.vfmadd231ps %116, %128, %146 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %149 = x86.rss.vfmadd231ps %117, %129, %146 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %150 = x86.rss.vfmadd231ps %118, %130, %146 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %151 = x86.dm.vbroadcastss [%98 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %152 = x86.rss.vfmadd231ps %119, %127, %151 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %153 = x86.rss.vfmadd231ps %120, %128, %151 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %154 = x86.rss.vfmadd231ps %121, %129, %151 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %155 = x86.rss.vfmadd231ps %122, %130, %151 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %156 = x86.dm.vbroadcastss [%98 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %157 = x86.ri.add %98, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %158 = x86.ri.add %97, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %159 = x86.rss.vfmadd231ps %123, %127, %156 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %160 = x86.rss.vfmadd231ps %124, %128, %156 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %161 = x86.rss.vfmadd231ps %125, %129, %156 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %162 = x86.rss.vfmadd231ps %126, %130, %156 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %163 = x86.dm.vmovups [%158] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %164 = x86.dm.vmovups [%158 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %165 = x86.dm.vmovups [%158 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %166 = x86.dmk.vmovups[%158 + 192], %102 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %167 = x86.dm.vbroadcastss [%157] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %168 = x86.rss.vfmadd231ps %132, %163, %167 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %169 = x86.rss.vfmadd231ps %133, %164, %167 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %170 = x86.rss.vfmadd231ps %134, %165, %167 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %171 = x86.rss.vfmadd231ps %135, %166, %167 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %172 = x86.dm.vbroadcastss [%157 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %173 = x86.rss.vfmadd231ps %137, %163, %172 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %174 = x86.rss.vfmadd231ps %138, %164, %172 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %175 = x86.rss.vfmadd231ps %139, %165, %172 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %176 = x86.rss.vfmadd231ps %140, %166, %172 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %177 = x86.dm.vbroadcastss [%157 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %178 = x86.rss.vfmadd231ps %142, %163, %177 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %179 = x86.rss.vfmadd231ps %143, %164, %177 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %180 = x86.rss.vfmadd231ps %144, %165, %177 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %181 = x86.rss.vfmadd231ps %145, %166, %177 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %182 = x86.dm.vbroadcastss [%157 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %183 = x86.rss.vfmadd231ps %147, %163, %182 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %184 = x86.rss.vfmadd231ps %148, %164, %182 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %185 = x86.rss.vfmadd231ps %149, %165, %182 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %186 = x86.rss.vfmadd231ps %150, %166, %182 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %187 = x86.dm.vbroadcastss [%157 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %188 = x86.rss.vfmadd231ps %152, %163, %187 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %189 = x86.rss.vfmadd231ps %153, %164, %187 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %190 = x86.rss.vfmadd231ps %154, %165, %187 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %191 = x86.rss.vfmadd231ps %155, %166, %187 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %192 = x86.dm.vbroadcastss [%157 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %193 = x86.ri.add %157, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %194 = x86.ri.add %158, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %195 = x86.rss.vfmadd231ps %159, %163, %192 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %196 = x86.rss.vfmadd231ps %160, %164, %192 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %197 = x86.rss.vfmadd231ps %161, %165, %192 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %198 = x86.rss.vfmadd231ps %162, %166, %192 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %199 = x86.dm.vmovups [%194] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %200 = x86.dm.vmovups [%194 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %201 = x86.dm.vmovups [%194 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %202 = x86.dmk.vmovups[%194 + 192], %102 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %203 = x86.dm.vbroadcastss [%193] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %204 = x86.rss.vfmadd231ps %168, %199, %203 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %205 = x86.rss.vfmadd231ps %169, %200, %203 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %206 = x86.rss.vfmadd231ps %170, %201, %203 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %207 = x86.rss.vfmadd231ps %171, %202, %203 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %208 = x86.dm.vbroadcastss [%193 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %209 = x86.rss.vfmadd231ps %173, %199, %208 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %210 = x86.rss.vfmadd231ps %174, %200, %208 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %211 = x86.rss.vfmadd231ps %175, %201, %208 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %212 = x86.rss.vfmadd231ps %176, %202, %208 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %213 = x86.dm.vbroadcastss [%193 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %214 = x86.rss.vfmadd231ps %178, %199, %213 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %215 = x86.rss.vfmadd231ps %179, %200, %213 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %216 = x86.rss.vfmadd231ps %180, %201, %213 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %217 = x86.rss.vfmadd231ps %181, %202, %213 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %218 = x86.dm.vbroadcastss [%193 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %219 = x86.rss.vfmadd231ps %183, %199, %218 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %220 = x86.rss.vfmadd231ps %184, %200, %218 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %221 = x86.rss.vfmadd231ps %185, %201, %218 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %222 = x86.rss.vfmadd231ps %186, %202, %218 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %223 = x86.dm.vbroadcastss [%193 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %224 = x86.rss.vfmadd231ps %188, %199, %223 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %225 = x86.rss.vfmadd231ps %189, %200, %223 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %226 = x86.rss.vfmadd231ps %190, %201, %223 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %227 = x86.rss.vfmadd231ps %191, %202, %223 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %228 = x86.dm.vbroadcastss [%193 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %229 = x86.ri.add %193, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %230 = x86.ri.add %194, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %231 = x86.rss.vfmadd231ps %195, %199, %228 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %232 = x86.rss.vfmadd231ps %196, %200, %228 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %233 = x86.rss.vfmadd231ps %197, %201, %228 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %234 = x86.rss.vfmadd231ps %198, %202, %228 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %235 = x86.dm.vmovups [%230] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %236 = x86.dm.vmovups [%230 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %237 = x86.dm.vmovups [%230 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %238 = x86.dmk.vmovups[%230 + 192], %102 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %239 = x86.dm.vbroadcastss [%229] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %240 = x86.rss.vfmadd231ps %204, %235, %239 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %241 = x86.rss.vfmadd231ps %205, %236, %239 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %242 = x86.rss.vfmadd231ps %206, %237, %239 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %243 = x86.rss.vfmadd231ps %207, %238, %239 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %244 = x86.dm.vbroadcastss [%229 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %245 = x86.rss.vfmadd231ps %209, %235, %244 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %246 = x86.rss.vfmadd231ps %210, %236, %244 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %247 = x86.rss.vfmadd231ps %211, %237, %244 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %248 = x86.rss.vfmadd231ps %212, %238, %244 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %249 = x86.dm.vbroadcastss [%229 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %250 = x86.rss.vfmadd231ps %214, %235, %249 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %251 = x86.rss.vfmadd231ps %215, %236, %249 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %252 = x86.rss.vfmadd231ps %216, %237, %249 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %253 = x86.rss.vfmadd231ps %217, %238, %249 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %254 = x86.dm.vbroadcastss [%229 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %255 = x86.rss.vfmadd231ps %219, %235, %254 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %256 = x86.rss.vfmadd231ps %220, %236, %254 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %257 = x86.rss.vfmadd231ps %221, %237, %254 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %258 = x86.rss.vfmadd231ps %222, %238, %254 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %259 = x86.dm.vbroadcastss [%229 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %260 = x86.rss.vfmadd231ps %224, %235, %259 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %261 = x86.rss.vfmadd231ps %225, %236, %259 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %262 = x86.rss.vfmadd231ps %226, %237, %259 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %263 = x86.rss.vfmadd231ps %227, %238, %259 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %264 = x86.dm.vbroadcastss [%229 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %265 = x86.ri.add %229, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %266 = x86.ri.add %230, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %267 = x86.rss.vfmadd231ps %231, %235, %264 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %268 = x86.rss.vfmadd231ps %232, %236, %264 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %269 = x86.rss.vfmadd231ps %233, %237, %264 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %270 = x86.rss.vfmadd231ps %234, %238, %264 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %266, %265, %99, %100, %101, %102, %240, %241, %242, %243, %245, %246, %247, %248, %250, %251, %252, %253, %255, %256, %257, %258, %260, %261, %262, %263, %267, %268, %269, %270 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %271 = x86.ri.sub %67, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%68], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 64], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 128], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%68 + 192], %75, %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 200], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 264], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 328], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%68 + 392], %79, %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 400], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 464], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 528], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%68 + 592], %83, %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 600], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 664], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 728], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%68 + 792], %87, %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 800], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 864], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 928], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%68 + 992], %91, %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 1000], %92 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 1064], %93 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%68 + 1128], %94 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%68 + 1192], %95, %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %272 = x86.ri.add %68, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %273 = x86.ri.sub %66, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %273, %271, %272, %69, %70, %71 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %274 = x86.ri.add %29, 1000 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %275 = x86.ri.add %28, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %276 = x86.ri.sub %27, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %276, %275, %274, %30, %31 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %277 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:  %278, %279, %280, %281, %282, %283 = x86_scf.for %284 : !x86.reg64<r11>  = %277 to 38 : si32 step 5 : si32 iter_args(%285 = %12, %286 = %13, %287 = %14, %288 = %15, %289 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:    %290 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:    %291 = x86.ks.kmovw %290 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %292 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %293, %294, %295, %296, %297, %298, %299 = x86_scf.for %300 : !x86.reg64<r10>  = %292 to 50 : si32 step 50 : si32 iter_args(%301 = %285, %302 = %286, %303 = %287, %304 = %288, %305 = %289, %306 = %291) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:      %307 = x86.dm.vmovups [%303] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %308 = x86.dm.vmovups [%303 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %309 = x86.dm.vmovups [%303 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %310 = x86.dmk.vmovups[%303 + 192], %306 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %311 = x86.dm.vmovups [%303 + 200] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %312 = x86.dm.vmovups [%303 + 264] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %313 = x86.dm.vmovups [%303 + 328] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %314 = x86.dmk.vmovups[%303 + 392], %306 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %315 = x86.dm.vmovups [%303 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %316 = x86.dm.vmovups [%303 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %317 = x86.dm.vmovups [%303 + 528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %318 = x86.dmk.vmovups[%303 + 592], %306 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %319 = x86.dm.vmovups [%303 + 600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %320 = x86.dm.vmovups [%303 + 664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %321 = x86.dm.vmovups [%303 + 728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %322 = x86.dmk.vmovups[%303 + 792], %306 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %323 = x86.dm.vmovups [%303 + 800] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %324 = x86.dm.vmovups [%303 + 864] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %325 = x86.dm.vmovups [%303 + 928] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %326 = x86.dmk.vmovups[%303 + 992], %306 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %327 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %328, %329, %330, %331, %332, %333, %334, %335, %336, %337, %338, %339, %340, %341, %342, %343, %344, %345, %346, %347, %348, %349, %350, %351, %352, %353, %354 = x86_scf.for %355 : !x86.reg64<r12>  = %327 to 128 : si32 step 4 : si32 iter_args(%356 = %301, %357 = %302, %358 = %303, %359 = %304, %360 = %305, %361 = %306, %362 = %307, %363 = %308, %364 = %309, %365 = %310, %366 = %311, %367 = %312, %368 = %313, %369 = %314, %370 = %315, %371 = %316, %372 = %317, %373 = %318, %374 = %319, %375 = %320, %376 = %321, %377 = %322, %378 = %323, %379 = %324, %380 = %325, %381 = %326) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %382 = x86.dm.vmovups [%356] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %383 = x86.dm.vmovups [%356 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %384 = x86.dm.vmovups [%356 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %385 = x86.dmk.vmovups[%356 + 192], %361 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %386 = x86.dm.vbroadcastss [%357] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %387 = x86.rss.vfmadd231ps %362, %382, %386 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %388 = x86.rss.vfmadd231ps %363, %383, %386 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %389 = x86.rss.vfmadd231ps %364, %384, %386 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %390 = x86.rss.vfmadd231ps %365, %385, %386 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %391 = x86.dm.vbroadcastss [%357 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %392 = x86.rss.vfmadd231ps %366, %382, %391 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %393 = x86.rss.vfmadd231ps %367, %383, %391 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %394 = x86.rss.vfmadd231ps %368, %384, %391 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %395 = x86.rss.vfmadd231ps %369, %385, %391 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %396 = x86.dm.vbroadcastss [%357 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %397 = x86.rss.vfmadd231ps %370, %382, %396 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %398 = x86.rss.vfmadd231ps %371, %383, %396 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %399 = x86.rss.vfmadd231ps %372, %384, %396 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %400 = x86.rss.vfmadd231ps %373, %385, %396 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %401 = x86.dm.vbroadcastss [%357 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %402 = x86.rss.vfmadd231ps %374, %382, %401 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %403 = x86.rss.vfmadd231ps %375, %383, %401 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %404 = x86.rss.vfmadd231ps %376, %384, %401 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %405 = x86.rss.vfmadd231ps %377, %385, %401 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %406 = x86.dm.vbroadcastss [%357 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %407 = x86.ri.add %357, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %408 = x86.ri.add %356, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %409 = x86.rss.vfmadd231ps %378, %382, %406 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %410 = x86.rss.vfmadd231ps %379, %383, %406 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %411 = x86.rss.vfmadd231ps %380, %384, %406 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %412 = x86.rss.vfmadd231ps %381, %385, %406 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %413 = x86.dm.vmovups [%408] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %414 = x86.dm.vmovups [%408 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %415 = x86.dm.vmovups [%408 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %416 = x86.dmk.vmovups[%408 + 192], %361 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %417 = x86.dm.vbroadcastss [%407] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %418 = x86.rss.vfmadd231ps %387, %413, %417 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %419 = x86.rss.vfmadd231ps %388, %414, %417 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %420 = x86.rss.vfmadd231ps %389, %415, %417 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %421 = x86.rss.vfmadd231ps %390, %416, %417 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %422 = x86.dm.vbroadcastss [%407 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %423 = x86.rss.vfmadd231ps %392, %413, %422 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %424 = x86.rss.vfmadd231ps %393, %414, %422 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %425 = x86.rss.vfmadd231ps %394, %415, %422 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %426 = x86.rss.vfmadd231ps %395, %416, %422 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %427 = x86.dm.vbroadcastss [%407 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %428 = x86.rss.vfmadd231ps %397, %413, %427 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %429 = x86.rss.vfmadd231ps %398, %414, %427 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %430 = x86.rss.vfmadd231ps %399, %415, %427 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %431 = x86.rss.vfmadd231ps %400, %416, %427 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %432 = x86.dm.vbroadcastss [%407 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %433 = x86.rss.vfmadd231ps %402, %413, %432 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %434 = x86.rss.vfmadd231ps %403, %414, %432 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %435 = x86.rss.vfmadd231ps %404, %415, %432 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %436 = x86.rss.vfmadd231ps %405, %416, %432 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %437 = x86.dm.vbroadcastss [%407 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %438 = x86.ri.add %407, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %439 = x86.ri.add %408, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %440 = x86.rss.vfmadd231ps %409, %413, %437 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %441 = x86.rss.vfmadd231ps %410, %414, %437 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %442 = x86.rss.vfmadd231ps %411, %415, %437 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %443 = x86.rss.vfmadd231ps %412, %416, %437 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %444 = x86.dm.vmovups [%439] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %445 = x86.dm.vmovups [%439 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %446 = x86.dm.vmovups [%439 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %447 = x86.dmk.vmovups[%439 + 192], %361 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %448 = x86.dm.vbroadcastss [%438] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %449 = x86.rss.vfmadd231ps %418, %444, %448 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %450 = x86.rss.vfmadd231ps %419, %445, %448 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %451 = x86.rss.vfmadd231ps %420, %446, %448 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %452 = x86.rss.vfmadd231ps %421, %447, %448 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %453 = x86.dm.vbroadcastss [%438 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %454 = x86.rss.vfmadd231ps %423, %444, %453 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %455 = x86.rss.vfmadd231ps %424, %445, %453 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %456 = x86.rss.vfmadd231ps %425, %446, %453 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %457 = x86.rss.vfmadd231ps %426, %447, %453 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %458 = x86.dm.vbroadcastss [%438 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %459 = x86.rss.vfmadd231ps %428, %444, %458 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %460 = x86.rss.vfmadd231ps %429, %445, %458 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %461 = x86.rss.vfmadd231ps %430, %446, %458 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %462 = x86.rss.vfmadd231ps %431, %447, %458 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %463 = x86.dm.vbroadcastss [%438 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %464 = x86.rss.vfmadd231ps %433, %444, %463 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %465 = x86.rss.vfmadd231ps %434, %445, %463 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %466 = x86.rss.vfmadd231ps %435, %446, %463 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %467 = x86.rss.vfmadd231ps %436, %447, %463 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %468 = x86.dm.vbroadcastss [%438 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %469 = x86.ri.add %438, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %470 = x86.ri.add %439, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %471 = x86.rss.vfmadd231ps %440, %444, %468 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %472 = x86.rss.vfmadd231ps %441, %445, %468 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %473 = x86.rss.vfmadd231ps %442, %446, %468 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %474 = x86.rss.vfmadd231ps %443, %447, %468 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %475 = x86.dm.vmovups [%470] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %476 = x86.dm.vmovups [%470 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %477 = x86.dm.vmovups [%470 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %478 = x86.dmk.vmovups[%470 + 192], %361 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %479 = x86.dm.vbroadcastss [%469] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %480 = x86.rss.vfmadd231ps %449, %475, %479 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %481 = x86.rss.vfmadd231ps %450, %476, %479 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %482 = x86.rss.vfmadd231ps %451, %477, %479 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %483 = x86.rss.vfmadd231ps %452, %478, %479 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %484 = x86.dm.vbroadcastss [%469 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %485 = x86.rss.vfmadd231ps %454, %475, %484 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %486 = x86.rss.vfmadd231ps %455, %476, %484 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %487 = x86.rss.vfmadd231ps %456, %477, %484 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %488 = x86.rss.vfmadd231ps %457, %478, %484 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %489 = x86.dm.vbroadcastss [%469 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %490 = x86.rss.vfmadd231ps %459, %475, %489 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %491 = x86.rss.vfmadd231ps %460, %476, %489 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %492 = x86.rss.vfmadd231ps %461, %477, %489 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %493 = x86.rss.vfmadd231ps %462, %478, %489 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %494 = x86.dm.vbroadcastss [%469 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %495 = x86.rss.vfmadd231ps %464, %475, %494 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %496 = x86.rss.vfmadd231ps %465, %476, %494 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %497 = x86.rss.vfmadd231ps %466, %477, %494 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %498 = x86.rss.vfmadd231ps %467, %478, %494 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %499 = x86.dm.vbroadcastss [%469 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %500 = x86.ri.add %469, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %501 = x86.ri.add %470, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %502 = x86.rss.vfmadd231ps %471, %475, %499 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %503 = x86.rss.vfmadd231ps %472, %476, %499 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %504 = x86.rss.vfmadd231ps %473, %477, %499 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %505 = x86.rss.vfmadd231ps %474, %478, %499 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %501, %500, %358, %359, %360, %361, %480, %481, %482, %483, %485, %486, %487, %488, %490, %491, %492, %493, %495, %496, %497, %498, %502, %503, %504, %505 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %506 = x86.ri.sub %330, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%331], %335 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 64], %336 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 128], %337 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%331 + 192], %338, %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 200], %339 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 264], %340 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 328], %341 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%331 + 392], %342, %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 400], %343 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 464], %344 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 528], %345 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%331 + 592], %346, %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 600], %347 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 664], %348 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 728], %349 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%331 + 792], %350, %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 800], %351 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 864], %352 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%331 + 928], %353 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%331 + 992], %354, %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %507 = x86.ri.add %331, 200 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %508 = x86.ri.sub %329, 25400 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %508, %506, %507, %332, %333, %334 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %509 = x86.ri.add %296, 800 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %510 = x86.ri.add %295, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %511 = x86.ri.sub %294, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %511, %510, %509, %297, %298 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %512 = x86.ds.mov %282 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %513, %514 = x86.d.pop %512 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:  x86_func.ret
// CHECK-NEXT:}

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
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 50
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm11 {k1}{z}, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+200]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+264]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+328]
// CHECK-REGALLOC-NEXT:      vmovups zmm15 {k1}{z}, [rdx+392]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+528]
// CHECK-REGALLOC-NEXT:      vmovups zmm19 {k1}{z}, [rdx+592]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+600]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+664]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+728]
// CHECK-REGALLOC-NEXT:      vmovups zmm23 {k1}{z}, [rdx+792]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+800]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+864]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+928]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx+992]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+1000]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+1064]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+1128]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1192]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
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
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192] {k1}, zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+200], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+264], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+328], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+392] {k1}, zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+400], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+464], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+528], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+592] {k1}, zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+600], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+664], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+728], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+792] {k1}, zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+800], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+864], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+928], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+992] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1000], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1064], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1128], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1192] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 200
// CHECK-REGALLOC-NEXT:      sub rdi, 25400
// CHECK-REGALLOC-NEXT:      cmp rcx, 50
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 1000
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      sub rdi, 200
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 18
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 50
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm15 {k1}{z}, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+200]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+264]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+328]
// CHECK-REGALLOC-NEXT:      vmovups zmm19 {k1}{z}, [rdx+392]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+528]
// CHECK-REGALLOC-NEXT:      vmovups zmm23 {k1}{z}, [rdx+592]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+600]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+664]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+728]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx+792]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+800]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+864]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+928]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+992]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+192]
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
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192] {k1}, zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+200], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+264], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+328], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+392] {k1}, zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+400], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+464], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+528], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+592] {k1}, zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+600], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+664], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+728], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+792] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+800], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+864], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+928], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+992] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 200
// CHECK-REGALLOC-NEXT:      sub rdi, 25400
// CHECK-REGALLOC-NEXT:      cmp rcx, 50
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      add rdx, 800
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      sub rdi, 200
// CHECK-REGALLOC-NEXT:      cmp rax, 38
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
