// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128 128 128 128 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 128 38 128 128 128 128 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

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
// CHECK-NEXT:    %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 128 : si32 step 64 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %36 = x86.dm.vmovaps [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %37 = x86.dm.vmovaps [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %38 = x86.dm.vmovaps [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %39 = x86.dm.vmovaps [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %40 = x86.dm.vmovaps [%33 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %41 = x86.dm.vmovaps [%33 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %42 = x86.dm.vmovaps [%33 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %43 = x86.dm.vmovaps [%33 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %44 = x86.dm.vmovaps [%33 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %45 = x86.dm.vmovaps [%33 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %46 = x86.dm.vmovaps [%33 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %47 = x86.dm.vmovaps [%33 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %48 = x86.dm.vmovaps [%33 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %49 = x86.dm.vmovaps [%33 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %50 = x86.dm.vmovaps [%33 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %51 = x86.dm.vmovaps [%33 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %52 = x86.dm.vmovaps [%33 + 2048] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %53 = x86.dm.vmovaps [%33 + 2112] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %54 = x86.dm.vmovaps [%33 + 2176] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %55 = x86.dm.vmovaps [%33 + 2240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %56 = x86.dm.vmovaps [%33 + 2560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %57 = x86.dm.vmovaps [%33 + 2624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %58 = x86.dm.vmovaps [%33 + 2688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %59 = x86.dm.vmovaps [%33 + 2752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %60 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90 = x86_scf.for %91 : !x86.reg64<r12>  = %60 to 128 : si32 step 4 : si32 iter_args(%92 = %31, %93 = %32, %94 = %33, %95 = %34, %96 = %35, %97 = %36, %98 = %37, %99 = %38, %100 = %39, %101 = %40, %102 = %41, %103 = %42, %104 = %43, %105 = %44, %106 = %45, %107 = %46, %108 = %47, %109 = %48, %110 = %49, %111 = %50, %112 = %51, %113 = %52, %114 = %53, %115 = %54, %116 = %55, %117 = %56, %118 = %57, %119 = %58, %120 = %59) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %121 = x86.dm.vmovaps [%92] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %122 = x86.dm.vmovaps [%92 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %123 = x86.dm.vmovaps [%92 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %124 = x86.dm.vmovaps [%92 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %125 = x86.dm.vbroadcastss [%93] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %126 = x86.rss.vfmadd231ps %97, %121, %125 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %127 = x86.rss.vfmadd231ps %98, %122, %125 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %128 = x86.rss.vfmadd231ps %99, %123, %125 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %129 = x86.rss.vfmadd231ps %100, %124, %125 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %130 = x86.dm.vbroadcastss [%93 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %131 = x86.rss.vfmadd231ps %101, %121, %130 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %132 = x86.rss.vfmadd231ps %102, %122, %130 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %133 = x86.rss.vfmadd231ps %103, %123, %130 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %134 = x86.rss.vfmadd231ps %104, %124, %130 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %135 = x86.dm.vbroadcastss [%93 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %136 = x86.rss.vfmadd231ps %105, %121, %135 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %137 = x86.rss.vfmadd231ps %106, %122, %135 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %138 = x86.rss.vfmadd231ps %107, %123, %135 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %139 = x86.rss.vfmadd231ps %108, %124, %135 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %140 = x86.dm.vbroadcastss [%93 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %141 = x86.rss.vfmadd231ps %109, %121, %140 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %142 = x86.rss.vfmadd231ps %110, %122, %140 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %143 = x86.rss.vfmadd231ps %111, %123, %140 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %144 = x86.rss.vfmadd231ps %112, %124, %140 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %145 = x86.dm.vbroadcastss [%93 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %146 = x86.rss.vfmadd231ps %113, %121, %145 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %147 = x86.rss.vfmadd231ps %114, %122, %145 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %148 = x86.rss.vfmadd231ps %115, %123, %145 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %149 = x86.rss.vfmadd231ps %116, %124, %145 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %150 = x86.dm.vbroadcastss [%93 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %151 = x86.ri.add %93, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %152 = x86.ri.add %92, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %153 = x86.rss.vfmadd231ps %117, %121, %150 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %154 = x86.rss.vfmadd231ps %118, %122, %150 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %155 = x86.rss.vfmadd231ps %119, %123, %150 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %156 = x86.rss.vfmadd231ps %120, %124, %150 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %157 = x86.dm.vmovaps [%152] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %158 = x86.dm.vmovaps [%152 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %159 = x86.dm.vmovaps [%152 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %160 = x86.dm.vmovaps [%152 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %161 = x86.dm.vbroadcastss [%151] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %162 = x86.rss.vfmadd231ps %126, %157, %161 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %163 = x86.rss.vfmadd231ps %127, %158, %161 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %164 = x86.rss.vfmadd231ps %128, %159, %161 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %165 = x86.rss.vfmadd231ps %129, %160, %161 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %166 = x86.dm.vbroadcastss [%151 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %167 = x86.rss.vfmadd231ps %131, %157, %166 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %168 = x86.rss.vfmadd231ps %132, %158, %166 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %169 = x86.rss.vfmadd231ps %133, %159, %166 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %170 = x86.rss.vfmadd231ps %134, %160, %166 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %171 = x86.dm.vbroadcastss [%151 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %172 = x86.rss.vfmadd231ps %136, %157, %171 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %173 = x86.rss.vfmadd231ps %137, %158, %171 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %174 = x86.rss.vfmadd231ps %138, %159, %171 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %175 = x86.rss.vfmadd231ps %139, %160, %171 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %176 = x86.dm.vbroadcastss [%151 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %177 = x86.rss.vfmadd231ps %141, %157, %176 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %178 = x86.rss.vfmadd231ps %142, %158, %176 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %179 = x86.rss.vfmadd231ps %143, %159, %176 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %180 = x86.rss.vfmadd231ps %144, %160, %176 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %181 = x86.dm.vbroadcastss [%151 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %182 = x86.rss.vfmadd231ps %146, %157, %181 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %183 = x86.rss.vfmadd231ps %147, %158, %181 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %184 = x86.rss.vfmadd231ps %148, %159, %181 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %185 = x86.rss.vfmadd231ps %149, %160, %181 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %186 = x86.dm.vbroadcastss [%151 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %187 = x86.ri.add %151, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %188 = x86.ri.add %152, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %189 = x86.rss.vfmadd231ps %153, %157, %186 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %190 = x86.rss.vfmadd231ps %154, %158, %186 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %191 = x86.rss.vfmadd231ps %155, %159, %186 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %192 = x86.rss.vfmadd231ps %156, %160, %186 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %193 = x86.dm.vmovaps [%188] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %194 = x86.dm.vmovaps [%188 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %195 = x86.dm.vmovaps [%188 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %196 = x86.dm.vmovaps [%188 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %197 = x86.dm.vbroadcastss [%187] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %198 = x86.rss.vfmadd231ps %162, %193, %197 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %199 = x86.rss.vfmadd231ps %163, %194, %197 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %200 = x86.rss.vfmadd231ps %164, %195, %197 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %201 = x86.rss.vfmadd231ps %165, %196, %197 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %202 = x86.dm.vbroadcastss [%187 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %203 = x86.rss.vfmadd231ps %167, %193, %202 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %204 = x86.rss.vfmadd231ps %168, %194, %202 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %205 = x86.rss.vfmadd231ps %169, %195, %202 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %206 = x86.rss.vfmadd231ps %170, %196, %202 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %207 = x86.dm.vbroadcastss [%187 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %208 = x86.rss.vfmadd231ps %172, %193, %207 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %209 = x86.rss.vfmadd231ps %173, %194, %207 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %210 = x86.rss.vfmadd231ps %174, %195, %207 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %211 = x86.rss.vfmadd231ps %175, %196, %207 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %212 = x86.dm.vbroadcastss [%187 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %213 = x86.rss.vfmadd231ps %177, %193, %212 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %214 = x86.rss.vfmadd231ps %178, %194, %212 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %215 = x86.rss.vfmadd231ps %179, %195, %212 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %216 = x86.rss.vfmadd231ps %180, %196, %212 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %217 = x86.dm.vbroadcastss [%187 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %218 = x86.rss.vfmadd231ps %182, %193, %217 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %219 = x86.rss.vfmadd231ps %183, %194, %217 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %220 = x86.rss.vfmadd231ps %184, %195, %217 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %221 = x86.rss.vfmadd231ps %185, %196, %217 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %222 = x86.dm.vbroadcastss [%187 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %223 = x86.ri.add %187, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %224 = x86.ri.add %188, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %225 = x86.rss.vfmadd231ps %189, %193, %222 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %226 = x86.rss.vfmadd231ps %190, %194, %222 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %227 = x86.rss.vfmadd231ps %191, %195, %222 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %228 = x86.rss.vfmadd231ps %192, %196, %222 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %229 = x86.dm.vmovaps [%224] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %230 = x86.dm.vmovaps [%224 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %231 = x86.dm.vmovaps [%224 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %232 = x86.dm.vmovaps [%224 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %233 = x86.dm.vbroadcastss [%223] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %234 = x86.rss.vfmadd231ps %198, %229, %233 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %235 = x86.rss.vfmadd231ps %199, %230, %233 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %236 = x86.rss.vfmadd231ps %200, %231, %233 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %237 = x86.rss.vfmadd231ps %201, %232, %233 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %238 = x86.dm.vbroadcastss [%223 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %239 = x86.rss.vfmadd231ps %203, %229, %238 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %240 = x86.rss.vfmadd231ps %204, %230, %238 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %241 = x86.rss.vfmadd231ps %205, %231, %238 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %242 = x86.rss.vfmadd231ps %206, %232, %238 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %243 = x86.dm.vbroadcastss [%223 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %244 = x86.rss.vfmadd231ps %208, %229, %243 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %245 = x86.rss.vfmadd231ps %209, %230, %243 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %246 = x86.rss.vfmadd231ps %210, %231, %243 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %247 = x86.rss.vfmadd231ps %211, %232, %243 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %248 = x86.dm.vbroadcastss [%223 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %249 = x86.rss.vfmadd231ps %213, %229, %248 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %250 = x86.rss.vfmadd231ps %214, %230, %248 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %251 = x86.rss.vfmadd231ps %215, %231, %248 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %252 = x86.rss.vfmadd231ps %216, %232, %248 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %253 = x86.dm.vbroadcastss [%223 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %254 = x86.rss.vfmadd231ps %218, %229, %253 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %255 = x86.rss.vfmadd231ps %219, %230, %253 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %256 = x86.rss.vfmadd231ps %220, %231, %253 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %257 = x86.rss.vfmadd231ps %221, %232, %253 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %258 = x86.dm.vbroadcastss [%223 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %259 = x86.ri.add %223, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %260 = x86.ri.add %224, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %261 = x86.rss.vfmadd231ps %225, %229, %258 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %262 = x86.rss.vfmadd231ps %226, %230, %258 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %263 = x86.rss.vfmadd231ps %227, %231, %258 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %264 = x86.rss.vfmadd231ps %228, %232, %258 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %260, %259, %94, %95, %96, %234, %235, %236, %237, %239, %240, %241, %242, %244, %245, %246, %247, %249, %250, %251, %252, %254, %255, %256, %257, %261, %262, %263, %264 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %265 = x86.ri.sub %63, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovaps [%64], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 64], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 128], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 192], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 512], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 576], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 640], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 704], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1024], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1088], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1152], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1216], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1536], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1600], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1664], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 1728], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2048], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2112], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2176], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2240], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2560], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2624], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2688], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%64 + 2752], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %266 = x86.ri.add %64, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %267 = x86.ri.sub %62, 65280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %267, %265, %266, %65, %66 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %268 = x86.ri.add %27, 2560 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %269 = x86.ri.add %26, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %270 = x86.ri.sub %25, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %270, %269, %268, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %271 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:  %272, %273, %274, %275, %276, %277 = x86_scf.for %278 : !x86.reg64<r11>  = %271 to 38 : si32 step 5 : si32 iter_args(%279 = %12, %280 = %13, %281 = %14, %282 = %15, %283 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:    %284 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %285, %286, %287, %288, %289, %290 = x86_scf.for %291 : !x86.reg64<r10>  = %284 to 128 : si32 step 64 : si32 iter_args(%292 = %279, %293 = %280, %294 = %281, %295 = %282, %296 = %283) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %297 = x86.dm.vmovaps [%294] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %298 = x86.dm.vmovaps [%294 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %299 = x86.dm.vmovaps [%294 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %300 = x86.dm.vmovaps [%294 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %301 = x86.dm.vmovaps [%294 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %302 = x86.dm.vmovaps [%294 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %303 = x86.dm.vmovaps [%294 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %304 = x86.dm.vmovaps [%294 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %305 = x86.dm.vmovaps [%294 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %306 = x86.dm.vmovaps [%294 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %307 = x86.dm.vmovaps [%294 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %308 = x86.dm.vmovaps [%294 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %309 = x86.dm.vmovaps [%294 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %310 = x86.dm.vmovaps [%294 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %311 = x86.dm.vmovaps [%294 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %312 = x86.dm.vmovaps [%294 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %313 = x86.dm.vmovaps [%294 + 2048] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %314 = x86.dm.vmovaps [%294 + 2112] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %315 = x86.dm.vmovaps [%294 + 2176] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %316 = x86.dm.vmovaps [%294 + 2240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %317 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %318, %319, %320, %321, %322, %323, %324, %325, %326, %327, %328, %329, %330, %331, %332, %333, %334, %335, %336, %337, %338, %339, %340, %341, %342, %343 = x86_scf.for %344 : !x86.reg64<r12>  = %317 to 128 : si32 step 4 : si32 iter_args(%345 = %292, %346 = %293, %347 = %294, %348 = %295, %349 = %296, %350 = %297, %351 = %298, %352 = %299, %353 = %300, %354 = %301, %355 = %302, %356 = %303, %357 = %304, %358 = %305, %359 = %306, %360 = %307, %361 = %308, %362 = %309, %363 = %310, %364 = %311, %365 = %312, %366 = %313, %367 = %314, %368 = %315, %369 = %316) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %370 = x86.dm.vmovaps [%345] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %371 = x86.dm.vmovaps [%345 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %372 = x86.dm.vmovaps [%345 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %373 = x86.dm.vmovaps [%345 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %374 = x86.dm.vbroadcastss [%346] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %375 = x86.rss.vfmadd231ps %350, %370, %374 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %376 = x86.rss.vfmadd231ps %351, %371, %374 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %377 = x86.rss.vfmadd231ps %352, %372, %374 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %378 = x86.rss.vfmadd231ps %353, %373, %374 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %379 = x86.dm.vbroadcastss [%346 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %380 = x86.rss.vfmadd231ps %354, %370, %379 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %381 = x86.rss.vfmadd231ps %355, %371, %379 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %382 = x86.rss.vfmadd231ps %356, %372, %379 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %383 = x86.rss.vfmadd231ps %357, %373, %379 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %384 = x86.dm.vbroadcastss [%346 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %385 = x86.rss.vfmadd231ps %358, %370, %384 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %386 = x86.rss.vfmadd231ps %359, %371, %384 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %387 = x86.rss.vfmadd231ps %360, %372, %384 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %388 = x86.rss.vfmadd231ps %361, %373, %384 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %389 = x86.dm.vbroadcastss [%346 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %390 = x86.rss.vfmadd231ps %362, %370, %389 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %391 = x86.rss.vfmadd231ps %363, %371, %389 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %392 = x86.rss.vfmadd231ps %364, %372, %389 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %393 = x86.rss.vfmadd231ps %365, %373, %389 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %394 = x86.dm.vbroadcastss [%346 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %395 = x86.ri.add %346, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %396 = x86.ri.add %345, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %397 = x86.rss.vfmadd231ps %366, %370, %394 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %398 = x86.rss.vfmadd231ps %367, %371, %394 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %399 = x86.rss.vfmadd231ps %368, %372, %394 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %400 = x86.rss.vfmadd231ps %369, %373, %394 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %401 = x86.dm.vmovaps [%396] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %402 = x86.dm.vmovaps [%396 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %403 = x86.dm.vmovaps [%396 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %404 = x86.dm.vmovaps [%396 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %405 = x86.dm.vbroadcastss [%395] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %406 = x86.rss.vfmadd231ps %375, %401, %405 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %407 = x86.rss.vfmadd231ps %376, %402, %405 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %408 = x86.rss.vfmadd231ps %377, %403, %405 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %409 = x86.rss.vfmadd231ps %378, %404, %405 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %410 = x86.dm.vbroadcastss [%395 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %411 = x86.rss.vfmadd231ps %380, %401, %410 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %412 = x86.rss.vfmadd231ps %381, %402, %410 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %413 = x86.rss.vfmadd231ps %382, %403, %410 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %414 = x86.rss.vfmadd231ps %383, %404, %410 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %415 = x86.dm.vbroadcastss [%395 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %416 = x86.rss.vfmadd231ps %385, %401, %415 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %417 = x86.rss.vfmadd231ps %386, %402, %415 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %418 = x86.rss.vfmadd231ps %387, %403, %415 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %419 = x86.rss.vfmadd231ps %388, %404, %415 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %420 = x86.dm.vbroadcastss [%395 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %421 = x86.rss.vfmadd231ps %390, %401, %420 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %422 = x86.rss.vfmadd231ps %391, %402, %420 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %423 = x86.rss.vfmadd231ps %392, %403, %420 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %424 = x86.rss.vfmadd231ps %393, %404, %420 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %425 = x86.dm.vbroadcastss [%395 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %426 = x86.ri.add %395, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %427 = x86.ri.add %396, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %428 = x86.rss.vfmadd231ps %397, %401, %425 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %429 = x86.rss.vfmadd231ps %398, %402, %425 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %430 = x86.rss.vfmadd231ps %399, %403, %425 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %431 = x86.rss.vfmadd231ps %400, %404, %425 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %432 = x86.dm.vmovaps [%427] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %433 = x86.dm.vmovaps [%427 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %434 = x86.dm.vmovaps [%427 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %435 = x86.dm.vmovaps [%427 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %436 = x86.dm.vbroadcastss [%426] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %437 = x86.rss.vfmadd231ps %406, %432, %436 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %438 = x86.rss.vfmadd231ps %407, %433, %436 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %439 = x86.rss.vfmadd231ps %408, %434, %436 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %440 = x86.rss.vfmadd231ps %409, %435, %436 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %441 = x86.dm.vbroadcastss [%426 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %442 = x86.rss.vfmadd231ps %411, %432, %441 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %443 = x86.rss.vfmadd231ps %412, %433, %441 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %444 = x86.rss.vfmadd231ps %413, %434, %441 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %445 = x86.rss.vfmadd231ps %414, %435, %441 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %446 = x86.dm.vbroadcastss [%426 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %447 = x86.rss.vfmadd231ps %416, %432, %446 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %448 = x86.rss.vfmadd231ps %417, %433, %446 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %449 = x86.rss.vfmadd231ps %418, %434, %446 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %450 = x86.rss.vfmadd231ps %419, %435, %446 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %451 = x86.dm.vbroadcastss [%426 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %452 = x86.rss.vfmadd231ps %421, %432, %451 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %453 = x86.rss.vfmadd231ps %422, %433, %451 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %454 = x86.rss.vfmadd231ps %423, %434, %451 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %455 = x86.rss.vfmadd231ps %424, %435, %451 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %456 = x86.dm.vbroadcastss [%426 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %457 = x86.ri.add %426, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %458 = x86.ri.add %427, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %459 = x86.rss.vfmadd231ps %428, %432, %456 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %460 = x86.rss.vfmadd231ps %429, %433, %456 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %461 = x86.rss.vfmadd231ps %430, %434, %456 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %462 = x86.rss.vfmadd231ps %431, %435, %456 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %463 = x86.dm.vmovaps [%458] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %464 = x86.dm.vmovaps [%458 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %465 = x86.dm.vmovaps [%458 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %466 = x86.dm.vmovaps [%458 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %467 = x86.dm.vbroadcastss [%457] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %468 = x86.rss.vfmadd231ps %437, %463, %467 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %469 = x86.rss.vfmadd231ps %438, %464, %467 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %470 = x86.rss.vfmadd231ps %439, %465, %467 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %471 = x86.rss.vfmadd231ps %440, %466, %467 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %472 = x86.dm.vbroadcastss [%457 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %473 = x86.rss.vfmadd231ps %442, %463, %472 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %474 = x86.rss.vfmadd231ps %443, %464, %472 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %475 = x86.rss.vfmadd231ps %444, %465, %472 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %476 = x86.rss.vfmadd231ps %445, %466, %472 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %477 = x86.dm.vbroadcastss [%457 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %478 = x86.rss.vfmadd231ps %447, %463, %477 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %479 = x86.rss.vfmadd231ps %448, %464, %477 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %480 = x86.rss.vfmadd231ps %449, %465, %477 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %481 = x86.rss.vfmadd231ps %450, %466, %477 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %482 = x86.dm.vbroadcastss [%457 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %483 = x86.rss.vfmadd231ps %452, %463, %482 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %484 = x86.rss.vfmadd231ps %453, %464, %482 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %485 = x86.rss.vfmadd231ps %454, %465, %482 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %486 = x86.rss.vfmadd231ps %455, %466, %482 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %487 = x86.dm.vbroadcastss [%457 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %488 = x86.ri.add %457, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %489 = x86.ri.add %458, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %490 = x86.rss.vfmadd231ps %459, %463, %487 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %491 = x86.rss.vfmadd231ps %460, %464, %487 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %492 = x86.rss.vfmadd231ps %461, %465, %487 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %493 = x86.rss.vfmadd231ps %462, %466, %487 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %489, %488, %347, %348, %349, %468, %469, %470, %471, %473, %474, %475, %476, %478, %479, %480, %481, %483, %484, %485, %486, %490, %491, %492, %493 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %494 = x86.ri.sub %320, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovaps [%321], %324 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 64], %325 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 128], %326 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 192], %327 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 512], %328 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 576], %329 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 640], %330 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 704], %331 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1024], %332 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1088], %333 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1152], %334 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1216], %335 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1536], %336 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1600], %337 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1664], %338 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 1728], %339 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 2048], %340 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 2112], %341 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 2176], %342 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovaps [%321 + 2240], %343 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %495 = x86.ri.add %321, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %496 = x86.ri.sub %319, 65280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %496, %494, %495, %322, %323 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %497 = x86.ri.add %288, 2048 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %498 = x86.ri.add %287, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %499 = x86.ri.sub %286, 512 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %499, %498, %497, %289, %290 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %500 = x86.ds.mov %276 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %501, %502 = x86.d.pop %500 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
