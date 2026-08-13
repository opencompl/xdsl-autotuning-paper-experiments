// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && cat %t | filecheck %s

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
// CHECK-NEXT:    %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 64 : si32 step 64 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %36 = x86.dm.vmovups [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %37 = x86.dm.vmovups [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %38 = x86.dm.vmovups [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %39 = x86.dm.vmovups [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %40 = x86.dm.vmovups [%33 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %41 = x86.dm.vmovups [%33 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %42 = x86.dm.vmovups [%33 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %43 = x86.dm.vmovups [%33 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %44 = x86.dm.vmovups [%33 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %45 = x86.dm.vmovups [%33 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %46 = x86.dm.vmovups [%33 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %47 = x86.dm.vmovups [%33 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %48 = x86.dm.vmovups [%33 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %49 = x86.dm.vmovups [%33 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %50 = x86.dm.vmovups [%33 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %51 = x86.dm.vmovups [%33 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %52 = x86.dm.vmovups [%33 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %53 = x86.dm.vmovups [%33 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %54 = x86.dm.vmovups [%33 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %55 = x86.dm.vmovups [%33 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %56 = x86.dm.vmovups [%33 + 1400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %57 = x86.dm.vmovups [%33 + 1464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %58 = x86.dm.vmovups [%33 + 1528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %59 = x86.dm.vmovups [%33 + 1592] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %60 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90 = x86_scf.for %91 : !x86.reg64<r12>  = %60 to 128 : si32 step 4 : si32 iter_args(%92 = %31, %93 = %32, %94 = %33, %95 = %34, %96 = %35, %97 = %36, %98 = %37, %99 = %38, %100 = %39, %101 = %40, %102 = %41, %103 = %42, %104 = %43, %105 = %44, %106 = %45, %107 = %46, %108 = %47, %109 = %48, %110 = %49, %111 = %50, %112 = %51, %113 = %52, %114 = %53, %115 = %54, %116 = %55, %117 = %56, %118 = %57, %119 = %58, %120 = %59) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %121 = x86.dm.vmovups [%92] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %122 = x86.dm.vmovups [%92 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %123 = x86.dm.vmovups [%92 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %124 = x86.dm.vmovups [%92 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-NEXT:        %152 = x86.ri.add %92, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %153 = x86.rss.vfmadd231ps %117, %121, %150 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %154 = x86.rss.vfmadd231ps %118, %122, %150 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %155 = x86.rss.vfmadd231ps %119, %123, %150 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %156 = x86.rss.vfmadd231ps %120, %124, %150 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %157 = x86.dm.vmovups [%152] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %158 = x86.dm.vmovups [%152 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %159 = x86.dm.vmovups [%152 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %160 = x86.dm.vmovups [%152 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-NEXT:        %188 = x86.ri.add %152, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %189 = x86.rss.vfmadd231ps %153, %157, %186 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %190 = x86.rss.vfmadd231ps %154, %158, %186 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %191 = x86.rss.vfmadd231ps %155, %159, %186 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %192 = x86.rss.vfmadd231ps %156, %160, %186 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %193 = x86.dm.vmovups [%188] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %194 = x86.dm.vmovups [%188 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %195 = x86.dm.vmovups [%188 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %196 = x86.dm.vmovups [%188 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-NEXT:        %224 = x86.ri.add %188, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %225 = x86.rss.vfmadd231ps %189, %193, %222 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %226 = x86.rss.vfmadd231ps %190, %194, %222 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %227 = x86.rss.vfmadd231ps %191, %195, %222 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %228 = x86.rss.vfmadd231ps %192, %196, %222 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %229 = x86.dm.vmovups [%224] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %230 = x86.dm.vmovups [%224 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %231 = x86.dm.vmovups [%224 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %232 = x86.dm.vmovups [%224 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
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
// CHECK-NEXT:        %260 = x86.ri.add %224, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %261 = x86.rss.vfmadd231ps %225, %229, %258 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %262 = x86.rss.vfmadd231ps %226, %230, %258 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %263 = x86.rss.vfmadd231ps %227, %231, %258 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %264 = x86.rss.vfmadd231ps %228, %232, %258 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %260, %259, %94, %95, %96, %234, %235, %236, %237, %239, %240, %241, %242, %244, %245, %246, %247, %249, %250, %251, %252, %254, %255, %256, %257, %261, %262, %263, %264 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %265 = x86.ri.sub %63, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%64], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 64], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 128], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 192], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 280], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 344], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 408], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 472], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 560], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 624], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 688], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 752], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 840], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 904], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 968], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1032], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1120], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1184], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1248], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1312], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1400], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1464], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1528], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%64 + 1592], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %266 = x86.ri.add %64, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %267 = x86.ri.sub %62, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %267, %265, %266, %65, %66 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %268 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:    %269 = x86.ks.kmovw %268 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %270 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %271, %272, %273, %274, %275, %276, %277 = x86_scf.for %278 : !x86.reg64<r10>  = %270 to 70 : si32 step 6 : si32 iter_args(%279 = %25, %280 = %26, %281 = %27, %282 = %28, %283 = %29, %284 = %269) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:      %285 = x86.dmk.vmovups[%281], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %286 = x86.dmk.vmovups[%281 + 280], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %287 = x86.dmk.vmovups[%281 + 560], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %288 = x86.dmk.vmovups[%281 + 840], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %289 = x86.dmk.vmovups[%281 + 1120], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %290 = x86.dmk.vmovups[%281 + 1400], %284 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %291 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304 = x86_scf.for %305 : !x86.reg64<r12>  = %291 to 128 : si32 step 4 : si32 iter_args(%306 = %279, %307 = %280, %308 = %281, %309 = %282, %310 = %283, %311 = %284, %312 = %285, %313 = %286, %314 = %287, %315 = %288, %316 = %289, %317 = %290) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %318 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-NEXT:        %319 = x86.dss.vpxord %318, %318 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %320 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-NEXT:        %321 = x86.dss.vpxord %320, %320 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %322 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-NEXT:        %323 = x86.dss.vpxord %322, %322 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %324 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-NEXT:        %325 = x86.dss.vpxord %324, %324 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %326 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-NEXT:        %327 = x86.dss.vpxord %326, %326 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %328 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-NEXT:        %329 = x86.dss.vpxord %328, %328 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %330 = x86.dmk.vmovups[%306], %311 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %331 = x86.dmk.vmovups[%306 + 280], %311 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %332 = x86.rsm.vfmadd231ps %312, %330, [%307] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %333 = x86.rsm.vfmadd231ps %313, %330, [%307 + 512] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %334 = x86.rsm.vfmadd231ps %314, %330, [%307 + 1024] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %335 = x86.rsm.vfmadd231ps %315, %330, [%307 + 1536] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %336 = x86.rsm.vfmadd231ps %316, %330, [%307 + 2048] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %337 = x86.rsm.vfmadd231ps %317, %330, [%307 + 2560] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %338 = x86.dmk.vmovups[%306 + 560], %311 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %339 = x86.rsm.vfmadd231ps %319, %331, [%307 + 4] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %340 = x86.rsm.vfmadd231ps %321, %331, [%307 + 516] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %341 = x86.rsm.vfmadd231ps %323, %331, [%307 + 1028] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %342 = x86.rsm.vfmadd231ps %325, %331, [%307 + 1540] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %343 = x86.rsm.vfmadd231ps %327, %331, [%307 + 2052] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %344 = x86.rsm.vfmadd231ps %329, %331, [%307 + 2564] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %345 = x86.dmk.vmovups[%306 + 840], %311 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %346 = x86.rsm.vfmadd231ps %332, %338, [%307 + 8] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %347 = x86.rsm.vfmadd231ps %333, %338, [%307 + 520] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %348 = x86.rsm.vfmadd231ps %334, %338, [%307 + 1032] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %349 = x86.rsm.vfmadd231ps %335, %338, [%307 + 1544] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %350 = x86.rsm.vfmadd231ps %336, %338, [%307 + 2056] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %351 = x86.rsm.vfmadd231ps %337, %338, [%307 + 2568] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %352 = x86.ri.add %306, 1120 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %353 = x86.rsm.vfmadd231ps %339, %345, [%307 + 12] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %354 = x86.rsm.vfmadd231ps %340, %345, [%307 + 524] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %355 = x86.rsm.vfmadd231ps %341, %345, [%307 + 1036] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %356 = x86.rsm.vfmadd231ps %342, %345, [%307 + 1548] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %357 = x86.rsm.vfmadd231ps %343, %345, [%307 + 2060] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %358 = x86.rsm.vfmadd231ps %344, %345, [%307 + 2572] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %359 = x86.ri.add %307, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %360 = x86.dss.vaddps %353, %346 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %361 = x86.dss.vaddps %354, %347 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %362 = x86.dss.vaddps %355, %348 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %363 = x86.dss.vaddps %356, %349 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %364 = x86.dss.vaddps %357, %350 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %365 = x86.dss.vaddps %358, %351 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %352, %359, %308, %309, %310, %311, %360, %361, %362, %363, %364, %365 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %366 = x86.ri.sub %294, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.msk.vmovups[%295], %299, %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%295 + 280], %300, %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%295 + 560], %301, %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%295 + 840], %302, %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%295 + 1120], %303, %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%295 + 1400], %304, %298 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %367 = x86.ri.add %295, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %368 = x86.ri.sub %293, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %368, %366, %367, %296, %297, %298 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %369 = x86.ri.add %274, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %370 = x86.ri.add %273, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %371 = x86.ri.sub %272, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %371, %370, %369, %275, %276 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %372 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:  %373, %374, %375, %376, %377, %378 = x86_scf.for %379 : !x86.reg64<r11>  = %372 to 38 : si32 step 5 : si32 iter_args(%380 = %12, %381 = %13, %382 = %14, %383 = %15, %384 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:    %385 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %386, %387, %388, %389, %390, %391 = x86_scf.for %392 : !x86.reg64<r10>  = %385 to 64 : si32 step 64 : si32 iter_args(%393 = %380, %394 = %381, %395 = %382, %396 = %383, %397 = %384) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %398 = x86.dm.vmovups [%395] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %399 = x86.dm.vmovups [%395 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %400 = x86.dm.vmovups [%395 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %401 = x86.dm.vmovups [%395 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %402 = x86.dm.vmovups [%395 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %403 = x86.dm.vmovups [%395 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %404 = x86.dm.vmovups [%395 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %405 = x86.dm.vmovups [%395 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %406 = x86.dm.vmovups [%395 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %407 = x86.dm.vmovups [%395 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %408 = x86.dm.vmovups [%395 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %409 = x86.dm.vmovups [%395 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %410 = x86.dm.vmovups [%395 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %411 = x86.dm.vmovups [%395 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %412 = x86.dm.vmovups [%395 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %413 = x86.dm.vmovups [%395 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %414 = x86.dm.vmovups [%395 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %415 = x86.dm.vmovups [%395 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %416 = x86.dm.vmovups [%395 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %417 = x86.dm.vmovups [%395 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %418 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %419, %420, %421, %422, %423, %424, %425, %426, %427, %428, %429, %430, %431, %432, %433, %434, %435, %436, %437, %438, %439, %440, %441, %442, %443, %444 = x86_scf.for %445 : !x86.reg64<r12>  = %418 to 128 : si32 step 4 : si32 iter_args(%446 = %393, %447 = %394, %448 = %395, %449 = %396, %450 = %397, %451 = %398, %452 = %399, %453 = %400, %454 = %401, %455 = %402, %456 = %403, %457 = %404, %458 = %405, %459 = %406, %460 = %407, %461 = %408, %462 = %409, %463 = %410, %464 = %411, %465 = %412, %466 = %413, %467 = %414, %468 = %415, %469 = %416, %470 = %417) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %471 = x86.dm.vmovups [%446] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %472 = x86.dm.vmovups [%446 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %473 = x86.dm.vmovups [%446 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %474 = x86.dm.vmovups [%446 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %475 = x86.dm.vbroadcastss [%447] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %476 = x86.rss.vfmadd231ps %451, %471, %475 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %477 = x86.rss.vfmadd231ps %452, %472, %475 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %478 = x86.rss.vfmadd231ps %453, %473, %475 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %479 = x86.rss.vfmadd231ps %454, %474, %475 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %480 = x86.dm.vbroadcastss [%447 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %481 = x86.rss.vfmadd231ps %455, %471, %480 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %482 = x86.rss.vfmadd231ps %456, %472, %480 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %483 = x86.rss.vfmadd231ps %457, %473, %480 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %484 = x86.rss.vfmadd231ps %458, %474, %480 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %485 = x86.dm.vbroadcastss [%447 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %486 = x86.rss.vfmadd231ps %459, %471, %485 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %487 = x86.rss.vfmadd231ps %460, %472, %485 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %488 = x86.rss.vfmadd231ps %461, %473, %485 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %489 = x86.rss.vfmadd231ps %462, %474, %485 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %490 = x86.dm.vbroadcastss [%447 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %491 = x86.rss.vfmadd231ps %463, %471, %490 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %492 = x86.rss.vfmadd231ps %464, %472, %490 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %493 = x86.rss.vfmadd231ps %465, %473, %490 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %494 = x86.rss.vfmadd231ps %466, %474, %490 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %495 = x86.dm.vbroadcastss [%447 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %496 = x86.ri.add %447, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %497 = x86.ri.add %446, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %498 = x86.rss.vfmadd231ps %467, %471, %495 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %499 = x86.rss.vfmadd231ps %468, %472, %495 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %500 = x86.rss.vfmadd231ps %469, %473, %495 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %501 = x86.rss.vfmadd231ps %470, %474, %495 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %502 = x86.dm.vmovups [%497] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %503 = x86.dm.vmovups [%497 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %504 = x86.dm.vmovups [%497 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %505 = x86.dm.vmovups [%497 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %506 = x86.dm.vbroadcastss [%496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %507 = x86.rss.vfmadd231ps %476, %502, %506 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %508 = x86.rss.vfmadd231ps %477, %503, %506 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %509 = x86.rss.vfmadd231ps %478, %504, %506 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %510 = x86.rss.vfmadd231ps %479, %505, %506 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %511 = x86.dm.vbroadcastss [%496 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %512 = x86.rss.vfmadd231ps %481, %502, %511 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %513 = x86.rss.vfmadd231ps %482, %503, %511 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %514 = x86.rss.vfmadd231ps %483, %504, %511 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %515 = x86.rss.vfmadd231ps %484, %505, %511 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %516 = x86.dm.vbroadcastss [%496 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %517 = x86.rss.vfmadd231ps %486, %502, %516 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %518 = x86.rss.vfmadd231ps %487, %503, %516 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %519 = x86.rss.vfmadd231ps %488, %504, %516 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %520 = x86.rss.vfmadd231ps %489, %505, %516 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %521 = x86.dm.vbroadcastss [%496 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %522 = x86.rss.vfmadd231ps %491, %502, %521 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %523 = x86.rss.vfmadd231ps %492, %503, %521 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %524 = x86.rss.vfmadd231ps %493, %504, %521 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %525 = x86.rss.vfmadd231ps %494, %505, %521 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %526 = x86.dm.vbroadcastss [%496 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %527 = x86.ri.add %496, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %528 = x86.ri.add %497, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %529 = x86.rss.vfmadd231ps %498, %502, %526 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %530 = x86.rss.vfmadd231ps %499, %503, %526 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %531 = x86.rss.vfmadd231ps %500, %504, %526 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %532 = x86.rss.vfmadd231ps %501, %505, %526 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %533 = x86.dm.vmovups [%528] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %534 = x86.dm.vmovups [%528 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %535 = x86.dm.vmovups [%528 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %536 = x86.dm.vmovups [%528 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %537 = x86.dm.vbroadcastss [%527] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %538 = x86.rss.vfmadd231ps %507, %533, %537 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %539 = x86.rss.vfmadd231ps %508, %534, %537 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %540 = x86.rss.vfmadd231ps %509, %535, %537 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %541 = x86.rss.vfmadd231ps %510, %536, %537 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %542 = x86.dm.vbroadcastss [%527 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %543 = x86.rss.vfmadd231ps %512, %533, %542 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %544 = x86.rss.vfmadd231ps %513, %534, %542 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %545 = x86.rss.vfmadd231ps %514, %535, %542 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %546 = x86.rss.vfmadd231ps %515, %536, %542 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %547 = x86.dm.vbroadcastss [%527 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %548 = x86.rss.vfmadd231ps %517, %533, %547 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %549 = x86.rss.vfmadd231ps %518, %534, %547 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %550 = x86.rss.vfmadd231ps %519, %535, %547 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %551 = x86.rss.vfmadd231ps %520, %536, %547 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %552 = x86.dm.vbroadcastss [%527 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %553 = x86.rss.vfmadd231ps %522, %533, %552 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %554 = x86.rss.vfmadd231ps %523, %534, %552 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %555 = x86.rss.vfmadd231ps %524, %535, %552 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %556 = x86.rss.vfmadd231ps %525, %536, %552 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %557 = x86.dm.vbroadcastss [%527 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %558 = x86.ri.add %527, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %559 = x86.ri.add %528, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %560 = x86.rss.vfmadd231ps %529, %533, %557 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %561 = x86.rss.vfmadd231ps %530, %534, %557 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %562 = x86.rss.vfmadd231ps %531, %535, %557 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %563 = x86.rss.vfmadd231ps %532, %536, %557 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %564 = x86.dm.vmovups [%559] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %565 = x86.dm.vmovups [%559 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %566 = x86.dm.vmovups [%559 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %567 = x86.dm.vmovups [%559 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %568 = x86.dm.vbroadcastss [%558] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %569 = x86.rss.vfmadd231ps %538, %564, %568 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %570 = x86.rss.vfmadd231ps %539, %565, %568 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %571 = x86.rss.vfmadd231ps %540, %566, %568 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %572 = x86.rss.vfmadd231ps %541, %567, %568 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %573 = x86.dm.vbroadcastss [%558 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %574 = x86.rss.vfmadd231ps %543, %564, %573 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %575 = x86.rss.vfmadd231ps %544, %565, %573 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %576 = x86.rss.vfmadd231ps %545, %566, %573 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %577 = x86.rss.vfmadd231ps %546, %567, %573 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %578 = x86.dm.vbroadcastss [%558 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %579 = x86.rss.vfmadd231ps %548, %564, %578 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %580 = x86.rss.vfmadd231ps %549, %565, %578 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %581 = x86.rss.vfmadd231ps %550, %566, %578 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %582 = x86.rss.vfmadd231ps %551, %567, %578 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %583 = x86.dm.vbroadcastss [%558 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %584 = x86.rss.vfmadd231ps %553, %564, %583 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %585 = x86.rss.vfmadd231ps %554, %565, %583 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %586 = x86.rss.vfmadd231ps %555, %566, %583 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %587 = x86.rss.vfmadd231ps %556, %567, %583 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %588 = x86.dm.vbroadcastss [%558 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %589 = x86.ri.add %558, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %590 = x86.ri.add %559, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %591 = x86.rss.vfmadd231ps %560, %564, %588 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %592 = x86.rss.vfmadd231ps %561, %565, %588 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %593 = x86.rss.vfmadd231ps %562, %566, %588 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %594 = x86.rss.vfmadd231ps %563, %567, %588 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %590, %589, %448, %449, %450, %569, %570, %571, %572, %574, %575, %576, %577, %579, %580, %581, %582, %584, %585, %586, %587, %591, %592, %593, %594 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %595 = x86.ri.sub %421, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovups [%422], %425 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 64], %426 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 128], %427 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 192], %428 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 280], %429 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 344], %430 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 408], %431 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 472], %432 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 560], %433 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 624], %434 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 688], %435 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 752], %436 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 840], %437 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 904], %438 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 968], %439 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 1032], %440 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 1120], %441 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 1184], %442 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 1248], %443 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovups [%422 + 1312], %444 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %596 = x86.ri.add %422, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %597 = x86.ri.sub %420, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %597, %595, %596, %423, %424 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %598 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:    %599 = x86.ks.kmovw %598 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:    %600 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %601, %602, %603, %604, %605, %606, %607 = x86_scf.for %608 : !x86.reg64<r10>  = %600 to 70 : si32 step 6 : si32 iter_args(%609 = %387, %610 = %388, %611 = %389, %612 = %390, %613 = %391, %614 = %599) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:      %615 = x86.dmk.vmovups[%611], %614 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %616 = x86.dmk.vmovups[%611 + 280], %614 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %617 = x86.dmk.vmovups[%611 + 560], %614 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %618 = x86.dmk.vmovups[%611 + 840], %614 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %619 = x86.dmk.vmovups[%611 + 1120], %614 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %620 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %621, %622, %623, %624, %625, %626, %627, %628, %629, %630, %631, %632 = x86_scf.for %633 : !x86.reg64<r12>  = %620 to 128 : si32 step 4 : si32 iter_args(%634 = %609, %635 = %610, %636 = %611, %637 = %612, %638 = %613, %639 = %614, %640 = %615, %641 = %616, %642 = %617, %643 = %618, %644 = %619) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %645 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-NEXT:        %646 = x86.dss.vpxord %645, %645 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %647 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-NEXT:        %648 = x86.dss.vpxord %647, %647 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %649 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-NEXT:        %650 = x86.dss.vpxord %649, %649 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %651 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-NEXT:        %652 = x86.dss.vpxord %651, %651 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %653 = x86.get_avx_register : !x86.avx512reg<zmm26>
// CHECK-NEXT:        %654 = x86.dss.vpxord %653, %653 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %655 = x86.get_avx_register : !x86.avx512reg<zmm17>
// CHECK-NEXT:        %656 = x86.dss.vpxord %655, %655 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm17>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %657 = x86.get_avx_register : !x86.avx512reg<zmm18>
// CHECK-NEXT:        %658 = x86.dss.vpxord %657, %657 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm18>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %659 = x86.get_avx_register : !x86.avx512reg<zmm19>
// CHECK-NEXT:        %660 = x86.dss.vpxord %659, %659 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm19>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %661 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-NEXT:        %662 = x86.dss.vpxord %661, %661 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %663 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-NEXT:        %664 = x86.dss.vpxord %663, %663 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %665 = x86.get_avx_register : !x86.avx512reg<zmm12>
// CHECK-NEXT:        %666 = x86.dss.vpxord %665, %665 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm12>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %667 = x86.get_avx_register : !x86.avx512reg<zmm13>
// CHECK-NEXT:        %668 = x86.dss.vpxord %667, %667 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm13>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %669 = x86.get_avx_register : !x86.avx512reg<zmm14>
// CHECK-NEXT:        %670 = x86.dss.vpxord %669, %669 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm14>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %671 = x86.get_avx_register : !x86.avx512reg<zmm15>
// CHECK-NEXT:        %672 = x86.dss.vpxord %671, %671 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm15>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %673 = x86.get_avx_register : !x86.avx512reg<zmm16>
// CHECK-NEXT:        %674 = x86.dss.vpxord %673, %673 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm16>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %675 = x86.dmk.vmovups[%634], %639 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %676 = x86.dmk.vmovups[%634 + 280], %639 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %677 = x86.rsm.vfmadd231ps %640, %675, [%635] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %678 = x86.rsm.vfmadd231ps %641, %675, [%635 + 512] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %679 = x86.rsm.vfmadd231ps %642, %675, [%635 + 1024] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %680 = x86.rsm.vfmadd231ps %643, %675, [%635 + 1536] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %681 = x86.rsm.vfmadd231ps %644, %675, [%635 + 2048] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %682 = x86.dmk.vmovups[%634 + 560], %639 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %683 = x86.rsm.vfmadd231ps %646, %676, [%635 + 4] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %684 = x86.rsm.vfmadd231ps %648, %676, [%635 + 516] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %685 = x86.rsm.vfmadd231ps %650, %676, [%635 + 1028] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %686 = x86.rsm.vfmadd231ps %652, %676, [%635 + 1540] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %687 = x86.rsm.vfmadd231ps %654, %676, [%635 + 2052] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %688 = x86.dmk.vmovups[%634 + 840], %639 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %689 = x86.rsm.vfmadd231ps %656, %682, [%635 + 8] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %690 = x86.rsm.vfmadd231ps %658, %682, [%635 + 520] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %691 = x86.rsm.vfmadd231ps %660, %682, [%635 + 1032] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %692 = x86.rsm.vfmadd231ps %662, %682, [%635 + 1544] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %693 = x86.rsm.vfmadd231ps %664, %682, [%635 + 2056] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %694 = x86.ri.add %634, 1120 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %695 = x86.rsm.vfmadd231ps %666, %688, [%635 + 12] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %696 = x86.rsm.vfmadd231ps %668, %688, [%635 + 524] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %697 = x86.rsm.vfmadd231ps %670, %688, [%635 + 1036] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %698 = x86.rsm.vfmadd231ps %672, %688, [%635 + 1548] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %699 = x86.rsm.vfmadd231ps %674, %688, [%635 + 2060] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %700 = x86.ri.add %635, 16 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %701 = x86.dss.vaddps %683, %677 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %702 = x86.dss.vaddps %684, %678 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %703 = x86.dss.vaddps %685, %679 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %704 = x86.dss.vaddps %686, %680 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %705 = x86.dss.vaddps %687, %681 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %706 = x86.dss.vaddps %689, %701 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %707 = x86.dss.vaddps %690, %702 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %708 = x86.dss.vaddps %691, %703 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %709 = x86.dss.vaddps %692, %704 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %710 = x86.dss.vaddps %693, %705 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %711 = x86.dss.vaddps %695, %706 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %712 = x86.dss.vaddps %696, %707 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %713 = x86.dss.vaddps %697, %708 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %714 = x86.dss.vaddps %698, %709 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %715 = x86.dss.vaddps %699, %710 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %694, %700, %636, %637, %638, %639, %711, %712, %713, %714, %715 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %716 = x86.ri.sub %623, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.msk.vmovups[%624], %628, %627 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%624 + 280], %629, %627 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%624 + 560], %630, %627 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%624 + 840], %631, %627 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      x86.msk.vmovups[%624 + 1120], %632, %627 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:      %717 = x86.ri.add %624, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %718 = x86.ri.sub %622, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %718, %716, %717, %625, %626, %627 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:    }
// CHECK-NEXT:    %719 = x86.ri.add %604, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %720 = x86.ri.add %603, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %721 = x86.ri.sub %602, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %721, %720, %719, %605, %606 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %722 = x86.ds.mov %377 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %723, %724 = x86.d.pop %722 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:  x86_func.ret
// CHECK-NEXT:}
