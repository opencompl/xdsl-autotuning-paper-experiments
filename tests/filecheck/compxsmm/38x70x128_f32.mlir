// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && cat %t | filecheck %s

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
// CHECK-NEXT:      %22 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %23, %24, %25, %26, %27 = x86_scf.for %28 : !x86.reg64<r10>  = %22 to 64 : si32 step 64 : si32 iter_args(%29 = %17, %30 = %18, %31 = %19, %32 = %20, %33 = %21) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %34 = x86.dm.vmovups [%31] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %35 = x86.dm.vmovups [%31 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %36 = x86.dm.vmovups [%31 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %37 = x86.dm.vmovups [%31 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %38 = x86.dm.vmovups [%31 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %39 = x86.dm.vmovups [%31 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %40 = x86.dm.vmovups [%31 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %41 = x86.dm.vmovups [%31 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %42 = x86.dm.vmovups [%31 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %43 = x86.dm.vmovups [%31 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %44 = x86.dm.vmovups [%31 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %45 = x86.dm.vmovups [%31 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %46 = x86.dm.vmovups [%31 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %47 = x86.dm.vmovups [%31 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %48 = x86.dm.vmovups [%31 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %49 = x86.dm.vmovups [%31 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %50 = x86.dm.vmovups [%31 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %51 = x86.dm.vmovups [%31 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %52 = x86.dm.vmovups [%31 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %53 = x86.dm.vmovups [%31 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %54 = x86.dm.vmovups [%31 + 1400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %55 = x86.dm.vmovups [%31 + 1464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %56 = x86.dm.vmovups [%31 + 1528] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %57 = x86.dm.vmovups [%31 + 1592] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %58 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87 = x86_scf.for %88 : !x86.reg64<r12>  = %58 to 128 : si32 step 4 : si32 iter_args(%89 = %29, %90 = %30, %91 = %31, %92 = %32, %93 = %33, %94 = %34, %95 = %35, %96 = %36, %97 = %37, %98 = %38, %99 = %39, %100 = %40, %101 = %41, %102 = %42, %103 = %43, %104 = %44, %105 = %45, %106 = %46, %107 = %47, %108 = %48, %109 = %49, %110 = %50, %111 = %51, %112 = %52, %113 = %53, %114 = %54, %115 = %55, %116 = %56, %117 = %57) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %118 = x86.dm.vmovups [%89] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %119 = x86.dm.vmovups [%89 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %120 = x86.dm.vmovups [%89 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %121 = x86.dm.vmovups [%89 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %122 = x86.dm.vbroadcastss [%90] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %123 = x86.rss.vfmadd231ps %94, %118, %122 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %124 = x86.rss.vfmadd231ps %95, %119, %122 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %125 = x86.rss.vfmadd231ps %96, %120, %122 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %126 = x86.rss.vfmadd231ps %97, %121, %122 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %127 = x86.dm.vbroadcastss [%90 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %128 = x86.rss.vfmadd231ps %98, %118, %127 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %129 = x86.rss.vfmadd231ps %99, %119, %127 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %130 = x86.rss.vfmadd231ps %100, %120, %127 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %131 = x86.rss.vfmadd231ps %101, %121, %127 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %132 = x86.dm.vbroadcastss [%90 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %133 = x86.rss.vfmadd231ps %102, %118, %132 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %134 = x86.rss.vfmadd231ps %103, %119, %132 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %135 = x86.rss.vfmadd231ps %104, %120, %132 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %136 = x86.rss.vfmadd231ps %105, %121, %132 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %137 = x86.dm.vbroadcastss [%90 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %138 = x86.rss.vfmadd231ps %106, %118, %137 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %139 = x86.rss.vfmadd231ps %107, %119, %137 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %140 = x86.rss.vfmadd231ps %108, %120, %137 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %141 = x86.rss.vfmadd231ps %109, %121, %137 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %142 = x86.dm.vbroadcastss [%90 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %143 = x86.rss.vfmadd231ps %110, %118, %142 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %144 = x86.rss.vfmadd231ps %111, %119, %142 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %145 = x86.rss.vfmadd231ps %112, %120, %142 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %146 = x86.rss.vfmadd231ps %113, %121, %142 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %147 = x86.dm.vbroadcastss [%90 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %148 = x86.ri.add %90, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %149 = x86.ri.add %89, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %150 = x86.rss.vfmadd231ps %114, %118, %147 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %151 = x86.rss.vfmadd231ps %115, %119, %147 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %152 = x86.rss.vfmadd231ps %116, %120, %147 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %153 = x86.rss.vfmadd231ps %117, %121, %147 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %154 = x86.dm.vmovups [%149] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %155 = x86.dm.vmovups [%149 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %156 = x86.dm.vmovups [%149 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %157 = x86.dm.vmovups [%149 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %158 = x86.dm.vbroadcastss [%148] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %159 = x86.rss.vfmadd231ps %123, %154, %158 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %160 = x86.rss.vfmadd231ps %124, %155, %158 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %161 = x86.rss.vfmadd231ps %125, %156, %158 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %162 = x86.rss.vfmadd231ps %126, %157, %158 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %163 = x86.dm.vbroadcastss [%148 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %164 = x86.rss.vfmadd231ps %128, %154, %163 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %165 = x86.rss.vfmadd231ps %129, %155, %163 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %166 = x86.rss.vfmadd231ps %130, %156, %163 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %167 = x86.rss.vfmadd231ps %131, %157, %163 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %168 = x86.dm.vbroadcastss [%148 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %169 = x86.rss.vfmadd231ps %133, %154, %168 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %170 = x86.rss.vfmadd231ps %134, %155, %168 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %171 = x86.rss.vfmadd231ps %135, %156, %168 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %172 = x86.rss.vfmadd231ps %136, %157, %168 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %173 = x86.dm.vbroadcastss [%148 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %174 = x86.rss.vfmadd231ps %138, %154, %173 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %175 = x86.rss.vfmadd231ps %139, %155, %173 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %176 = x86.rss.vfmadd231ps %140, %156, %173 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %177 = x86.rss.vfmadd231ps %141, %157, %173 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %178 = x86.dm.vbroadcastss [%148 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %179 = x86.rss.vfmadd231ps %143, %154, %178 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %180 = x86.rss.vfmadd231ps %144, %155, %178 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %181 = x86.rss.vfmadd231ps %145, %156, %178 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %182 = x86.rss.vfmadd231ps %146, %157, %178 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %183 = x86.dm.vbroadcastss [%148 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %184 = x86.ri.add %148, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %185 = x86.ri.add %149, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %186 = x86.rss.vfmadd231ps %150, %154, %183 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %187 = x86.rss.vfmadd231ps %151, %155, %183 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %188 = x86.rss.vfmadd231ps %152, %156, %183 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %189 = x86.rss.vfmadd231ps %153, %157, %183 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %190 = x86.dm.vmovups [%185] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %191 = x86.dm.vmovups [%185 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %192 = x86.dm.vmovups [%185 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %193 = x86.dm.vmovups [%185 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %194 = x86.dm.vbroadcastss [%184] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %195 = x86.rss.vfmadd231ps %159, %190, %194 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %196 = x86.rss.vfmadd231ps %160, %191, %194 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %197 = x86.rss.vfmadd231ps %161, %192, %194 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %198 = x86.rss.vfmadd231ps %162, %193, %194 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %199 = x86.dm.vbroadcastss [%184 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %200 = x86.rss.vfmadd231ps %164, %190, %199 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %201 = x86.rss.vfmadd231ps %165, %191, %199 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %202 = x86.rss.vfmadd231ps %166, %192, %199 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %203 = x86.rss.vfmadd231ps %167, %193, %199 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %204 = x86.dm.vbroadcastss [%184 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %205 = x86.rss.vfmadd231ps %169, %190, %204 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %206 = x86.rss.vfmadd231ps %170, %191, %204 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %207 = x86.rss.vfmadd231ps %171, %192, %204 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %208 = x86.rss.vfmadd231ps %172, %193, %204 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %209 = x86.dm.vbroadcastss [%184 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %210 = x86.rss.vfmadd231ps %174, %190, %209 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %211 = x86.rss.vfmadd231ps %175, %191, %209 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %212 = x86.rss.vfmadd231ps %176, %192, %209 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %213 = x86.rss.vfmadd231ps %177, %193, %209 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %214 = x86.dm.vbroadcastss [%184 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %215 = x86.rss.vfmadd231ps %179, %190, %214 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %216 = x86.rss.vfmadd231ps %180, %191, %214 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %217 = x86.rss.vfmadd231ps %181, %192, %214 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %218 = x86.rss.vfmadd231ps %182, %193, %214 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %219 = x86.dm.vbroadcastss [%184 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %220 = x86.ri.add %184, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %221 = x86.ri.add %185, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %222 = x86.rss.vfmadd231ps %186, %190, %219 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %223 = x86.rss.vfmadd231ps %187, %191, %219 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %224 = x86.rss.vfmadd231ps %188, %192, %219 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %225 = x86.rss.vfmadd231ps %189, %193, %219 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %226 = x86.dm.vmovups [%221] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %227 = x86.dm.vmovups [%221 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %228 = x86.dm.vmovups [%221 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %229 = x86.dm.vmovups [%221 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %230 = x86.dm.vbroadcastss [%220] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %231 = x86.rss.vfmadd231ps %195, %226, %230 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %232 = x86.rss.vfmadd231ps %196, %227, %230 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %233 = x86.rss.vfmadd231ps %197, %228, %230 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %234 = x86.rss.vfmadd231ps %198, %229, %230 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %235 = x86.dm.vbroadcastss [%220 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %236 = x86.rss.vfmadd231ps %200, %226, %235 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %237 = x86.rss.vfmadd231ps %201, %227, %235 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %238 = x86.rss.vfmadd231ps %202, %228, %235 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %239 = x86.rss.vfmadd231ps %203, %229, %235 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %240 = x86.dm.vbroadcastss [%220 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %241 = x86.rss.vfmadd231ps %205, %226, %240 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %242 = x86.rss.vfmadd231ps %206, %227, %240 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %243 = x86.rss.vfmadd231ps %207, %228, %240 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %244 = x86.rss.vfmadd231ps %208, %229, %240 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %245 = x86.dm.vbroadcastss [%220 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %246 = x86.rss.vfmadd231ps %210, %226, %245 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %247 = x86.rss.vfmadd231ps %211, %227, %245 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %248 = x86.rss.vfmadd231ps %212, %228, %245 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %249 = x86.rss.vfmadd231ps %213, %229, %245 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %250 = x86.dm.vbroadcastss [%220 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %251 = x86.rss.vfmadd231ps %215, %226, %250 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %252 = x86.rss.vfmadd231ps %216, %227, %250 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %253 = x86.rss.vfmadd231ps %217, %228, %250 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %254 = x86.rss.vfmadd231ps %218, %229, %250 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %255 = x86.dm.vbroadcastss [%220 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %256 = x86.ri.add %220, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %257 = x86.ri.add %221, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %258 = x86.rss.vfmadd231ps %222, %226, %255 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %259 = x86.rss.vfmadd231ps %223, %227, %255 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %260 = x86.rss.vfmadd231ps %224, %228, %255 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %261 = x86.rss.vfmadd231ps %225, %229, %255 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %257, %256, %91, %92, %93, %231, %232, %233, %234, %236, %237, %238, %239, %241, %242, %243, %244, %246, %247, %248, %249, %251, %252, %253, %254, %258, %259, %260, %261 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %262 = x86.ri.sub %60, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%61], %64 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 64], %65 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 128], %66 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 192], %67 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 280], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 344], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 408], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 472], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 560], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 624], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 688], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 752], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 840], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 904], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 968], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1032], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1120], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1184], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1248], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1312], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1400], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1464], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1528], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%61 + 1592], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %263 = x86.ri.add %61, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %264 = x86.ri.sub %59, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %264, %262, %263, %62, %63 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %265 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %266 = x86.ks.kmovw %265 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %267 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %268, %269, %270, %271, %272, %273 = x86_scf.for %274 : !x86.reg64<r10>  = %267 to 70 : si32 step 6 : si32 iter_args(%275 = %23, %276 = %24, %277 = %25, %278 = %26, %279 = %27, %280 = %266) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %281 = x86.dmk.vmovups[%277], %280 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %282 = x86.dmk.vmovups[%277 + 280], %280 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %283 = x86.dmk.vmovups[%277 + 560], %280 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %284 = x86.dmk.vmovups[%277 + 840], %280 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %285 = x86.dmk.vmovups[%277 + 1120], %280 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %286 = x86.dmk.vmovups[%277 + 1400], %280 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %287 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %288, %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299 = x86_scf.for %300 : !x86.reg64<r12>  = %287 to 128 : si32 step 4 : si32 iter_args(%301 = %275, %302 = %276, %303 = %277, %304 = %278, %305 = %279, %306 = %280, %307 = %281, %308 = %282, %309 = %283, %310 = %284, %311 = %285, %312 = %286) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %313 = x86.dmk.vmovups[%301], %306 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %314 = x86.dm.vbroadcastss [%302] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %315 = x86.rss.vfmadd231ps %307, %313, %314 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %316 = x86.dm.vbroadcastss [%302 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %317 = x86.rss.vfmadd231ps %308, %313, %316 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %318 = x86.dm.vbroadcastss [%302 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %319 = x86.rss.vfmadd231ps %309, %313, %318 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %320 = x86.dm.vbroadcastss [%302 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %321 = x86.rss.vfmadd231ps %310, %313, %320 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %322 = x86.dm.vbroadcastss [%302 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %323 = x86.rss.vfmadd231ps %311, %313, %322 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %324 = x86.dm.vbroadcastss [%302 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %325 = x86.ri.add %302, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %326 = x86.ri.add %301, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %327 = x86.rss.vfmadd231ps %312, %313, %324 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %328 = x86.dmk.vmovups[%326], %306 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %329 = x86.dm.vbroadcastss [%325] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %330 = x86.rss.vfmadd231ps %315, %328, %329 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %331 = x86.dm.vbroadcastss [%325 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %332 = x86.rss.vfmadd231ps %317, %328, %331 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %333 = x86.dm.vbroadcastss [%325 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %334 = x86.rss.vfmadd231ps %319, %328, %333 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %335 = x86.dm.vbroadcastss [%325 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %336 = x86.rss.vfmadd231ps %321, %328, %335 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %337 = x86.dm.vbroadcastss [%325 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %338 = x86.rss.vfmadd231ps %323, %328, %337 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %339 = x86.dm.vbroadcastss [%325 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %340 = x86.ri.add %325, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %341 = x86.ri.add %326, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %342 = x86.rss.vfmadd231ps %327, %328, %339 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %343 = x86.dmk.vmovups[%341], %306 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %344 = x86.dm.vbroadcastss [%340] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %345 = x86.rss.vfmadd231ps %330, %343, %344 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %346 = x86.dm.vbroadcastss [%340 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %347 = x86.rss.vfmadd231ps %332, %343, %346 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %348 = x86.dm.vbroadcastss [%340 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %349 = x86.rss.vfmadd231ps %334, %343, %348 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %350 = x86.dm.vbroadcastss [%340 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %351 = x86.rss.vfmadd231ps %336, %343, %350 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %352 = x86.dm.vbroadcastss [%340 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %353 = x86.rss.vfmadd231ps %338, %343, %352 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %354 = x86.dm.vbroadcastss [%340 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %355 = x86.ri.add %340, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %356 = x86.ri.add %341, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %357 = x86.rss.vfmadd231ps %342, %343, %354 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %358 = x86.dmk.vmovups[%356], %306 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %359 = x86.dm.vbroadcastss [%355] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %360 = x86.rss.vfmadd231ps %345, %358, %359 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %361 = x86.dm.vbroadcastss [%355 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %362 = x86.rss.vfmadd231ps %347, %358, %361 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %363 = x86.dm.vbroadcastss [%355 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %364 = x86.rss.vfmadd231ps %349, %358, %363 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %365 = x86.dm.vbroadcastss [%355 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %366 = x86.rss.vfmadd231ps %351, %358, %365 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %367 = x86.dm.vbroadcastss [%355 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %368 = x86.rss.vfmadd231ps %353, %358, %367 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %369 = x86.dm.vbroadcastss [%355 + 2560] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %370 = x86.ri.add %355, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %371 = x86.ri.add %356, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %372 = x86.rss.vfmadd231ps %357, %358, %369 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %371, %370, %303, %304, %305, %306, %360, %362, %364, %366, %368, %372 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %373 = x86.ri.sub %289, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovups[%290], %294, %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%290 + 280], %295, %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%290 + 560], %296, %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%290 + 840], %297, %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%290 + 1120], %298, %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%290 + 1400], %299, %293 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %374 = x86.ri.add %290, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %375 = x86.ri.sub %288, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %375, %373, %374, %291, %292, %293 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %376 = x86.ri.add %270, 1400 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %377 = x86.ri.add %269, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %378 = x86.ri.sub %268, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %378, %377, %376, %271, %272 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %379 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %380, %381, %382, %383, %384 = x86_scf.for %385 : !x86.reg64<r11>  = %379 to 38 : si32 step 5 : si32 iter_args(%386 = %11, %387 = %12, %388 = %13, %389 = %14, %390 = %15) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %391 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %392, %393, %394, %395, %396 = x86_scf.for %397 : !x86.reg64<r10>  = %391 to 64 : si32 step 64 : si32 iter_args(%398 = %386, %399 = %387, %400 = %388, %401 = %389, %402 = %390) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %403 = x86.dm.vmovups [%400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %404 = x86.dm.vmovups [%400 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %405 = x86.dm.vmovups [%400 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %406 = x86.dm.vmovups [%400 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %407 = x86.dm.vmovups [%400 + 280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %408 = x86.dm.vmovups [%400 + 344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %409 = x86.dm.vmovups [%400 + 408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %410 = x86.dm.vmovups [%400 + 472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %411 = x86.dm.vmovups [%400 + 560] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %412 = x86.dm.vmovups [%400 + 624] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %413 = x86.dm.vmovups [%400 + 688] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %414 = x86.dm.vmovups [%400 + 752] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %415 = x86.dm.vmovups [%400 + 840] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %416 = x86.dm.vmovups [%400 + 904] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %417 = x86.dm.vmovups [%400 + 968] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %418 = x86.dm.vmovups [%400 + 1032] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %419 = x86.dm.vmovups [%400 + 1120] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %420 = x86.dm.vmovups [%400 + 1184] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %421 = x86.dm.vmovups [%400 + 1248] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %422 = x86.dm.vmovups [%400 + 1312] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %423 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %424, %425, %426, %427, %428, %429, %430, %431, %432, %433, %434, %435, %436, %437, %438, %439, %440, %441, %442, %443, %444, %445, %446, %447, %448 = x86_scf.for %449 : !x86.reg64<r12>  = %423 to 128 : si32 step 4 : si32 iter_args(%450 = %398, %451 = %399, %452 = %400, %453 = %401, %454 = %402, %455 = %403, %456 = %404, %457 = %405, %458 = %406, %459 = %407, %460 = %408, %461 = %409, %462 = %410, %463 = %411, %464 = %412, %465 = %413, %466 = %414, %467 = %415, %468 = %416, %469 = %417, %470 = %418, %471 = %419, %472 = %420, %473 = %421, %474 = %422) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %475 = x86.dm.vmovups [%450] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %476 = x86.dm.vmovups [%450 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %477 = x86.dm.vmovups [%450 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %478 = x86.dm.vmovups [%450 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %479 = x86.dm.vbroadcastss [%451] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %480 = x86.rss.vfmadd231ps %455, %475, %479 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %481 = x86.rss.vfmadd231ps %456, %476, %479 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %482 = x86.rss.vfmadd231ps %457, %477, %479 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %483 = x86.rss.vfmadd231ps %458, %478, %479 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %484 = x86.dm.vbroadcastss [%451 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %485 = x86.rss.vfmadd231ps %459, %475, %484 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %486 = x86.rss.vfmadd231ps %460, %476, %484 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %487 = x86.rss.vfmadd231ps %461, %477, %484 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %488 = x86.rss.vfmadd231ps %462, %478, %484 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %489 = x86.dm.vbroadcastss [%451 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %490 = x86.rss.vfmadd231ps %463, %475, %489 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %491 = x86.rss.vfmadd231ps %464, %476, %489 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %492 = x86.rss.vfmadd231ps %465, %477, %489 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %493 = x86.rss.vfmadd231ps %466, %478, %489 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %494 = x86.dm.vbroadcastss [%451 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %495 = x86.rss.vfmadd231ps %467, %475, %494 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %496 = x86.rss.vfmadd231ps %468, %476, %494 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %497 = x86.rss.vfmadd231ps %469, %477, %494 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %498 = x86.rss.vfmadd231ps %470, %478, %494 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %499 = x86.dm.vbroadcastss [%451 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %500 = x86.ri.add %451, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %501 = x86.ri.add %450, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %502 = x86.rss.vfmadd231ps %471, %475, %499 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %503 = x86.rss.vfmadd231ps %472, %476, %499 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %504 = x86.rss.vfmadd231ps %473, %477, %499 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %505 = x86.rss.vfmadd231ps %474, %478, %499 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %506 = x86.dm.vmovups [%501] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %507 = x86.dm.vmovups [%501 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %508 = x86.dm.vmovups [%501 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %509 = x86.dm.vmovups [%501 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %510 = x86.dm.vbroadcastss [%500] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %511 = x86.rss.vfmadd231ps %480, %506, %510 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %512 = x86.rss.vfmadd231ps %481, %507, %510 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %513 = x86.rss.vfmadd231ps %482, %508, %510 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %514 = x86.rss.vfmadd231ps %483, %509, %510 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %515 = x86.dm.vbroadcastss [%500 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %516 = x86.rss.vfmadd231ps %485, %506, %515 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %517 = x86.rss.vfmadd231ps %486, %507, %515 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %518 = x86.rss.vfmadd231ps %487, %508, %515 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %519 = x86.rss.vfmadd231ps %488, %509, %515 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %520 = x86.dm.vbroadcastss [%500 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %521 = x86.rss.vfmadd231ps %490, %506, %520 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %522 = x86.rss.vfmadd231ps %491, %507, %520 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %523 = x86.rss.vfmadd231ps %492, %508, %520 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %524 = x86.rss.vfmadd231ps %493, %509, %520 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %525 = x86.dm.vbroadcastss [%500 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %526 = x86.rss.vfmadd231ps %495, %506, %525 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %527 = x86.rss.vfmadd231ps %496, %507, %525 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %528 = x86.rss.vfmadd231ps %497, %508, %525 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %529 = x86.rss.vfmadd231ps %498, %509, %525 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %530 = x86.dm.vbroadcastss [%500 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %531 = x86.ri.add %500, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %532 = x86.ri.add %501, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %533 = x86.rss.vfmadd231ps %502, %506, %530 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %534 = x86.rss.vfmadd231ps %503, %507, %530 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %535 = x86.rss.vfmadd231ps %504, %508, %530 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %536 = x86.rss.vfmadd231ps %505, %509, %530 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %537 = x86.dm.vmovups [%532] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %538 = x86.dm.vmovups [%532 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %539 = x86.dm.vmovups [%532 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %540 = x86.dm.vmovups [%532 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %541 = x86.dm.vbroadcastss [%531] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %542 = x86.rss.vfmadd231ps %511, %537, %541 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %543 = x86.rss.vfmadd231ps %512, %538, %541 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %544 = x86.rss.vfmadd231ps %513, %539, %541 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %545 = x86.rss.vfmadd231ps %514, %540, %541 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %546 = x86.dm.vbroadcastss [%531 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %547 = x86.rss.vfmadd231ps %516, %537, %546 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %548 = x86.rss.vfmadd231ps %517, %538, %546 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %549 = x86.rss.vfmadd231ps %518, %539, %546 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %550 = x86.rss.vfmadd231ps %519, %540, %546 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %551 = x86.dm.vbroadcastss [%531 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %552 = x86.rss.vfmadd231ps %521, %537, %551 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %553 = x86.rss.vfmadd231ps %522, %538, %551 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %554 = x86.rss.vfmadd231ps %523, %539, %551 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %555 = x86.rss.vfmadd231ps %524, %540, %551 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %556 = x86.dm.vbroadcastss [%531 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %557 = x86.rss.vfmadd231ps %526, %537, %556 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %558 = x86.rss.vfmadd231ps %527, %538, %556 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %559 = x86.rss.vfmadd231ps %528, %539, %556 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %560 = x86.rss.vfmadd231ps %529, %540, %556 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %561 = x86.dm.vbroadcastss [%531 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %562 = x86.ri.add %531, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %563 = x86.ri.add %532, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %564 = x86.rss.vfmadd231ps %533, %537, %561 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %565 = x86.rss.vfmadd231ps %534, %538, %561 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %566 = x86.rss.vfmadd231ps %535, %539, %561 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %567 = x86.rss.vfmadd231ps %536, %540, %561 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %568 = x86.dm.vmovups [%563] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %569 = x86.dm.vmovups [%563 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %570 = x86.dm.vmovups [%563 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:          %571 = x86.dm.vmovups [%563 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %572 = x86.dm.vbroadcastss [%562] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %573 = x86.rss.vfmadd231ps %542, %568, %572 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %574 = x86.rss.vfmadd231ps %543, %569, %572 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %575 = x86.rss.vfmadd231ps %544, %570, %572 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %576 = x86.rss.vfmadd231ps %545, %571, %572 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %577 = x86.dm.vbroadcastss [%562 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %578 = x86.rss.vfmadd231ps %547, %568, %577 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %579 = x86.rss.vfmadd231ps %548, %569, %577 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %580 = x86.rss.vfmadd231ps %549, %570, %577 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %581 = x86.rss.vfmadd231ps %550, %571, %577 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %582 = x86.dm.vbroadcastss [%562 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %583 = x86.rss.vfmadd231ps %552, %568, %582 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %584 = x86.rss.vfmadd231ps %553, %569, %582 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %585 = x86.rss.vfmadd231ps %554, %570, %582 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %586 = x86.rss.vfmadd231ps %555, %571, %582 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %587 = x86.dm.vbroadcastss [%562 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %588 = x86.rss.vfmadd231ps %557, %568, %587 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %589 = x86.rss.vfmadd231ps %558, %569, %587 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %590 = x86.rss.vfmadd231ps %559, %570, %587 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %591 = x86.rss.vfmadd231ps %560, %571, %587 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %592 = x86.dm.vbroadcastss [%562 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %593 = x86.ri.add %562, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %594 = x86.ri.add %563, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %595 = x86.rss.vfmadd231ps %564, %568, %592 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %596 = x86.rss.vfmadd231ps %565, %569, %592 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %597 = x86.rss.vfmadd231ps %566, %570, %592 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %598 = x86.rss.vfmadd231ps %567, %571, %592 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %594, %593, %452, %453, %454, %573, %574, %575, %576, %578, %579, %580, %581, %583, %584, %585, %586, %588, %589, %590, %591, %595, %596, %597, %598 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %599 = x86.ri.sub %425, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovups [%426], %429 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 64], %430 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 128], %431 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 192], %432 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 280], %433 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 344], %434 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 408], %435 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 472], %436 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 560], %437 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 624], %438 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 688], %439 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 752], %440 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 840], %441 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 904], %442 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 968], %443 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 1032], %444 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 1120], %445 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 1184], %446 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 1248], %447 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovups [%426 + 1312], %448 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %600 = x86.ri.add %426, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %601 = x86.ri.sub %424, 35584 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %601, %599, %600, %427, %428 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %602 = x86.di.mov 63 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %603 = x86.ks.kmovw %602 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %604 = x86.di.mov 64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %605, %606, %607, %608, %609, %610 = x86_scf.for %611 : !x86.reg64<r10>  = %604 to 70 : si32 step 6 : si32 iter_args(%612 = %392, %613 = %393, %614 = %394, %615 = %395, %616 = %396, %617 = %603) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %618 = x86.dmk.vmovups[%614], %617 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %619 = x86.dmk.vmovups[%614 + 280], %617 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %620 = x86.dmk.vmovups[%614 + 560], %617 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %621 = x86.dmk.vmovups[%614 + 840], %617 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %622 = x86.dmk.vmovups[%614 + 1120], %617 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %623 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %624, %625, %626, %627, %628, %629, %630, %631, %632, %633, %634 = x86_scf.for %635 : !x86.reg64<r12>  = %623 to 128 : si32 step 4 : si32 iter_args(%636 = %612, %637 = %613, %638 = %614, %639 = %615, %640 = %616, %641 = %617, %642 = %618, %643 = %619, %644 = %620, %645 = %621, %646 = %622) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %647 = x86.dmk.vmovups[%636], %641 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %648 = x86.dm.vbroadcastss [%637] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %649 = x86.rss.vfmadd231ps %642, %647, %648 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %650 = x86.dm.vbroadcastss [%637 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %651 = x86.rss.vfmadd231ps %643, %647, %650 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %652 = x86.dm.vbroadcastss [%637 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %653 = x86.rss.vfmadd231ps %644, %647, %652 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %654 = x86.dm.vbroadcastss [%637 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %655 = x86.rss.vfmadd231ps %645, %647, %654 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %656 = x86.dm.vbroadcastss [%637 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %657 = x86.ri.add %637, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %658 = x86.ri.add %636, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %659 = x86.rss.vfmadd231ps %646, %647, %656 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %660 = x86.dmk.vmovups[%658], %641 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %661 = x86.dm.vbroadcastss [%657] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %662 = x86.rss.vfmadd231ps %649, %660, %661 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %663 = x86.dm.vbroadcastss [%657 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %664 = x86.rss.vfmadd231ps %651, %660, %663 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %665 = x86.dm.vbroadcastss [%657 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %666 = x86.rss.vfmadd231ps %653, %660, %665 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %667 = x86.dm.vbroadcastss [%657 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %668 = x86.rss.vfmadd231ps %655, %660, %667 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %669 = x86.dm.vbroadcastss [%657 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %670 = x86.ri.add %657, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %671 = x86.ri.add %658, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %672 = x86.rss.vfmadd231ps %659, %660, %669 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %673 = x86.dmk.vmovups[%671], %641 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %674 = x86.dm.vbroadcastss [%670] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %675 = x86.rss.vfmadd231ps %662, %673, %674 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %676 = x86.dm.vbroadcastss [%670 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %677 = x86.rss.vfmadd231ps %664, %673, %676 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %678 = x86.dm.vbroadcastss [%670 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %679 = x86.rss.vfmadd231ps %666, %673, %678 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %680 = x86.dm.vbroadcastss [%670 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %681 = x86.rss.vfmadd231ps %668, %673, %680 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %682 = x86.dm.vbroadcastss [%670 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %683 = x86.ri.add %670, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %684 = x86.ri.add %671, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %685 = x86.rss.vfmadd231ps %672, %673, %682 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %686 = x86.dmk.vmovups[%684], %641 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %687 = x86.dm.vbroadcastss [%683] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %688 = x86.rss.vfmadd231ps %675, %686, %687 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %689 = x86.dm.vbroadcastss [%683 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %690 = x86.rss.vfmadd231ps %677, %686, %689 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %691 = x86.dm.vbroadcastss [%683 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %692 = x86.rss.vfmadd231ps %679, %686, %691 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %693 = x86.dm.vbroadcastss [%683 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %694 = x86.rss.vfmadd231ps %681, %686, %693 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %695 = x86.dm.vbroadcastss [%683 + 2048] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %696 = x86.ri.add %683, 4 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %697 = x86.ri.add %684, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %698 = x86.rss.vfmadd231ps %685, %686, %695 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %697, %696, %638, %639, %640, %641, %688, %690, %692, %694, %698 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %699 = x86.ri.sub %625, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovups[%626], %630, %629 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%626 + 280], %631, %629 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%626 + 560], %632, %629 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%626 + 840], %633, %629 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovups[%626 + 1120], %634, %629 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %700 = x86.ri.add %626, 24 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %701 = x86.ri.sub %624, 35816 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %701, %699, %700, %627, %628, %629 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %702 = x86.ri.add %607, 1120 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %703 = x86.ri.add %606, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %704 = x86.ri.sub %605, 280 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %704, %703, %702, %608, %609 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %705 = x86.ds.mov %383 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %706, %707 = x86.d.pop %705 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
