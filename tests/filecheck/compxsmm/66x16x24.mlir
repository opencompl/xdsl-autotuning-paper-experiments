// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15 = x86_scf.for %16 : !x86.reg64<r11>  = %10 to 14 : si32 step 14 : si32 iter_args(%17 = %0, %18 = %1, %19 = %2, %20 = %6, %21 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %22 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %23, %24, %25, %26, %27 = x86_scf.for %28 : !x86.reg64<r10>  = %22 to 16 : si32 step 16 : si32 iter_args(%29 = %17, %30 = %18, %31 = %19, %32 = %20, %33 = %21) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %34 = x86.dm.vmovapd [%31] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %35 = x86.dm.vmovapd [%31 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:        %36 = x86.dm.vmovapd [%31 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %37 = x86.dm.vmovapd [%31 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %38 = x86.dm.vmovapd [%31 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %39 = x86.dm.vmovapd [%31 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %40 = x86.dm.vmovapd [%31 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %41 = x86.dm.vmovapd [%31 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %42 = x86.dm.vmovapd [%31 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %43 = x86.dm.vmovapd [%31 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %44 = x86.dm.vmovapd [%31 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %45 = x86.dm.vmovapd [%31 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %46 = x86.dm.vmovapd [%31 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %47 = x86.dm.vmovapd [%31 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %48 = x86.dm.vmovapd [%31 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %49 = x86.dm.vmovapd [%31 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %50 = x86.dm.vmovapd [%31 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %51 = x86.dm.vmovapd [%31 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %52 = x86.dm.vmovapd [%31 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %53 = x86.dm.vmovapd [%31 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %54 = x86.dm.vmovapd [%31 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %55 = x86.dm.vmovapd [%31 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %56 = x86.dm.vmovapd [%31 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %57 = x86.dm.vmovapd [%31 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %58 = x86.dm.vmovapd [%31 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %59 = x86.dm.vmovapd [%31 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %60 = x86.dm.vmovapd [%31 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %61 = x86.dm.vmovapd [%31 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %62 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95 = x86_scf.for %96 : !x86.reg64<r12>  = %62 to 24 : si32 step 4 : si32 iter_args(%97 = %29, %98 = %30, %99 = %31, %100 = %32, %101 = %33, %102 = %34, %103 = %35, %104 = %36, %105 = %37, %106 = %38, %107 = %39, %108 = %40, %109 = %41, %110 = %42, %111 = %43, %112 = %44, %113 = %45, %114 = %46, %115 = %47, %116 = %48, %117 = %49, %118 = %50, %119 = %51, %120 = %52, %121 = %53, %122 = %54, %123 = %55, %124 = %56, %125 = %57, %126 = %58, %127 = %59, %128 = %60, %129 = %61) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %130 = x86.dm.vmovapd [%97] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %131 = x86.dm.vmovapd [%97 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %132 = x86.dm.vbroadcastsd [%98] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %133 = x86.rss.vfmadd231pd %102, %130, %132 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %134 = x86.rss.vfmadd231pd %103, %131, %132 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:          %135 = x86.dm.vbroadcastsd [%98 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %136 = x86.rss.vfmadd231pd %104, %130, %135 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %137 = x86.rss.vfmadd231pd %105, %131, %135 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %138 = x86.dm.vbroadcastsd [%98 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %139 = x86.rss.vfmadd231pd %106, %130, %138 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %140 = x86.rss.vfmadd231pd %107, %131, %138 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %141 = x86.dm.vbroadcastsd [%98 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %142 = x86.rss.vfmadd231pd %108, %130, %141 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %143 = x86.rss.vfmadd231pd %109, %131, %141 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %144 = x86.dm.vbroadcastsd [%98 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %145 = x86.rss.vfmadd231pd %110, %130, %144 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %146 = x86.rss.vfmadd231pd %111, %131, %144 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %147 = x86.dm.vbroadcastsd [%98 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %148 = x86.rss.vfmadd231pd %112, %130, %147 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %149 = x86.rss.vfmadd231pd %113, %131, %147 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %150 = x86.dm.vbroadcastsd [%98 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %151 = x86.rss.vfmadd231pd %114, %130, %150 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %152 = x86.rss.vfmadd231pd %115, %131, %150 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %153 = x86.dm.vbroadcastsd [%98 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %154 = x86.rss.vfmadd231pd %116, %130, %153 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %155 = x86.rss.vfmadd231pd %117, %131, %153 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %156 = x86.dm.vbroadcastsd [%98 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %157 = x86.rss.vfmadd231pd %118, %130, %156 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %158 = x86.rss.vfmadd231pd %119, %131, %156 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %159 = x86.dm.vbroadcastsd [%98 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %160 = x86.rss.vfmadd231pd %120, %130, %159 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %161 = x86.rss.vfmadd231pd %121, %131, %159 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %162 = x86.dm.vbroadcastsd [%98 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %163 = x86.rss.vfmadd231pd %122, %130, %162 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %164 = x86.rss.vfmadd231pd %123, %131, %162 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %165 = x86.dm.vbroadcastsd [%98 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %166 = x86.rss.vfmadd231pd %124, %130, %165 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %167 = x86.rss.vfmadd231pd %125, %131, %165 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %168 = x86.dm.vbroadcastsd [%98 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %169 = x86.rss.vfmadd231pd %126, %130, %168 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %170 = x86.rss.vfmadd231pd %127, %131, %168 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %171 = x86.dm.vbroadcastsd [%98 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %172 = x86.ri.add %98, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %173 = x86.ri.add %97, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %174 = x86.rss.vfmadd231pd %128, %130, %171 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %175 = x86.rss.vfmadd231pd %129, %131, %171 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %176 = x86.dm.vmovapd [%173] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %177 = x86.dm.vmovapd [%173 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %178 = x86.dm.vbroadcastsd [%172] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %179 = x86.rss.vfmadd231pd %133, %176, %178 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %180 = x86.rss.vfmadd231pd %134, %177, %178 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:          %181 = x86.dm.vbroadcastsd [%172 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %182 = x86.rss.vfmadd231pd %136, %176, %181 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %183 = x86.rss.vfmadd231pd %137, %177, %181 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %184 = x86.dm.vbroadcastsd [%172 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %185 = x86.rss.vfmadd231pd %139, %176, %184 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %186 = x86.rss.vfmadd231pd %140, %177, %184 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %187 = x86.dm.vbroadcastsd [%172 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %188 = x86.rss.vfmadd231pd %142, %176, %187 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %189 = x86.rss.vfmadd231pd %143, %177, %187 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %190 = x86.dm.vbroadcastsd [%172 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %191 = x86.rss.vfmadd231pd %145, %176, %190 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %192 = x86.rss.vfmadd231pd %146, %177, %190 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %193 = x86.dm.vbroadcastsd [%172 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %194 = x86.rss.vfmadd231pd %148, %176, %193 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %195 = x86.rss.vfmadd231pd %149, %177, %193 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %196 = x86.dm.vbroadcastsd [%172 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %197 = x86.rss.vfmadd231pd %151, %176, %196 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %198 = x86.rss.vfmadd231pd %152, %177, %196 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %199 = x86.dm.vbroadcastsd [%172 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %200 = x86.rss.vfmadd231pd %154, %176, %199 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %201 = x86.rss.vfmadd231pd %155, %177, %199 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %202 = x86.dm.vbroadcastsd [%172 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %203 = x86.rss.vfmadd231pd %157, %176, %202 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %204 = x86.rss.vfmadd231pd %158, %177, %202 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %205 = x86.dm.vbroadcastsd [%172 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %206 = x86.rss.vfmadd231pd %160, %176, %205 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %207 = x86.rss.vfmadd231pd %161, %177, %205 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %208 = x86.dm.vbroadcastsd [%172 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %209 = x86.rss.vfmadd231pd %163, %176, %208 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %210 = x86.rss.vfmadd231pd %164, %177, %208 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %211 = x86.dm.vbroadcastsd [%172 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %212 = x86.rss.vfmadd231pd %166, %176, %211 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %213 = x86.rss.vfmadd231pd %167, %177, %211 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %214 = x86.dm.vbroadcastsd [%172 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %215 = x86.rss.vfmadd231pd %169, %176, %214 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %216 = x86.rss.vfmadd231pd %170, %177, %214 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %217 = x86.dm.vbroadcastsd [%172 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %218 = x86.ri.add %172, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %219 = x86.ri.add %173, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %220 = x86.rss.vfmadd231pd %174, %176, %217 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %221 = x86.rss.vfmadd231pd %175, %177, %217 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %222 = x86.dm.vmovapd [%219] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %223 = x86.dm.vmovapd [%219 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %224 = x86.dm.vbroadcastsd [%218] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %225 = x86.rss.vfmadd231pd %179, %222, %224 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %226 = x86.rss.vfmadd231pd %180, %223, %224 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:          %227 = x86.dm.vbroadcastsd [%218 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %228 = x86.rss.vfmadd231pd %182, %222, %227 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %229 = x86.rss.vfmadd231pd %183, %223, %227 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %230 = x86.dm.vbroadcastsd [%218 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %231 = x86.rss.vfmadd231pd %185, %222, %230 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %232 = x86.rss.vfmadd231pd %186, %223, %230 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %233 = x86.dm.vbroadcastsd [%218 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %234 = x86.rss.vfmadd231pd %188, %222, %233 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %235 = x86.rss.vfmadd231pd %189, %223, %233 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %236 = x86.dm.vbroadcastsd [%218 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %237 = x86.rss.vfmadd231pd %191, %222, %236 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %238 = x86.rss.vfmadd231pd %192, %223, %236 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %239 = x86.dm.vbroadcastsd [%218 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %240 = x86.rss.vfmadd231pd %194, %222, %239 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %241 = x86.rss.vfmadd231pd %195, %223, %239 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %242 = x86.dm.vbroadcastsd [%218 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %243 = x86.rss.vfmadd231pd %197, %222, %242 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %244 = x86.rss.vfmadd231pd %198, %223, %242 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %245 = x86.dm.vbroadcastsd [%218 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %246 = x86.rss.vfmadd231pd %200, %222, %245 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %247 = x86.rss.vfmadd231pd %201, %223, %245 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %248 = x86.dm.vbroadcastsd [%218 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %249 = x86.rss.vfmadd231pd %203, %222, %248 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %250 = x86.rss.vfmadd231pd %204, %223, %248 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %251 = x86.dm.vbroadcastsd [%218 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %252 = x86.rss.vfmadd231pd %206, %222, %251 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %253 = x86.rss.vfmadd231pd %207, %223, %251 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %254 = x86.dm.vbroadcastsd [%218 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %255 = x86.rss.vfmadd231pd %209, %222, %254 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %256 = x86.rss.vfmadd231pd %210, %223, %254 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %257 = x86.dm.vbroadcastsd [%218 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %258 = x86.rss.vfmadd231pd %212, %222, %257 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %259 = x86.rss.vfmadd231pd %213, %223, %257 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %260 = x86.dm.vbroadcastsd [%218 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %261 = x86.rss.vfmadd231pd %215, %222, %260 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %262 = x86.rss.vfmadd231pd %216, %223, %260 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %263 = x86.dm.vbroadcastsd [%218 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %264 = x86.ri.add %218, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %265 = x86.ri.add %219, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %266 = x86.rss.vfmadd231pd %220, %222, %263 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %267 = x86.rss.vfmadd231pd %221, %223, %263 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %268 = x86.dm.vmovapd [%265] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %269 = x86.dm.vmovapd [%265 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %270 = x86.dm.vbroadcastsd [%264] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %271 = x86.rss.vfmadd231pd %225, %268, %270 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:          %272 = x86.rss.vfmadd231pd %226, %269, %270 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:          %273 = x86.dm.vbroadcastsd [%264 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %274 = x86.rss.vfmadd231pd %228, %268, %273 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %275 = x86.rss.vfmadd231pd %229, %269, %273 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %276 = x86.dm.vbroadcastsd [%264 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %277 = x86.rss.vfmadd231pd %231, %268, %276 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %278 = x86.rss.vfmadd231pd %232, %269, %276 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %279 = x86.dm.vbroadcastsd [%264 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %280 = x86.rss.vfmadd231pd %234, %268, %279 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %281 = x86.rss.vfmadd231pd %235, %269, %279 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %282 = x86.dm.vbroadcastsd [%264 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %283 = x86.rss.vfmadd231pd %237, %268, %282 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %284 = x86.rss.vfmadd231pd %238, %269, %282 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %285 = x86.dm.vbroadcastsd [%264 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %286 = x86.rss.vfmadd231pd %240, %268, %285 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %287 = x86.rss.vfmadd231pd %241, %269, %285 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %288 = x86.dm.vbroadcastsd [%264 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %289 = x86.rss.vfmadd231pd %243, %268, %288 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %290 = x86.rss.vfmadd231pd %244, %269, %288 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %291 = x86.dm.vbroadcastsd [%264 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %292 = x86.rss.vfmadd231pd %246, %268, %291 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %293 = x86.rss.vfmadd231pd %247, %269, %291 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %294 = x86.dm.vbroadcastsd [%264 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %295 = x86.rss.vfmadd231pd %249, %268, %294 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %296 = x86.rss.vfmadd231pd %250, %269, %294 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %297 = x86.dm.vbroadcastsd [%264 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %298 = x86.rss.vfmadd231pd %252, %268, %297 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %299 = x86.rss.vfmadd231pd %253, %269, %297 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %300 = x86.dm.vbroadcastsd [%264 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %301 = x86.rss.vfmadd231pd %255, %268, %300 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %302 = x86.rss.vfmadd231pd %256, %269, %300 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %303 = x86.dm.vbroadcastsd [%264 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %304 = x86.rss.vfmadd231pd %258, %268, %303 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %305 = x86.rss.vfmadd231pd %259, %269, %303 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %306 = x86.dm.vbroadcastsd [%264 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %307 = x86.rss.vfmadd231pd %261, %268, %306 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %308 = x86.rss.vfmadd231pd %262, %269, %306 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %309 = x86.dm.vbroadcastsd [%264 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %310 = x86.ri.add %264, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %311 = x86.ri.add %265, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %312 = x86.rss.vfmadd231pd %266, %268, %309 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %313 = x86.rss.vfmadd231pd %267, %269, %309 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %311, %310, %99, %100, %101, %271, %272, %274, %275, %277, %278, %280, %281, %283, %284, %286, %287, %289, %290, %292, %293, %295, %296, %298, %299, %301, %302, %304, %305, %307, %308, %312, %313 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %314 = x86.ri.sub %64, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%65], %68 : (!x86.reg64<rdx>, !x86.avx512reg<zmm4>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 64], %69 : (!x86.reg64<rdx>, !x86.avx512reg<zmm5>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 128], %70 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 192], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 256], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 320], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 384], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 448], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 512], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 576], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 640], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 704], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 768], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 832], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 896], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 960], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1024], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1088], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1152], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1216], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1280], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1344], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1408], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1472], %91 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1536], %92 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1600], %93 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1664], %94 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%65 + 1728], %95 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %315 = x86.ri.add %65, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %316 = x86.ri.sub %63, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %316, %314, %315, %66, %67 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %317 = x86.ri.add %25, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %318 = x86.ri.add %24, 2688 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %319 = x86.ri.sub %23, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %319, %318, %317, %26, %27 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %320 = x86.di.mov 14 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %321, %322, %323, %324, %325 = x86_scf.for %326 : !x86.reg64<r11>  = %320 to 66 : si32 step 13 : si32 iter_args(%327 = %11, %328 = %12, %329 = %13, %330 = %14, %331 = %15) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %332 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %333, %334, %335, %336, %337 = x86_scf.for %338 : !x86.reg64<r10>  = %332 to 16 : si32 step 16 : si32 iter_args(%339 = %327, %340 = %328, %341 = %329, %342 = %330, %343 = %331) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %344 = x86.dm.vmovapd [%341] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %345 = x86.dm.vmovapd [%341 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %346 = x86.dm.vmovapd [%341 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %347 = x86.dm.vmovapd [%341 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %348 = x86.dm.vmovapd [%341 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %349 = x86.dm.vmovapd [%341 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %350 = x86.dm.vmovapd [%341 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %351 = x86.dm.vmovapd [%341 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %352 = x86.dm.vmovapd [%341 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %353 = x86.dm.vmovapd [%341 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %354 = x86.dm.vmovapd [%341 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %355 = x86.dm.vmovapd [%341 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %356 = x86.dm.vmovapd [%341 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %357 = x86.dm.vmovapd [%341 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %358 = x86.dm.vmovapd [%341 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %359 = x86.dm.vmovapd [%341 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %360 = x86.dm.vmovapd [%341 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %361 = x86.dm.vmovapd [%341 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %362 = x86.dm.vmovapd [%341 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %363 = x86.dm.vmovapd [%341 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %364 = x86.dm.vmovapd [%341 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %365 = x86.dm.vmovapd [%341 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %366 = x86.dm.vmovapd [%341 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %367 = x86.dm.vmovapd [%341 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %368 = x86.dm.vmovapd [%341 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %369 = x86.dm.vmovapd [%341 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %370 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:        %371, %372, %373, %374, %375, %376, %377, %378, %379, %380, %381, %382, %383, %384, %385, %386, %387, %388, %389, %390, %391, %392, %393, %394, %395, %396, %397, %398, %399, %400, %401 = x86_scf.for %402 : !x86.reg64<r12>  = %370 to 24 : si32 step 4 : si32 iter_args(%403 = %339, %404 = %340, %405 = %341, %406 = %342, %407 = %343, %408 = %344, %409 = %345, %410 = %346, %411 = %347, %412 = %348, %413 = %349, %414 = %350, %415 = %351, %416 = %352, %417 = %353, %418 = %354, %419 = %355, %420 = %356, %421 = %357, %422 = %358, %423 = %359, %424 = %360, %425 = %361, %426 = %362, %427 = %363, %428 = %364, %429 = %365, %430 = %366, %431 = %367, %432 = %368, %433 = %369) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:          %434 = x86.dm.vmovapd [%403] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %435 = x86.dm.vmovapd [%403 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %436 = x86.dm.vbroadcastsd [%404] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %437 = x86.rss.vfmadd231pd %408, %434, %436 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %438 = x86.rss.vfmadd231pd %409, %435, %436 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %439 = x86.dm.vbroadcastsd [%404 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %440 = x86.rss.vfmadd231pd %410, %434, %439 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %441 = x86.rss.vfmadd231pd %411, %435, %439 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %442 = x86.dm.vbroadcastsd [%404 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %443 = x86.rss.vfmadd231pd %412, %434, %442 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %444 = x86.rss.vfmadd231pd %413, %435, %442 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %445 = x86.dm.vbroadcastsd [%404 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %446 = x86.rss.vfmadd231pd %414, %434, %445 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %447 = x86.rss.vfmadd231pd %415, %435, %445 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %448 = x86.dm.vbroadcastsd [%404 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %449 = x86.rss.vfmadd231pd %416, %434, %448 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %450 = x86.rss.vfmadd231pd %417, %435, %448 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %451 = x86.dm.vbroadcastsd [%404 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %452 = x86.rss.vfmadd231pd %418, %434, %451 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %453 = x86.rss.vfmadd231pd %419, %435, %451 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %454 = x86.dm.vbroadcastsd [%404 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %455 = x86.rss.vfmadd231pd %420, %434, %454 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %456 = x86.rss.vfmadd231pd %421, %435, %454 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %457 = x86.dm.vbroadcastsd [%404 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %458 = x86.rss.vfmadd231pd %422, %434, %457 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %459 = x86.rss.vfmadd231pd %423, %435, %457 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %460 = x86.dm.vbroadcastsd [%404 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %461 = x86.rss.vfmadd231pd %424, %434, %460 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %462 = x86.rss.vfmadd231pd %425, %435, %460 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %463 = x86.dm.vbroadcastsd [%404 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %464 = x86.rss.vfmadd231pd %426, %434, %463 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %465 = x86.rss.vfmadd231pd %427, %435, %463 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %466 = x86.dm.vbroadcastsd [%404 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %467 = x86.rss.vfmadd231pd %428, %434, %466 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %468 = x86.rss.vfmadd231pd %429, %435, %466 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %469 = x86.dm.vbroadcastsd [%404 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %470 = x86.rss.vfmadd231pd %430, %434, %469 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %471 = x86.rss.vfmadd231pd %431, %435, %469 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %472 = x86.dm.vbroadcastsd [%404 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %473 = x86.ri.add %404, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %474 = x86.ri.add %403, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %475 = x86.rss.vfmadd231pd %432, %434, %472 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %476 = x86.rss.vfmadd231pd %433, %435, %472 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %477 = x86.dm.vmovapd [%474] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %478 = x86.dm.vmovapd [%474 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %479 = x86.dm.vbroadcastsd [%473] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %480 = x86.rss.vfmadd231pd %437, %477, %479 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %481 = x86.rss.vfmadd231pd %438, %478, %479 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %482 = x86.dm.vbroadcastsd [%473 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %483 = x86.rss.vfmadd231pd %440, %477, %482 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %484 = x86.rss.vfmadd231pd %441, %478, %482 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %485 = x86.dm.vbroadcastsd [%473 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %486 = x86.rss.vfmadd231pd %443, %477, %485 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %487 = x86.rss.vfmadd231pd %444, %478, %485 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %488 = x86.dm.vbroadcastsd [%473 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %489 = x86.rss.vfmadd231pd %446, %477, %488 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %490 = x86.rss.vfmadd231pd %447, %478, %488 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %491 = x86.dm.vbroadcastsd [%473 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %492 = x86.rss.vfmadd231pd %449, %477, %491 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %493 = x86.rss.vfmadd231pd %450, %478, %491 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %494 = x86.dm.vbroadcastsd [%473 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %495 = x86.rss.vfmadd231pd %452, %477, %494 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %496 = x86.rss.vfmadd231pd %453, %478, %494 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %497 = x86.dm.vbroadcastsd [%473 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %498 = x86.rss.vfmadd231pd %455, %477, %497 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %499 = x86.rss.vfmadd231pd %456, %478, %497 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %500 = x86.dm.vbroadcastsd [%473 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %501 = x86.rss.vfmadd231pd %458, %477, %500 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %502 = x86.rss.vfmadd231pd %459, %478, %500 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %503 = x86.dm.vbroadcastsd [%473 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %504 = x86.rss.vfmadd231pd %461, %477, %503 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %505 = x86.rss.vfmadd231pd %462, %478, %503 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %506 = x86.dm.vbroadcastsd [%473 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %507 = x86.rss.vfmadd231pd %464, %477, %506 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %508 = x86.rss.vfmadd231pd %465, %478, %506 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %509 = x86.dm.vbroadcastsd [%473 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %510 = x86.rss.vfmadd231pd %467, %477, %509 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %511 = x86.rss.vfmadd231pd %468, %478, %509 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %512 = x86.dm.vbroadcastsd [%473 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %513 = x86.rss.vfmadd231pd %470, %477, %512 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %514 = x86.rss.vfmadd231pd %471, %478, %512 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %515 = x86.dm.vbroadcastsd [%473 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %516 = x86.ri.add %473, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %517 = x86.ri.add %474, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %518 = x86.rss.vfmadd231pd %475, %477, %515 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %519 = x86.rss.vfmadd231pd %476, %478, %515 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %520 = x86.dm.vmovapd [%517] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %521 = x86.dm.vmovapd [%517 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %522 = x86.dm.vbroadcastsd [%516] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %523 = x86.rss.vfmadd231pd %480, %520, %522 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %524 = x86.rss.vfmadd231pd %481, %521, %522 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %525 = x86.dm.vbroadcastsd [%516 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %526 = x86.rss.vfmadd231pd %483, %520, %525 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %527 = x86.rss.vfmadd231pd %484, %521, %525 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %528 = x86.dm.vbroadcastsd [%516 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %529 = x86.rss.vfmadd231pd %486, %520, %528 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %530 = x86.rss.vfmadd231pd %487, %521, %528 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %531 = x86.dm.vbroadcastsd [%516 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %532 = x86.rss.vfmadd231pd %489, %520, %531 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %533 = x86.rss.vfmadd231pd %490, %521, %531 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %534 = x86.dm.vbroadcastsd [%516 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %535 = x86.rss.vfmadd231pd %492, %520, %534 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %536 = x86.rss.vfmadd231pd %493, %521, %534 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %537 = x86.dm.vbroadcastsd [%516 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %538 = x86.rss.vfmadd231pd %495, %520, %537 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %539 = x86.rss.vfmadd231pd %496, %521, %537 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %540 = x86.dm.vbroadcastsd [%516 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %541 = x86.rss.vfmadd231pd %498, %520, %540 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %542 = x86.rss.vfmadd231pd %499, %521, %540 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %543 = x86.dm.vbroadcastsd [%516 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %544 = x86.rss.vfmadd231pd %501, %520, %543 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %545 = x86.rss.vfmadd231pd %502, %521, %543 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %546 = x86.dm.vbroadcastsd [%516 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %547 = x86.rss.vfmadd231pd %504, %520, %546 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %548 = x86.rss.vfmadd231pd %505, %521, %546 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %549 = x86.dm.vbroadcastsd [%516 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %550 = x86.rss.vfmadd231pd %507, %520, %549 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %551 = x86.rss.vfmadd231pd %508, %521, %549 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %552 = x86.dm.vbroadcastsd [%516 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %553 = x86.rss.vfmadd231pd %510, %520, %552 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %554 = x86.rss.vfmadd231pd %511, %521, %552 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %555 = x86.dm.vbroadcastsd [%516 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %556 = x86.rss.vfmadd231pd %513, %520, %555 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %557 = x86.rss.vfmadd231pd %514, %521, %555 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %558 = x86.dm.vbroadcastsd [%516 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %559 = x86.ri.add %516, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %560 = x86.ri.add %517, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %561 = x86.rss.vfmadd231pd %518, %520, %558 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %562 = x86.rss.vfmadd231pd %519, %521, %558 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          %563 = x86.dm.vmovapd [%560] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:          %564 = x86.dm.vmovapd [%560 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:          %565 = x86.dm.vbroadcastsd [%559] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %566 = x86.rss.vfmadd231pd %523, %563, %565 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:          %567 = x86.rss.vfmadd231pd %524, %564, %565 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:          %568 = x86.dm.vbroadcastsd [%559 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %569 = x86.rss.vfmadd231pd %526, %563, %568 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:          %570 = x86.rss.vfmadd231pd %527, %564, %568 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:          %571 = x86.dm.vbroadcastsd [%559 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %572 = x86.rss.vfmadd231pd %529, %563, %571 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:          %573 = x86.rss.vfmadd231pd %530, %564, %571 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:          %574 = x86.dm.vbroadcastsd [%559 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %575 = x86.rss.vfmadd231pd %532, %563, %574 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:          %576 = x86.rss.vfmadd231pd %533, %564, %574 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:          %577 = x86.dm.vbroadcastsd [%559 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %578 = x86.rss.vfmadd231pd %535, %563, %577 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:          %579 = x86.rss.vfmadd231pd %536, %564, %577 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:          %580 = x86.dm.vbroadcastsd [%559 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %581 = x86.rss.vfmadd231pd %538, %563, %580 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:          %582 = x86.rss.vfmadd231pd %539, %564, %580 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:          %583 = x86.dm.vbroadcastsd [%559 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %584 = x86.rss.vfmadd231pd %541, %563, %583 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:          %585 = x86.rss.vfmadd231pd %542, %564, %583 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:          %586 = x86.dm.vbroadcastsd [%559 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %587 = x86.rss.vfmadd231pd %544, %563, %586 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:          %588 = x86.rss.vfmadd231pd %545, %564, %586 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:          %589 = x86.dm.vbroadcastsd [%559 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %590 = x86.rss.vfmadd231pd %547, %563, %589 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:          %591 = x86.rss.vfmadd231pd %548, %564, %589 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:          %592 = x86.dm.vbroadcastsd [%559 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %593 = x86.rss.vfmadd231pd %550, %563, %592 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:          %594 = x86.rss.vfmadd231pd %551, %564, %592 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:          %595 = x86.dm.vbroadcastsd [%559 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %596 = x86.rss.vfmadd231pd %553, %563, %595 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:          %597 = x86.rss.vfmadd231pd %554, %564, %595 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:          %598 = x86.dm.vbroadcastsd [%559 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %599 = x86.rss.vfmadd231pd %556, %563, %598 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:          %600 = x86.rss.vfmadd231pd %557, %564, %598 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:          %601 = x86.dm.vbroadcastsd [%559 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:          %602 = x86.ri.add %559, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:          %603 = x86.ri.add %560, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:          %604 = x86.rss.vfmadd231pd %561, %563, %601 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:          %605 = x86.rss.vfmadd231pd %562, %564, %601 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:          x86_scf.yield %603, %602, %405, %406, %407, %566, %567, %569, %570, %572, %573, %575, %576, %578, %579, %581, %582, %584, %585, %587, %588, %590, %591, %593, %594, %596, %597, %599, %600, %604, %605 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:        }
// CHECK-NEXT:        %606 = x86.ri.sub %372, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%373], %376 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 64], %377 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 128], %378 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 192], %379 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 256], %380 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 320], %381 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 384], %382 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 448], %383 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 512], %384 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 576], %385 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 640], %386 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 704], %387 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 768], %388 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 832], %389 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 896], %390 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 960], %391 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1024], %392 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1088], %393 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1152], %394 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1216], %395 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1280], %396 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1344], %397 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1408], %398 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1472], %399 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1536], %400 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%373 + 1600], %401 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %607 = x86.ri.add %373, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %608 = x86.ri.sub %371, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %608, %606, %607, %374, %375 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %609 = x86.ri.add %335, 1536 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %610 = x86.ri.add %334, 2496 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %611 = x86.ri.sub %333, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %611, %610, %609, %336, %337 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %612 = x86.ds.mov %324 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %613, %614 = x86.d.pop %612 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
