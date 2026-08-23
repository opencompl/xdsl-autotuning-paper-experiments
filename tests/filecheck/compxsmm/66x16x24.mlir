// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:     x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:  %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:  %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:  %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:  %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:  %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:  %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 14 : si32 step 14 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:    %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 16 : si32 step 16 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %36 = x86.dm.vmovapd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %37 = x86.dm.vmovapd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:      %38 = x86.dm.vmovapd [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:      %39 = x86.dm.vmovapd [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:      %40 = x86.dm.vmovapd [%33 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %41 = x86.dm.vmovapd [%33 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %42 = x86.dm.vmovapd [%33 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %43 = x86.dm.vmovapd [%33 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %44 = x86.dm.vmovapd [%33 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %45 = x86.dm.vmovapd [%33 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %46 = x86.dm.vmovapd [%33 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %47 = x86.dm.vmovapd [%33 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %48 = x86.dm.vmovapd [%33 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %49 = x86.dm.vmovapd [%33 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %50 = x86.dm.vmovapd [%33 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %51 = x86.dm.vmovapd [%33 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %52 = x86.dm.vmovapd [%33 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %53 = x86.dm.vmovapd [%33 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %54 = x86.dm.vmovapd [%33 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %55 = x86.dm.vmovapd [%33 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %56 = x86.dm.vmovapd [%33 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %57 = x86.dm.vmovapd [%33 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %58 = x86.dm.vmovapd [%33 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %59 = x86.dm.vmovapd [%33 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %60 = x86.dm.vmovapd [%33 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %61 = x86.dm.vmovapd [%33 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %62 = x86.dm.vmovapd [%33 + 1664] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %63 = x86.dm.vmovapd [%33 + 1728] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %64 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98 = x86_scf.for %99 : !x86.reg64<r12>  = %64 to 24 : si32 step 4 : si32 iter_args(%100 = %31, %101 = %32, %102 = %33, %103 = %34, %104 = %35, %105 = %36, %106 = %37, %107 = %38, %108 = %39, %109 = %40, %110 = %41, %111 = %42, %112 = %43, %113 = %44, %114 = %45, %115 = %46, %116 = %47, %117 = %48, %118 = %49, %119 = %50, %120 = %51, %121 = %52, %122 = %53, %123 = %54, %124 = %55, %125 = %56, %126 = %57, %127 = %58, %128 = %59, %129 = %60, %130 = %61, %131 = %62, %132 = %63) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %133 = x86.dm.vmovapd [%100] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %134 = x86.dm.vmovapd [%100 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %135 = x86.dm.vbroadcastsd [%101] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %136 = x86.rss.vfmadd231pd %105, %133, %135 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %137 = x86.rss.vfmadd231pd %106, %134, %135 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:        %138 = x86.dm.vbroadcastsd [%101 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %139 = x86.rss.vfmadd231pd %107, %133, %138 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %140 = x86.rss.vfmadd231pd %108, %134, %138 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %141 = x86.dm.vbroadcastsd [%101 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %142 = x86.rss.vfmadd231pd %109, %133, %141 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %143 = x86.rss.vfmadd231pd %110, %134, %141 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %144 = x86.dm.vbroadcastsd [%101 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %145 = x86.rss.vfmadd231pd %111, %133, %144 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %146 = x86.rss.vfmadd231pd %112, %134, %144 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %147 = x86.dm.vbroadcastsd [%101 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %148 = x86.rss.vfmadd231pd %113, %133, %147 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %149 = x86.rss.vfmadd231pd %114, %134, %147 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %150 = x86.dm.vbroadcastsd [%101 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %151 = x86.rss.vfmadd231pd %115, %133, %150 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %152 = x86.rss.vfmadd231pd %116, %134, %150 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %153 = x86.dm.vbroadcastsd [%101 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %154 = x86.rss.vfmadd231pd %117, %133, %153 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %155 = x86.rss.vfmadd231pd %118, %134, %153 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %156 = x86.dm.vbroadcastsd [%101 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %157 = x86.rss.vfmadd231pd %119, %133, %156 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %158 = x86.rss.vfmadd231pd %120, %134, %156 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %159 = x86.dm.vbroadcastsd [%101 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %160 = x86.rss.vfmadd231pd %121, %133, %159 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %161 = x86.rss.vfmadd231pd %122, %134, %159 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %162 = x86.dm.vbroadcastsd [%101 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %163 = x86.rss.vfmadd231pd %123, %133, %162 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %164 = x86.rss.vfmadd231pd %124, %134, %162 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %165 = x86.dm.vbroadcastsd [%101 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %166 = x86.rss.vfmadd231pd %125, %133, %165 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %167 = x86.rss.vfmadd231pd %126, %134, %165 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %168 = x86.dm.vbroadcastsd [%101 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %169 = x86.rss.vfmadd231pd %127, %133, %168 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %170 = x86.rss.vfmadd231pd %128, %134, %168 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %171 = x86.dm.vbroadcastsd [%101 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %172 = x86.rss.vfmadd231pd %129, %133, %171 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %173 = x86.rss.vfmadd231pd %130, %134, %171 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %174 = x86.dm.vbroadcastsd [%101 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %175 = x86.ri.add %101, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %176 = x86.ri.add %100, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %177 = x86.rss.vfmadd231pd %131, %133, %174 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %178 = x86.rss.vfmadd231pd %132, %134, %174 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %179 = x86.dm.vmovapd [%176] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %180 = x86.dm.vmovapd [%176 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %181 = x86.dm.vbroadcastsd [%175] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %182 = x86.rss.vfmadd231pd %136, %179, %181 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %183 = x86.rss.vfmadd231pd %137, %180, %181 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:        %184 = x86.dm.vbroadcastsd [%175 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %185 = x86.rss.vfmadd231pd %139, %179, %184 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %186 = x86.rss.vfmadd231pd %140, %180, %184 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %187 = x86.dm.vbroadcastsd [%175 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %188 = x86.rss.vfmadd231pd %142, %179, %187 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %189 = x86.rss.vfmadd231pd %143, %180, %187 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %190 = x86.dm.vbroadcastsd [%175 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %191 = x86.rss.vfmadd231pd %145, %179, %190 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %192 = x86.rss.vfmadd231pd %146, %180, %190 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %193 = x86.dm.vbroadcastsd [%175 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %194 = x86.rss.vfmadd231pd %148, %179, %193 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %195 = x86.rss.vfmadd231pd %149, %180, %193 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %196 = x86.dm.vbroadcastsd [%175 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %197 = x86.rss.vfmadd231pd %151, %179, %196 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %198 = x86.rss.vfmadd231pd %152, %180, %196 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %199 = x86.dm.vbroadcastsd [%175 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %200 = x86.rss.vfmadd231pd %154, %179, %199 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %201 = x86.rss.vfmadd231pd %155, %180, %199 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %202 = x86.dm.vbroadcastsd [%175 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %203 = x86.rss.vfmadd231pd %157, %179, %202 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %204 = x86.rss.vfmadd231pd %158, %180, %202 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %205 = x86.dm.vbroadcastsd [%175 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %206 = x86.rss.vfmadd231pd %160, %179, %205 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %207 = x86.rss.vfmadd231pd %161, %180, %205 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %208 = x86.dm.vbroadcastsd [%175 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %209 = x86.rss.vfmadd231pd %163, %179, %208 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %210 = x86.rss.vfmadd231pd %164, %180, %208 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %211 = x86.dm.vbroadcastsd [%175 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %212 = x86.rss.vfmadd231pd %166, %179, %211 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %213 = x86.rss.vfmadd231pd %167, %180, %211 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %214 = x86.dm.vbroadcastsd [%175 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %215 = x86.rss.vfmadd231pd %169, %179, %214 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %216 = x86.rss.vfmadd231pd %170, %180, %214 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %217 = x86.dm.vbroadcastsd [%175 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %218 = x86.rss.vfmadd231pd %172, %179, %217 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %219 = x86.rss.vfmadd231pd %173, %180, %217 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %220 = x86.dm.vbroadcastsd [%175 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %221 = x86.ri.add %175, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %222 = x86.ri.add %176, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %223 = x86.rss.vfmadd231pd %177, %179, %220 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %224 = x86.rss.vfmadd231pd %178, %180, %220 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %225 = x86.dm.vmovapd [%222] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %226 = x86.dm.vmovapd [%222 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %227 = x86.dm.vbroadcastsd [%221] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %228 = x86.rss.vfmadd231pd %182, %225, %227 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %229 = x86.rss.vfmadd231pd %183, %226, %227 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:        %230 = x86.dm.vbroadcastsd [%221 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %231 = x86.rss.vfmadd231pd %185, %225, %230 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %232 = x86.rss.vfmadd231pd %186, %226, %230 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %233 = x86.dm.vbroadcastsd [%221 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %234 = x86.rss.vfmadd231pd %188, %225, %233 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %235 = x86.rss.vfmadd231pd %189, %226, %233 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %236 = x86.dm.vbroadcastsd [%221 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %237 = x86.rss.vfmadd231pd %191, %225, %236 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %238 = x86.rss.vfmadd231pd %192, %226, %236 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %239 = x86.dm.vbroadcastsd [%221 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %240 = x86.rss.vfmadd231pd %194, %225, %239 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %241 = x86.rss.vfmadd231pd %195, %226, %239 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %242 = x86.dm.vbroadcastsd [%221 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %243 = x86.rss.vfmadd231pd %197, %225, %242 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %244 = x86.rss.vfmadd231pd %198, %226, %242 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %245 = x86.dm.vbroadcastsd [%221 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %246 = x86.rss.vfmadd231pd %200, %225, %245 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %247 = x86.rss.vfmadd231pd %201, %226, %245 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %248 = x86.dm.vbroadcastsd [%221 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %249 = x86.rss.vfmadd231pd %203, %225, %248 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %250 = x86.rss.vfmadd231pd %204, %226, %248 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %251 = x86.dm.vbroadcastsd [%221 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %252 = x86.rss.vfmadd231pd %206, %225, %251 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %253 = x86.rss.vfmadd231pd %207, %226, %251 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %254 = x86.dm.vbroadcastsd [%221 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %255 = x86.rss.vfmadd231pd %209, %225, %254 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %256 = x86.rss.vfmadd231pd %210, %226, %254 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %257 = x86.dm.vbroadcastsd [%221 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %258 = x86.rss.vfmadd231pd %212, %225, %257 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %259 = x86.rss.vfmadd231pd %213, %226, %257 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %260 = x86.dm.vbroadcastsd [%221 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %261 = x86.rss.vfmadd231pd %215, %225, %260 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %262 = x86.rss.vfmadd231pd %216, %226, %260 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %263 = x86.dm.vbroadcastsd [%221 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %264 = x86.rss.vfmadd231pd %218, %225, %263 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %265 = x86.rss.vfmadd231pd %219, %226, %263 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %266 = x86.dm.vbroadcastsd [%221 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %267 = x86.ri.add %221, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %268 = x86.ri.add %222, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %269 = x86.rss.vfmadd231pd %223, %225, %266 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %270 = x86.rss.vfmadd231pd %224, %226, %266 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %271 = x86.dm.vmovapd [%268] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %272 = x86.dm.vmovapd [%268 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %273 = x86.dm.vbroadcastsd [%267] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %274 = x86.rss.vfmadd231pd %228, %271, %273 : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %275 = x86.rss.vfmadd231pd %229, %272, %273 : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm5>
// CHECK-NEXT:        %276 = x86.dm.vbroadcastsd [%267 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %277 = x86.rss.vfmadd231pd %231, %271, %276 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %278 = x86.rss.vfmadd231pd %232, %272, %276 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %279 = x86.dm.vbroadcastsd [%267 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %280 = x86.rss.vfmadd231pd %234, %271, %279 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %281 = x86.rss.vfmadd231pd %235, %272, %279 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %282 = x86.dm.vbroadcastsd [%267 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %283 = x86.rss.vfmadd231pd %237, %271, %282 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %284 = x86.rss.vfmadd231pd %238, %272, %282 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %285 = x86.dm.vbroadcastsd [%267 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %286 = x86.rss.vfmadd231pd %240, %271, %285 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %287 = x86.rss.vfmadd231pd %241, %272, %285 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %288 = x86.dm.vbroadcastsd [%267 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %289 = x86.rss.vfmadd231pd %243, %271, %288 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %290 = x86.rss.vfmadd231pd %244, %272, %288 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %291 = x86.dm.vbroadcastsd [%267 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %292 = x86.rss.vfmadd231pd %246, %271, %291 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %293 = x86.rss.vfmadd231pd %247, %272, %291 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %294 = x86.dm.vbroadcastsd [%267 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %295 = x86.rss.vfmadd231pd %249, %271, %294 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %296 = x86.rss.vfmadd231pd %250, %272, %294 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %297 = x86.dm.vbroadcastsd [%267 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %298 = x86.rss.vfmadd231pd %252, %271, %297 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %299 = x86.rss.vfmadd231pd %253, %272, %297 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %300 = x86.dm.vbroadcastsd [%267 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %301 = x86.rss.vfmadd231pd %255, %271, %300 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %302 = x86.rss.vfmadd231pd %256, %272, %300 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %303 = x86.dm.vbroadcastsd [%267 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %304 = x86.rss.vfmadd231pd %258, %271, %303 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %305 = x86.rss.vfmadd231pd %259, %272, %303 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %306 = x86.dm.vbroadcastsd [%267 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %307 = x86.rss.vfmadd231pd %261, %271, %306 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %308 = x86.rss.vfmadd231pd %262, %272, %306 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %309 = x86.dm.vbroadcastsd [%267 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %310 = x86.rss.vfmadd231pd %264, %271, %309 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %311 = x86.rss.vfmadd231pd %265, %272, %309 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %312 = x86.dm.vbroadcastsd [%267 + 2496] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %313 = x86.ri.add %267, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %314 = x86.ri.add %268, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %315 = x86.rss.vfmadd231pd %269, %271, %312 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %316 = x86.rss.vfmadd231pd %270, %272, %312 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %314, %313, %102, %103, %104, %274, %275, %277, %278, %280, %281, %283, %284, %286, %287, %289, %290, %292, %293, %295, %296, %298, %299, %301, %302, %304, %305, %307, %308, %310, %311, %315, %316 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm5>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %317 = x86.ri.sub %67, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%68], %71 : (!x86.reg64<rdx>, !x86.avx512reg<zmm4>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 64], %72 : (!x86.reg64<rdx>, !x86.avx512reg<zmm5>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 128], %73 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 192], %74 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 256], %75 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 320], %76 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 384], %77 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 448], %78 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 512], %79 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 576], %80 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 640], %81 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 704], %82 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 768], %83 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 832], %84 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 896], %85 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 960], %86 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1024], %87 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1088], %88 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1152], %89 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1216], %90 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1280], %91 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1344], %92 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1408], %93 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1472], %94 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1536], %95 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1600], %96 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1664], %97 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%68 + 1728], %98 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %318 = x86.ri.add %68, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %319 = x86.ri.sub %66, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %319, %317, %318, %69, %70 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %320 = x86.ri.add %27, 1664 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %321 = x86.ri.add %26, 2688 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %322 = x86.ri.sub %25, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %322, %321, %320, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %323 = x86.di.mov 14 : () -> !x86.reg64<r11>
// CHECK-NEXT:  %324, %325, %326, %327, %328, %329 = x86_scf.for %330 : !x86.reg64<r11>  = %323 to 66 : si32 step 13 : si32 iter_args(%331 = %12, %332 = %13, %333 = %14, %334 = %15, %335 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:    %336 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %337, %338, %339, %340, %341, %342 = x86_scf.for %343 : !x86.reg64<r10>  = %336 to 16 : si32 step 16 : si32 iter_args(%344 = %331, %345 = %332, %346 = %333, %347 = %334, %348 = %335) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %349 = x86.dm.vmovapd [%346] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:      %350 = x86.dm.vmovapd [%346 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:      %351 = x86.dm.vmovapd [%346 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:      %352 = x86.dm.vmovapd [%346 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:      %353 = x86.dm.vmovapd [%346 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:      %354 = x86.dm.vmovapd [%346 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:      %355 = x86.dm.vmovapd [%346 + 384] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:      %356 = x86.dm.vmovapd [%346 + 448] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:      %357 = x86.dm.vmovapd [%346 + 512] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:      %358 = x86.dm.vmovapd [%346 + 576] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:      %359 = x86.dm.vmovapd [%346 + 640] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:      %360 = x86.dm.vmovapd [%346 + 704] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:      %361 = x86.dm.vmovapd [%346 + 768] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:      %362 = x86.dm.vmovapd [%346 + 832] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:      %363 = x86.dm.vmovapd [%346 + 896] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:      %364 = x86.dm.vmovapd [%346 + 960] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:      %365 = x86.dm.vmovapd [%346 + 1024] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:      %366 = x86.dm.vmovapd [%346 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:      %367 = x86.dm.vmovapd [%346 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:      %368 = x86.dm.vmovapd [%346 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:      %369 = x86.dm.vmovapd [%346 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %370 = x86.dm.vmovapd [%346 + 1344] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %371 = x86.dm.vmovapd [%346 + 1408] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %372 = x86.dm.vmovapd [%346 + 1472] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %373 = x86.dm.vmovapd [%346 + 1536] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %374 = x86.dm.vmovapd [%346 + 1600] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %375 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %376, %377, %378, %379, %380, %381, %382, %383, %384, %385, %386, %387, %388, %389, %390, %391, %392, %393, %394, %395, %396, %397, %398, %399, %400, %401, %402, %403, %404, %405, %406, %407 = x86_scf.for %408 : !x86.reg64<r12>  = %375 to 24 : si32 step 4 : si32 iter_args(%409 = %344, %410 = %345, %411 = %346, %412 = %347, %413 = %348, %414 = %349, %415 = %350, %416 = %351, %417 = %352, %418 = %353, %419 = %354, %420 = %355, %421 = %356, %422 = %357, %423 = %358, %424 = %359, %425 = %360, %426 = %361, %427 = %362, %428 = %363, %429 = %364, %430 = %365, %431 = %366, %432 = %367, %433 = %368, %434 = %369, %435 = %370, %436 = %371, %437 = %372, %438 = %373, %439 = %374) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %440 = x86.dm.vmovapd [%409] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %441 = x86.dm.vmovapd [%409 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %442 = x86.dm.vbroadcastsd [%410] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %443 = x86.rss.vfmadd231pd %414, %440, %442 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %444 = x86.rss.vfmadd231pd %415, %441, %442 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %445 = x86.dm.vbroadcastsd [%410 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %446 = x86.rss.vfmadd231pd %416, %440, %445 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %447 = x86.rss.vfmadd231pd %417, %441, %445 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %448 = x86.dm.vbroadcastsd [%410 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %449 = x86.rss.vfmadd231pd %418, %440, %448 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %450 = x86.rss.vfmadd231pd %419, %441, %448 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %451 = x86.dm.vbroadcastsd [%410 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %452 = x86.rss.vfmadd231pd %420, %440, %451 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %453 = x86.rss.vfmadd231pd %421, %441, %451 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %454 = x86.dm.vbroadcastsd [%410 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %455 = x86.rss.vfmadd231pd %422, %440, %454 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %456 = x86.rss.vfmadd231pd %423, %441, %454 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %457 = x86.dm.vbroadcastsd [%410 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %458 = x86.rss.vfmadd231pd %424, %440, %457 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %459 = x86.rss.vfmadd231pd %425, %441, %457 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %460 = x86.dm.vbroadcastsd [%410 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %461 = x86.rss.vfmadd231pd %426, %440, %460 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %462 = x86.rss.vfmadd231pd %427, %441, %460 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %463 = x86.dm.vbroadcastsd [%410 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %464 = x86.rss.vfmadd231pd %428, %440, %463 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %465 = x86.rss.vfmadd231pd %429, %441, %463 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %466 = x86.dm.vbroadcastsd [%410 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %467 = x86.rss.vfmadd231pd %430, %440, %466 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %468 = x86.rss.vfmadd231pd %431, %441, %466 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %469 = x86.dm.vbroadcastsd [%410 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %470 = x86.rss.vfmadd231pd %432, %440, %469 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %471 = x86.rss.vfmadd231pd %433, %441, %469 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %472 = x86.dm.vbroadcastsd [%410 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %473 = x86.rss.vfmadd231pd %434, %440, %472 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %474 = x86.rss.vfmadd231pd %435, %441, %472 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %475 = x86.dm.vbroadcastsd [%410 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %476 = x86.rss.vfmadd231pd %436, %440, %475 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %477 = x86.rss.vfmadd231pd %437, %441, %475 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %478 = x86.dm.vbroadcastsd [%410 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %479 = x86.ri.add %410, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %480 = x86.ri.add %409, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %481 = x86.rss.vfmadd231pd %438, %440, %478 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %482 = x86.rss.vfmadd231pd %439, %441, %478 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %483 = x86.dm.vmovapd [%480] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %484 = x86.dm.vmovapd [%480 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %485 = x86.dm.vbroadcastsd [%479] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %486 = x86.rss.vfmadd231pd %443, %483, %485 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %487 = x86.rss.vfmadd231pd %444, %484, %485 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %488 = x86.dm.vbroadcastsd [%479 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %489 = x86.rss.vfmadd231pd %446, %483, %488 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %490 = x86.rss.vfmadd231pd %447, %484, %488 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %491 = x86.dm.vbroadcastsd [%479 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %492 = x86.rss.vfmadd231pd %449, %483, %491 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %493 = x86.rss.vfmadd231pd %450, %484, %491 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %494 = x86.dm.vbroadcastsd [%479 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %495 = x86.rss.vfmadd231pd %452, %483, %494 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %496 = x86.rss.vfmadd231pd %453, %484, %494 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %497 = x86.dm.vbroadcastsd [%479 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %498 = x86.rss.vfmadd231pd %455, %483, %497 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %499 = x86.rss.vfmadd231pd %456, %484, %497 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %500 = x86.dm.vbroadcastsd [%479 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %501 = x86.rss.vfmadd231pd %458, %483, %500 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %502 = x86.rss.vfmadd231pd %459, %484, %500 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %503 = x86.dm.vbroadcastsd [%479 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %504 = x86.rss.vfmadd231pd %461, %483, %503 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %505 = x86.rss.vfmadd231pd %462, %484, %503 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %506 = x86.dm.vbroadcastsd [%479 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %507 = x86.rss.vfmadd231pd %464, %483, %506 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %508 = x86.rss.vfmadd231pd %465, %484, %506 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %509 = x86.dm.vbroadcastsd [%479 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %510 = x86.rss.vfmadd231pd %467, %483, %509 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %511 = x86.rss.vfmadd231pd %468, %484, %509 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %512 = x86.dm.vbroadcastsd [%479 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %513 = x86.rss.vfmadd231pd %470, %483, %512 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %514 = x86.rss.vfmadd231pd %471, %484, %512 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %515 = x86.dm.vbroadcastsd [%479 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %516 = x86.rss.vfmadd231pd %473, %483, %515 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %517 = x86.rss.vfmadd231pd %474, %484, %515 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %518 = x86.dm.vbroadcastsd [%479 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %519 = x86.rss.vfmadd231pd %476, %483, %518 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %520 = x86.rss.vfmadd231pd %477, %484, %518 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %521 = x86.dm.vbroadcastsd [%479 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %522 = x86.ri.add %479, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %523 = x86.ri.add %480, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %524 = x86.rss.vfmadd231pd %481, %483, %521 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %525 = x86.rss.vfmadd231pd %482, %484, %521 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %526 = x86.dm.vmovapd [%523] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %527 = x86.dm.vmovapd [%523 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %528 = x86.dm.vbroadcastsd [%522] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %529 = x86.rss.vfmadd231pd %486, %526, %528 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %530 = x86.rss.vfmadd231pd %487, %527, %528 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %531 = x86.dm.vbroadcastsd [%522 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %532 = x86.rss.vfmadd231pd %489, %526, %531 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %533 = x86.rss.vfmadd231pd %490, %527, %531 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %534 = x86.dm.vbroadcastsd [%522 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %535 = x86.rss.vfmadd231pd %492, %526, %534 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %536 = x86.rss.vfmadd231pd %493, %527, %534 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %537 = x86.dm.vbroadcastsd [%522 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %538 = x86.rss.vfmadd231pd %495, %526, %537 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %539 = x86.rss.vfmadd231pd %496, %527, %537 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %540 = x86.dm.vbroadcastsd [%522 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %541 = x86.rss.vfmadd231pd %498, %526, %540 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %542 = x86.rss.vfmadd231pd %499, %527, %540 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %543 = x86.dm.vbroadcastsd [%522 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %544 = x86.rss.vfmadd231pd %501, %526, %543 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %545 = x86.rss.vfmadd231pd %502, %527, %543 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %546 = x86.dm.vbroadcastsd [%522 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %547 = x86.rss.vfmadd231pd %504, %526, %546 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %548 = x86.rss.vfmadd231pd %505, %527, %546 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %549 = x86.dm.vbroadcastsd [%522 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %550 = x86.rss.vfmadd231pd %507, %526, %549 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %551 = x86.rss.vfmadd231pd %508, %527, %549 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %552 = x86.dm.vbroadcastsd [%522 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %553 = x86.rss.vfmadd231pd %510, %526, %552 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %554 = x86.rss.vfmadd231pd %511, %527, %552 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %555 = x86.dm.vbroadcastsd [%522 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %556 = x86.rss.vfmadd231pd %513, %526, %555 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %557 = x86.rss.vfmadd231pd %514, %527, %555 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %558 = x86.dm.vbroadcastsd [%522 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %559 = x86.rss.vfmadd231pd %516, %526, %558 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %560 = x86.rss.vfmadd231pd %517, %527, %558 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %561 = x86.dm.vbroadcastsd [%522 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %562 = x86.rss.vfmadd231pd %519, %526, %561 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %563 = x86.rss.vfmadd231pd %520, %527, %561 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %564 = x86.dm.vbroadcastsd [%522 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %565 = x86.ri.add %522, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %566 = x86.ri.add %523, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %567 = x86.rss.vfmadd231pd %524, %526, %564 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %568 = x86.rss.vfmadd231pd %525, %527, %564 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %569 = x86.dm.vmovapd [%566] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %570 = x86.dm.vmovapd [%566 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %571 = x86.dm.vbroadcastsd [%565] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %572 = x86.rss.vfmadd231pd %529, %569, %571 : (!x86.avx512reg<zmm6>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm6>
// CHECK-NEXT:        %573 = x86.rss.vfmadd231pd %530, %570, %571 : (!x86.avx512reg<zmm7>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm7>
// CHECK-NEXT:        %574 = x86.dm.vbroadcastsd [%565 + 192] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %575 = x86.rss.vfmadd231pd %532, %569, %574 : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
// CHECK-NEXT:        %576 = x86.rss.vfmadd231pd %533, %570, %574 : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm9>
// CHECK-NEXT:        %577 = x86.dm.vbroadcastsd [%565 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %578 = x86.rss.vfmadd231pd %535, %569, %577 : (!x86.avx512reg<zmm10>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
// CHECK-NEXT:        %579 = x86.rss.vfmadd231pd %536, %570, %577 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm11>
// CHECK-NEXT:        %580 = x86.dm.vbroadcastsd [%565 + 576] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %581 = x86.rss.vfmadd231pd %538, %569, %580 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %582 = x86.rss.vfmadd231pd %539, %570, %580 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %583 = x86.dm.vbroadcastsd [%565 + 768] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %584 = x86.rss.vfmadd231pd %541, %569, %583 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %585 = x86.rss.vfmadd231pd %542, %570, %583 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %586 = x86.dm.vbroadcastsd [%565 + 960] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %587 = x86.rss.vfmadd231pd %544, %569, %586 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %588 = x86.rss.vfmadd231pd %545, %570, %586 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %589 = x86.dm.vbroadcastsd [%565 + 1152] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %590 = x86.rss.vfmadd231pd %547, %569, %589 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %591 = x86.rss.vfmadd231pd %548, %570, %589 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %592 = x86.dm.vbroadcastsd [%565 + 1344] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %593 = x86.rss.vfmadd231pd %550, %569, %592 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %594 = x86.rss.vfmadd231pd %551, %570, %592 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %595 = x86.dm.vbroadcastsd [%565 + 1536] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %596 = x86.rss.vfmadd231pd %553, %569, %595 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %597 = x86.rss.vfmadd231pd %554, %570, %595 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %598 = x86.dm.vbroadcastsd [%565 + 1728] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %599 = x86.rss.vfmadd231pd %556, %569, %598 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %600 = x86.rss.vfmadd231pd %557, %570, %598 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %601 = x86.dm.vbroadcastsd [%565 + 1920] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %602 = x86.rss.vfmadd231pd %559, %569, %601 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %603 = x86.rss.vfmadd231pd %560, %570, %601 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %604 = x86.dm.vbroadcastsd [%565 + 2112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %605 = x86.rss.vfmadd231pd %562, %569, %604 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %606 = x86.rss.vfmadd231pd %563, %570, %604 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %607 = x86.dm.vbroadcastsd [%565 + 2304] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %608 = x86.ri.add %565, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %609 = x86.ri.add %566, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %610 = x86.rss.vfmadd231pd %567, %569, %607 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %611 = x86.rss.vfmadd231pd %568, %570, %607 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %609, %608, %411, %412, %413, %572, %573, %575, %576, %578, %579, %581, %582, %584, %585, %587, %588, %590, %591, %593, %594, %596, %597, %599, %600, %602, %603, %605, %606, %610, %611 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm6>, !x86.avx512reg<zmm7>, !x86.avx512reg<zmm8>, !x86.avx512reg<zmm9>, !x86.avx512reg<zmm10>, !x86.avx512reg<zmm11>, !x86.avx512reg<zmm12>, !x86.avx512reg<zmm13>, !x86.avx512reg<zmm14>, !x86.avx512reg<zmm15>, !x86.avx512reg<zmm16>, !x86.avx512reg<zmm17>, !x86.avx512reg<zmm18>, !x86.avx512reg<zmm19>, !x86.avx512reg<zmm20>, !x86.avx512reg<zmm21>, !x86.avx512reg<zmm22>, !x86.avx512reg<zmm23>, !x86.avx512reg<zmm24>, !x86.avx512reg<zmm25>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %612 = x86.ri.sub %378, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%379], %382 : (!x86.reg64<rdx>, !x86.avx512reg<zmm6>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 64], %383 : (!x86.reg64<rdx>, !x86.avx512reg<zmm7>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 128], %384 : (!x86.reg64<rdx>, !x86.avx512reg<zmm8>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 192], %385 : (!x86.reg64<rdx>, !x86.avx512reg<zmm9>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 256], %386 : (!x86.reg64<rdx>, !x86.avx512reg<zmm10>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 320], %387 : (!x86.reg64<rdx>, !x86.avx512reg<zmm11>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 384], %388 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 448], %389 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 512], %390 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 576], %391 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 640], %392 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 704], %393 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 768], %394 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 832], %395 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 896], %396 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 960], %397 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1024], %398 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1088], %399 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1152], %400 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1216], %401 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1280], %402 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1344], %403 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1408], %404 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1472], %405 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1536], %406 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%379 + 1600], %407 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %613 = x86.ri.add %379, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %614 = x86.ri.sub %377, 2944 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %614, %612, %613, %380, %381 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %615 = x86.ri.add %340, 1536 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %616 = x86.ri.add %339, 2496 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %617 = x86.ri.sub %338, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    x86_scf.yield %617, %616, %615, %341, %342 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:  }
// CHECK-NEXT:  %618 = x86.ds.mov %328 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:  %619, %620 = x86.d.pop %618 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
