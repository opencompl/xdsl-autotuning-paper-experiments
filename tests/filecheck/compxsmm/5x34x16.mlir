// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 5 : si32 step 5 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 32 : si32 step 32 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovupd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %37 = x86.dm.vmovupd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %38 = x86.dm.vmovupd [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %39 = x86.dm.vmovupd [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %40 = x86.dm.vmovupd [%33 + 272] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %41 = x86.dm.vmovupd [%33 + 336] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %42 = x86.dm.vmovupd [%33 + 400] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %43 = x86.dm.vmovupd [%33 + 464] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %44 = x86.dm.vmovupd [%33 + 544] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %45 = x86.dm.vmovupd [%33 + 608] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %46 = x86.dm.vmovupd [%33 + 672] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %47 = x86.dm.vmovupd [%33 + 736] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %48 = x86.dm.vmovupd [%33 + 816] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %49 = x86.dm.vmovupd [%33 + 880] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %50 = x86.dm.vmovupd [%33 + 944] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %51 = x86.dm.vmovupd [%33 + 1008] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %52 = x86.dm.vmovupd [%33 + 1088] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %53 = x86.dm.vmovupd [%33 + 1152] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %54 = x86.dm.vmovupd [%33 + 1216] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %55 = x86.dm.vmovupd [%33 + 1280] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %56 = x86.dm.vmovupd [%31] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %57 = x86.dm.vmovupd [%31 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %58 = x86.dm.vmovupd [%31 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %59 = x86.dm.vmovupd [%31 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %60 = x86.dm.vbroadcastsd [%32] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %61 = x86.rss.vfmadd231pd %36, %56, %60 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %62 = x86.rss.vfmadd231pd %37, %57, %60 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %63 = x86.rss.vfmadd231pd %38, %58, %60 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %64 = x86.rss.vfmadd231pd %39, %59, %60 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %65 = x86.dm.vbroadcastsd [%32 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %66 = x86.rss.vfmadd231pd %40, %56, %65 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %67 = x86.rss.vfmadd231pd %41, %57, %65 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %68 = x86.rss.vfmadd231pd %42, %58, %65 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %69 = x86.rss.vfmadd231pd %43, %59, %65 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %70 = x86.dm.vbroadcastsd [%32 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %71 = x86.rss.vfmadd231pd %44, %56, %70 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %72 = x86.rss.vfmadd231pd %45, %57, %70 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %73 = x86.rss.vfmadd231pd %46, %58, %70 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %74 = x86.rss.vfmadd231pd %47, %59, %70 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %75 = x86.dm.vbroadcastsd [%32 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %76 = x86.rss.vfmadd231pd %48, %56, %75 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %77 = x86.rss.vfmadd231pd %49, %57, %75 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %78 = x86.rss.vfmadd231pd %50, %58, %75 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %79 = x86.rss.vfmadd231pd %51, %59, %75 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %80 = x86.dm.vbroadcastsd [%32 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %81 = x86.ri.add %32, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %82 = x86.ri.add %31, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %83 = x86.rss.vfmadd231pd %52, %56, %80 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %84 = x86.rss.vfmadd231pd %53, %57, %80 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %85 = x86.rss.vfmadd231pd %54, %58, %80 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %86 = x86.rss.vfmadd231pd %55, %59, %80 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %87 = x86.dm.vmovupd [%82] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %88 = x86.dm.vmovupd [%82 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %89 = x86.dm.vmovupd [%82 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %90 = x86.dm.vmovupd [%82 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %91 = x86.dm.vbroadcastsd [%81] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %92 = x86.rss.vfmadd231pd %61, %87, %91 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %93 = x86.rss.vfmadd231pd %62, %88, %91 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %94 = x86.rss.vfmadd231pd %63, %89, %91 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %95 = x86.rss.vfmadd231pd %64, %90, %91 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %96 = x86.dm.vbroadcastsd [%81 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %97 = x86.rss.vfmadd231pd %66, %87, %96 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %98 = x86.rss.vfmadd231pd %67, %88, %96 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %99 = x86.rss.vfmadd231pd %68, %89, %96 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %100 = x86.rss.vfmadd231pd %69, %90, %96 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %101 = x86.dm.vbroadcastsd [%81 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %102 = x86.rss.vfmadd231pd %71, %87, %101 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %103 = x86.rss.vfmadd231pd %72, %88, %101 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %104 = x86.rss.vfmadd231pd %73, %89, %101 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %105 = x86.rss.vfmadd231pd %74, %90, %101 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %106 = x86.dm.vbroadcastsd [%81 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %107 = x86.rss.vfmadd231pd %76, %87, %106 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %108 = x86.rss.vfmadd231pd %77, %88, %106 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %109 = x86.rss.vfmadd231pd %78, %89, %106 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %110 = x86.rss.vfmadd231pd %79, %90, %106 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %111 = x86.dm.vbroadcastsd [%81 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %112 = x86.ri.add %81, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %113 = x86.ri.add %82, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %114 = x86.rss.vfmadd231pd %83, %87, %111 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %115 = x86.rss.vfmadd231pd %84, %88, %111 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %116 = x86.rss.vfmadd231pd %85, %89, %111 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %117 = x86.rss.vfmadd231pd %86, %90, %111 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %118 = x86.dm.vmovupd [%113] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %119 = x86.dm.vmovupd [%113 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %120 = x86.dm.vmovupd [%113 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %121 = x86.dm.vmovupd [%113 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %122 = x86.dm.vbroadcastsd [%112] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %123 = x86.rss.vfmadd231pd %92, %118, %122 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %124 = x86.rss.vfmadd231pd %93, %119, %122 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %125 = x86.rss.vfmadd231pd %94, %120, %122 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %126 = x86.rss.vfmadd231pd %95, %121, %122 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %127 = x86.dm.vbroadcastsd [%112 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %128 = x86.rss.vfmadd231pd %97, %118, %127 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %129 = x86.rss.vfmadd231pd %98, %119, %127 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %130 = x86.rss.vfmadd231pd %99, %120, %127 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %131 = x86.rss.vfmadd231pd %100, %121, %127 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %132 = x86.dm.vbroadcastsd [%112 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %133 = x86.rss.vfmadd231pd %102, %118, %132 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %134 = x86.rss.vfmadd231pd %103, %119, %132 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %135 = x86.rss.vfmadd231pd %104, %120, %132 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %136 = x86.rss.vfmadd231pd %105, %121, %132 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %137 = x86.dm.vbroadcastsd [%112 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %138 = x86.rss.vfmadd231pd %107, %118, %137 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %139 = x86.rss.vfmadd231pd %108, %119, %137 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %140 = x86.rss.vfmadd231pd %109, %120, %137 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %141 = x86.rss.vfmadd231pd %110, %121, %137 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %142 = x86.dm.vbroadcastsd [%112 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %143 = x86.ri.add %112, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %144 = x86.ri.add %113, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %145 = x86.rss.vfmadd231pd %114, %118, %142 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %146 = x86.rss.vfmadd231pd %115, %119, %142 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %147 = x86.rss.vfmadd231pd %116, %120, %142 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %148 = x86.rss.vfmadd231pd %117, %121, %142 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %149 = x86.dm.vmovupd [%144] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %150 = x86.dm.vmovupd [%144 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %151 = x86.dm.vmovupd [%144 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %152 = x86.dm.vmovupd [%144 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %153 = x86.dm.vbroadcastsd [%143] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %154 = x86.rss.vfmadd231pd %123, %149, %153 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %155 = x86.rss.vfmadd231pd %124, %150, %153 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %156 = x86.rss.vfmadd231pd %125, %151, %153 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %157 = x86.rss.vfmadd231pd %126, %152, %153 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %158 = x86.dm.vbroadcastsd [%143 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %159 = x86.rss.vfmadd231pd %128, %149, %158 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %160 = x86.rss.vfmadd231pd %129, %150, %158 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %161 = x86.rss.vfmadd231pd %130, %151, %158 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %162 = x86.rss.vfmadd231pd %131, %152, %158 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %163 = x86.dm.vbroadcastsd [%143 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %164 = x86.rss.vfmadd231pd %133, %149, %163 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %165 = x86.rss.vfmadd231pd %134, %150, %163 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %166 = x86.rss.vfmadd231pd %135, %151, %163 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %167 = x86.rss.vfmadd231pd %136, %152, %163 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %168 = x86.dm.vbroadcastsd [%143 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %169 = x86.rss.vfmadd231pd %138, %149, %168 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %170 = x86.rss.vfmadd231pd %139, %150, %168 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %171 = x86.rss.vfmadd231pd %140, %151, %168 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %172 = x86.rss.vfmadd231pd %141, %152, %168 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %173 = x86.dm.vbroadcastsd [%143 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %174 = x86.ri.add %143, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %175 = x86.ri.add %144, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %176 = x86.rss.vfmadd231pd %145, %149, %173 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %177 = x86.rss.vfmadd231pd %146, %150, %173 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %178 = x86.rss.vfmadd231pd %147, %151, %173 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %179 = x86.rss.vfmadd231pd %148, %152, %173 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %180 = x86.dm.vmovupd [%175] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %181 = x86.dm.vmovupd [%175 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %182 = x86.dm.vmovupd [%175 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %183 = x86.dm.vmovupd [%175 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %184 = x86.dm.vbroadcastsd [%174] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %185 = x86.rss.vfmadd231pd %154, %180, %184 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %186 = x86.rss.vfmadd231pd %155, %181, %184 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %187 = x86.rss.vfmadd231pd %156, %182, %184 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %188 = x86.rss.vfmadd231pd %157, %183, %184 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %189 = x86.dm.vbroadcastsd [%174 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %190 = x86.rss.vfmadd231pd %159, %180, %189 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %191 = x86.rss.vfmadd231pd %160, %181, %189 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %192 = x86.rss.vfmadd231pd %161, %182, %189 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %193 = x86.rss.vfmadd231pd %162, %183, %189 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %194 = x86.dm.vbroadcastsd [%174 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %195 = x86.rss.vfmadd231pd %164, %180, %194 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %196 = x86.rss.vfmadd231pd %165, %181, %194 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %197 = x86.rss.vfmadd231pd %166, %182, %194 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %198 = x86.rss.vfmadd231pd %167, %183, %194 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %199 = x86.dm.vbroadcastsd [%174 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %200 = x86.rss.vfmadd231pd %169, %180, %199 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %201 = x86.rss.vfmadd231pd %170, %181, %199 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %202 = x86.rss.vfmadd231pd %171, %182, %199 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %203 = x86.rss.vfmadd231pd %172, %183, %199 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %204 = x86.dm.vbroadcastsd [%174 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %205 = x86.ri.add %174, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %206 = x86.ri.add %175, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %207 = x86.rss.vfmadd231pd %176, %180, %204 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %208 = x86.rss.vfmadd231pd %177, %181, %204 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %209 = x86.rss.vfmadd231pd %178, %182, %204 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %210 = x86.rss.vfmadd231pd %179, %183, %204 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %211 = x86.dm.vmovupd [%206] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %212 = x86.dm.vmovupd [%206 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %213 = x86.dm.vmovupd [%206 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %214 = x86.dm.vmovupd [%206 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %215 = x86.dm.vbroadcastsd [%205] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %216 = x86.rss.vfmadd231pd %185, %211, %215 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %217 = x86.rss.vfmadd231pd %186, %212, %215 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %218 = x86.rss.vfmadd231pd %187, %213, %215 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %219 = x86.rss.vfmadd231pd %188, %214, %215 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %220 = x86.dm.vbroadcastsd [%205 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %221 = x86.rss.vfmadd231pd %190, %211, %220 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %222 = x86.rss.vfmadd231pd %191, %212, %220 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %223 = x86.rss.vfmadd231pd %192, %213, %220 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %224 = x86.rss.vfmadd231pd %193, %214, %220 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %225 = x86.dm.vbroadcastsd [%205 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %226 = x86.rss.vfmadd231pd %195, %211, %225 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %227 = x86.rss.vfmadd231pd %196, %212, %225 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %228 = x86.rss.vfmadd231pd %197, %213, %225 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %229 = x86.rss.vfmadd231pd %198, %214, %225 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %230 = x86.dm.vbroadcastsd [%205 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %231 = x86.rss.vfmadd231pd %200, %211, %230 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %232 = x86.rss.vfmadd231pd %201, %212, %230 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %233 = x86.rss.vfmadd231pd %202, %213, %230 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %234 = x86.rss.vfmadd231pd %203, %214, %230 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %235 = x86.dm.vbroadcastsd [%205 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %236 = x86.ri.add %205, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %237 = x86.ri.add %206, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %238 = x86.rss.vfmadd231pd %207, %211, %235 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %239 = x86.rss.vfmadd231pd %208, %212, %235 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %240 = x86.rss.vfmadd231pd %209, %213, %235 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %241 = x86.rss.vfmadd231pd %210, %214, %235 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %242 = x86.dm.vmovupd [%237] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %243 = x86.dm.vmovupd [%237 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %244 = x86.dm.vmovupd [%237 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %245 = x86.dm.vmovupd [%237 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %246 = x86.dm.vbroadcastsd [%236] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %247 = x86.rss.vfmadd231pd %216, %242, %246 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %248 = x86.rss.vfmadd231pd %217, %243, %246 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %249 = x86.rss.vfmadd231pd %218, %244, %246 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %250 = x86.rss.vfmadd231pd %219, %245, %246 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %251 = x86.dm.vbroadcastsd [%236 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %252 = x86.rss.vfmadd231pd %221, %242, %251 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %253 = x86.rss.vfmadd231pd %222, %243, %251 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %254 = x86.rss.vfmadd231pd %223, %244, %251 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %255 = x86.rss.vfmadd231pd %224, %245, %251 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %256 = x86.dm.vbroadcastsd [%236 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %257 = x86.rss.vfmadd231pd %226, %242, %256 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %258 = x86.rss.vfmadd231pd %227, %243, %256 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %259 = x86.rss.vfmadd231pd %228, %244, %256 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %260 = x86.rss.vfmadd231pd %229, %245, %256 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %261 = x86.dm.vbroadcastsd [%236 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %262 = x86.rss.vfmadd231pd %231, %242, %261 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %263 = x86.rss.vfmadd231pd %232, %243, %261 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %264 = x86.rss.vfmadd231pd %233, %244, %261 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %265 = x86.rss.vfmadd231pd %234, %245, %261 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %266 = x86.dm.vbroadcastsd [%236 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %267 = x86.ri.add %236, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %268 = x86.ri.add %237, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %269 = x86.rss.vfmadd231pd %238, %242, %266 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %270 = x86.rss.vfmadd231pd %239, %243, %266 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %271 = x86.rss.vfmadd231pd %240, %244, %266 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %272 = x86.rss.vfmadd231pd %241, %245, %266 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %273 = x86.dm.vmovupd [%268] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %274 = x86.dm.vmovupd [%268 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %275 = x86.dm.vmovupd [%268 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %276 = x86.dm.vmovupd [%268 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %277 = x86.dm.vbroadcastsd [%267] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %278 = x86.rss.vfmadd231pd %247, %273, %277 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %279 = x86.rss.vfmadd231pd %248, %274, %277 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %280 = x86.rss.vfmadd231pd %249, %275, %277 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %281 = x86.rss.vfmadd231pd %250, %276, %277 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %282 = x86.dm.vbroadcastsd [%267 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %283 = x86.rss.vfmadd231pd %252, %273, %282 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %284 = x86.rss.vfmadd231pd %253, %274, %282 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %285 = x86.rss.vfmadd231pd %254, %275, %282 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %286 = x86.rss.vfmadd231pd %255, %276, %282 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %287 = x86.dm.vbroadcastsd [%267 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %288 = x86.rss.vfmadd231pd %257, %273, %287 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %289 = x86.rss.vfmadd231pd %258, %274, %287 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %290 = x86.rss.vfmadd231pd %259, %275, %287 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %291 = x86.rss.vfmadd231pd %260, %276, %287 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %292 = x86.dm.vbroadcastsd [%267 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %293 = x86.rss.vfmadd231pd %262, %273, %292 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %294 = x86.rss.vfmadd231pd %263, %274, %292 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %295 = x86.rss.vfmadd231pd %264, %275, %292 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %296 = x86.rss.vfmadd231pd %265, %276, %292 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %297 = x86.dm.vbroadcastsd [%267 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %298 = x86.ri.add %267, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %299 = x86.ri.add %268, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %300 = x86.rss.vfmadd231pd %269, %273, %297 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %301 = x86.rss.vfmadd231pd %270, %274, %297 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %302 = x86.rss.vfmadd231pd %271, %275, %297 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %303 = x86.rss.vfmadd231pd %272, %276, %297 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %304 = x86.dm.vmovupd [%299] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %305 = x86.dm.vmovupd [%299 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %306 = x86.dm.vmovupd [%299 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %307 = x86.dm.vmovupd [%299 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %308 = x86.dm.vbroadcastsd [%298] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %309 = x86.rss.vfmadd231pd %278, %304, %308 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %310 = x86.rss.vfmadd231pd %279, %305, %308 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %311 = x86.rss.vfmadd231pd %280, %306, %308 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %312 = x86.rss.vfmadd231pd %281, %307, %308 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %313 = x86.dm.vbroadcastsd [%298 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %314 = x86.rss.vfmadd231pd %283, %304, %313 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %315 = x86.rss.vfmadd231pd %284, %305, %313 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %316 = x86.rss.vfmadd231pd %285, %306, %313 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %317 = x86.rss.vfmadd231pd %286, %307, %313 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %318 = x86.dm.vbroadcastsd [%298 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %319 = x86.rss.vfmadd231pd %288, %304, %318 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %320 = x86.rss.vfmadd231pd %289, %305, %318 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %321 = x86.rss.vfmadd231pd %290, %306, %318 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %322 = x86.rss.vfmadd231pd %291, %307, %318 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %323 = x86.dm.vbroadcastsd [%298 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %324 = x86.rss.vfmadd231pd %293, %304, %323 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %325 = x86.rss.vfmadd231pd %294, %305, %323 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %326 = x86.rss.vfmadd231pd %295, %306, %323 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %327 = x86.rss.vfmadd231pd %296, %307, %323 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %328 = x86.dm.vbroadcastsd [%298 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %329 = x86.ri.add %298, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %330 = x86.ri.add %299, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %331 = x86.rss.vfmadd231pd %300, %304, %328 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %332 = x86.rss.vfmadd231pd %301, %305, %328 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %333 = x86.rss.vfmadd231pd %302, %306, %328 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %334 = x86.rss.vfmadd231pd %303, %307, %328 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %335 = x86.dm.vmovupd [%330] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %336 = x86.dm.vmovupd [%330 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %337 = x86.dm.vmovupd [%330 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %338 = x86.dm.vmovupd [%330 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %339 = x86.dm.vbroadcastsd [%329] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %340 = x86.rss.vfmadd231pd %309, %335, %339 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %341 = x86.rss.vfmadd231pd %310, %336, %339 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %342 = x86.rss.vfmadd231pd %311, %337, %339 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %343 = x86.rss.vfmadd231pd %312, %338, %339 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %344 = x86.dm.vbroadcastsd [%329 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %345 = x86.rss.vfmadd231pd %314, %335, %344 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %346 = x86.rss.vfmadd231pd %315, %336, %344 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %347 = x86.rss.vfmadd231pd %316, %337, %344 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %348 = x86.rss.vfmadd231pd %317, %338, %344 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %349 = x86.dm.vbroadcastsd [%329 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %350 = x86.rss.vfmadd231pd %319, %335, %349 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %351 = x86.rss.vfmadd231pd %320, %336, %349 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %352 = x86.rss.vfmadd231pd %321, %337, %349 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %353 = x86.rss.vfmadd231pd %322, %338, %349 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %354 = x86.dm.vbroadcastsd [%329 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %355 = x86.rss.vfmadd231pd %324, %335, %354 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %356 = x86.rss.vfmadd231pd %325, %336, %354 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %357 = x86.rss.vfmadd231pd %326, %337, %354 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %358 = x86.rss.vfmadd231pd %327, %338, %354 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %359 = x86.dm.vbroadcastsd [%329 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %360 = x86.ri.add %329, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %361 = x86.ri.add %330, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %362 = x86.rss.vfmadd231pd %331, %335, %359 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %363 = x86.rss.vfmadd231pd %332, %336, %359 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %364 = x86.rss.vfmadd231pd %333, %337, %359 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %365 = x86.rss.vfmadd231pd %334, %338, %359 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %366 = x86.dm.vmovupd [%361] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %367 = x86.dm.vmovupd [%361 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %368 = x86.dm.vmovupd [%361 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %369 = x86.dm.vmovupd [%361 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %370 = x86.dm.vbroadcastsd [%360] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %371 = x86.rss.vfmadd231pd %340, %366, %370 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %372 = x86.rss.vfmadd231pd %341, %367, %370 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %373 = x86.rss.vfmadd231pd %342, %368, %370 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %374 = x86.rss.vfmadd231pd %343, %369, %370 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %375 = x86.dm.vbroadcastsd [%360 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %376 = x86.rss.vfmadd231pd %345, %366, %375 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %377 = x86.rss.vfmadd231pd %346, %367, %375 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %378 = x86.rss.vfmadd231pd %347, %368, %375 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %379 = x86.rss.vfmadd231pd %348, %369, %375 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %380 = x86.dm.vbroadcastsd [%360 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %381 = x86.rss.vfmadd231pd %350, %366, %380 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %382 = x86.rss.vfmadd231pd %351, %367, %380 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %383 = x86.rss.vfmadd231pd %352, %368, %380 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %384 = x86.rss.vfmadd231pd %353, %369, %380 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %385 = x86.dm.vbroadcastsd [%360 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %386 = x86.rss.vfmadd231pd %355, %366, %385 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %387 = x86.rss.vfmadd231pd %356, %367, %385 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %388 = x86.rss.vfmadd231pd %357, %368, %385 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %389 = x86.rss.vfmadd231pd %358, %369, %385 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %390 = x86.dm.vbroadcastsd [%360 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %391 = x86.ri.add %360, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %392 = x86.ri.add %361, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %393 = x86.rss.vfmadd231pd %362, %366, %390 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %394 = x86.rss.vfmadd231pd %363, %367, %390 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %395 = x86.rss.vfmadd231pd %364, %368, %390 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %396 = x86.rss.vfmadd231pd %365, %369, %390 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %397 = x86.dm.vmovupd [%392] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %398 = x86.dm.vmovupd [%392 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %399 = x86.dm.vmovupd [%392 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %400 = x86.dm.vmovupd [%392 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %401 = x86.dm.vbroadcastsd [%391] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %402 = x86.rss.vfmadd231pd %371, %397, %401 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %403 = x86.rss.vfmadd231pd %372, %398, %401 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %404 = x86.rss.vfmadd231pd %373, %399, %401 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %405 = x86.rss.vfmadd231pd %374, %400, %401 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %406 = x86.dm.vbroadcastsd [%391 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %407 = x86.rss.vfmadd231pd %376, %397, %406 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %408 = x86.rss.vfmadd231pd %377, %398, %406 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %409 = x86.rss.vfmadd231pd %378, %399, %406 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %410 = x86.rss.vfmadd231pd %379, %400, %406 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %411 = x86.dm.vbroadcastsd [%391 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %412 = x86.rss.vfmadd231pd %381, %397, %411 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %413 = x86.rss.vfmadd231pd %382, %398, %411 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %414 = x86.rss.vfmadd231pd %383, %399, %411 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %415 = x86.rss.vfmadd231pd %384, %400, %411 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %416 = x86.dm.vbroadcastsd [%391 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %417 = x86.rss.vfmadd231pd %386, %397, %416 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %418 = x86.rss.vfmadd231pd %387, %398, %416 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %419 = x86.rss.vfmadd231pd %388, %399, %416 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %420 = x86.rss.vfmadd231pd %389, %400, %416 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %421 = x86.dm.vbroadcastsd [%391 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %422 = x86.ri.add %391, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %423 = x86.ri.add %392, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %424 = x86.rss.vfmadd231pd %393, %397, %421 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %425 = x86.rss.vfmadd231pd %394, %398, %421 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %426 = x86.rss.vfmadd231pd %395, %399, %421 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %427 = x86.rss.vfmadd231pd %396, %400, %421 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %428 = x86.dm.vmovupd [%423] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %429 = x86.dm.vmovupd [%423 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %430 = x86.dm.vmovupd [%423 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %431 = x86.dm.vmovupd [%423 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %432 = x86.dm.vbroadcastsd [%422] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %433 = x86.rss.vfmadd231pd %402, %428, %432 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %434 = x86.rss.vfmadd231pd %403, %429, %432 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %435 = x86.rss.vfmadd231pd %404, %430, %432 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %436 = x86.rss.vfmadd231pd %405, %431, %432 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %437 = x86.dm.vbroadcastsd [%422 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %438 = x86.rss.vfmadd231pd %407, %428, %437 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %439 = x86.rss.vfmadd231pd %408, %429, %437 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %440 = x86.rss.vfmadd231pd %409, %430, %437 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %441 = x86.rss.vfmadd231pd %410, %431, %437 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %442 = x86.dm.vbroadcastsd [%422 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %443 = x86.rss.vfmadd231pd %412, %428, %442 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %444 = x86.rss.vfmadd231pd %413, %429, %442 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %445 = x86.rss.vfmadd231pd %414, %430, %442 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %446 = x86.rss.vfmadd231pd %415, %431, %442 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %447 = x86.dm.vbroadcastsd [%422 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %448 = x86.rss.vfmadd231pd %417, %428, %447 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %449 = x86.rss.vfmadd231pd %418, %429, %447 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %450 = x86.rss.vfmadd231pd %419, %430, %447 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %451 = x86.rss.vfmadd231pd %420, %431, %447 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %452 = x86.dm.vbroadcastsd [%422 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %453 = x86.ri.add %422, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %454 = x86.ri.add %423, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %455 = x86.rss.vfmadd231pd %424, %428, %452 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %456 = x86.rss.vfmadd231pd %425, %429, %452 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %457 = x86.rss.vfmadd231pd %426, %430, %452 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %458 = x86.rss.vfmadd231pd %427, %431, %452 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %459 = x86.dm.vmovupd [%454] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %460 = x86.dm.vmovupd [%454 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %461 = x86.dm.vmovupd [%454 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %462 = x86.dm.vmovupd [%454 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %463 = x86.dm.vbroadcastsd [%453] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %464 = x86.rss.vfmadd231pd %433, %459, %463 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %465 = x86.rss.vfmadd231pd %434, %460, %463 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %466 = x86.rss.vfmadd231pd %435, %461, %463 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %467 = x86.rss.vfmadd231pd %436, %462, %463 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %468 = x86.dm.vbroadcastsd [%453 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %469 = x86.rss.vfmadd231pd %438, %459, %468 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %470 = x86.rss.vfmadd231pd %439, %460, %468 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %471 = x86.rss.vfmadd231pd %440, %461, %468 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %472 = x86.rss.vfmadd231pd %441, %462, %468 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %473 = x86.dm.vbroadcastsd [%453 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %474 = x86.rss.vfmadd231pd %443, %459, %473 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %475 = x86.rss.vfmadd231pd %444, %460, %473 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %476 = x86.rss.vfmadd231pd %445, %461, %473 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %477 = x86.rss.vfmadd231pd %446, %462, %473 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %478 = x86.dm.vbroadcastsd [%453 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %479 = x86.rss.vfmadd231pd %448, %459, %478 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %480 = x86.rss.vfmadd231pd %449, %460, %478 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %481 = x86.rss.vfmadd231pd %450, %461, %478 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %482 = x86.rss.vfmadd231pd %451, %462, %478 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %483 = x86.dm.vbroadcastsd [%453 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %484 = x86.ri.add %453, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %485 = x86.ri.add %454, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %486 = x86.rss.vfmadd231pd %455, %459, %483 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %487 = x86.rss.vfmadd231pd %456, %460, %483 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %488 = x86.rss.vfmadd231pd %457, %461, %483 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %489 = x86.rss.vfmadd231pd %458, %462, %483 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %490 = x86.dm.vmovupd [%485] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %491 = x86.dm.vmovupd [%485 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %492 = x86.dm.vmovupd [%485 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %493 = x86.dm.vmovupd [%485 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %494 = x86.dm.vbroadcastsd [%484] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %495 = x86.rss.vfmadd231pd %464, %490, %494 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %496 = x86.rss.vfmadd231pd %465, %491, %494 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %497 = x86.rss.vfmadd231pd %466, %492, %494 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %498 = x86.rss.vfmadd231pd %467, %493, %494 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %499 = x86.dm.vbroadcastsd [%484 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %500 = x86.rss.vfmadd231pd %469, %490, %499 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %501 = x86.rss.vfmadd231pd %470, %491, %499 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %502 = x86.rss.vfmadd231pd %471, %492, %499 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %503 = x86.rss.vfmadd231pd %472, %493, %499 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %504 = x86.dm.vbroadcastsd [%484 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %505 = x86.rss.vfmadd231pd %474, %490, %504 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %506 = x86.rss.vfmadd231pd %475, %491, %504 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %507 = x86.rss.vfmadd231pd %476, %492, %504 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %508 = x86.rss.vfmadd231pd %477, %493, %504 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %509 = x86.dm.vbroadcastsd [%484 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %510 = x86.rss.vfmadd231pd %479, %490, %509 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %511 = x86.rss.vfmadd231pd %480, %491, %509 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %512 = x86.rss.vfmadd231pd %481, %492, %509 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %513 = x86.rss.vfmadd231pd %482, %493, %509 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %514 = x86.dm.vbroadcastsd [%484 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %515 = x86.ri.add %484, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %516 = x86.ri.add %485, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %517 = x86.rss.vfmadd231pd %486, %490, %514 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %518 = x86.rss.vfmadd231pd %487, %491, %514 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %519 = x86.rss.vfmadd231pd %488, %492, %514 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %520 = x86.rss.vfmadd231pd %489, %493, %514 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %521 = x86.dm.vmovupd [%516] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %522 = x86.dm.vmovupd [%516 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %523 = x86.dm.vmovupd [%516 + 128] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm3>
// CHECK-NEXT:        %524 = x86.dm.vmovupd [%516 + 192] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm4>
// CHECK-NEXT:        %525 = x86.dm.vbroadcastsd [%515] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %526 = x86.rss.vfmadd231pd %495, %521, %525 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %527 = x86.rss.vfmadd231pd %496, %522, %525 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %528 = x86.rss.vfmadd231pd %497, %523, %525 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %529 = x86.rss.vfmadd231pd %498, %524, %525 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %530 = x86.dm.vbroadcastsd [%515 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %531 = x86.rss.vfmadd231pd %500, %521, %530 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %532 = x86.rss.vfmadd231pd %501, %522, %530 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %533 = x86.rss.vfmadd231pd %502, %523, %530 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %534 = x86.rss.vfmadd231pd %503, %524, %530 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %535 = x86.dm.vbroadcastsd [%515 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %536 = x86.rss.vfmadd231pd %505, %521, %535 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %537 = x86.rss.vfmadd231pd %506, %522, %535 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %538 = x86.rss.vfmadd231pd %507, %523, %535 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %539 = x86.rss.vfmadd231pd %508, %524, %535 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %540 = x86.dm.vbroadcastsd [%515 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %541 = x86.rss.vfmadd231pd %510, %521, %540 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %542 = x86.rss.vfmadd231pd %511, %522, %540 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %543 = x86.rss.vfmadd231pd %512, %523, %540 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %544 = x86.rss.vfmadd231pd %513, %524, %540 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %545 = x86.dm.vbroadcastsd [%515 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %546 = x86.ri.add %515, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %547 = x86.ri.add %516, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %548 = x86.rss.vfmadd231pd %517, %521, %545 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %549 = x86.rss.vfmadd231pd %518, %522, %545 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %550 = x86.rss.vfmadd231pd %519, %523, %545 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %551 = x86.rss.vfmadd231pd %520, %524, %545 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %552 = x86.ri.sub %546, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovupd [%33], %526 : (!x86.reg64<rdx>, !x86.avx512reg<zmm12>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 64], %527 : (!x86.reg64<rdx>, !x86.avx512reg<zmm13>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 128], %528 : (!x86.reg64<rdx>, !x86.avx512reg<zmm14>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 192], %529 : (!x86.reg64<rdx>, !x86.avx512reg<zmm15>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 272], %531 : (!x86.reg64<rdx>, !x86.avx512reg<zmm16>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 336], %532 : (!x86.reg64<rdx>, !x86.avx512reg<zmm17>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 400], %533 : (!x86.reg64<rdx>, !x86.avx512reg<zmm18>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 464], %534 : (!x86.reg64<rdx>, !x86.avx512reg<zmm19>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 544], %536 : (!x86.reg64<rdx>, !x86.avx512reg<zmm20>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 608], %537 : (!x86.reg64<rdx>, !x86.avx512reg<zmm21>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 672], %538 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 736], %539 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 816], %541 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 880], %542 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 944], %543 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 1008], %544 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 1088], %548 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 1152], %549 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 1216], %550 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovupd [%33 + 1280], %551 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %553 = x86.ri.add %33, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %554 = x86.ri.sub %547, 4096 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %554, %552, %553, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %555 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %556 = x86.ks.kmovb %555 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %557 = x86.di.mov 32 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %558, %559, %560, %561, %562, %563, %564 = x86_scf.for %565 : !x86.reg64<r10>  = %557 to 34 : si32 step 2 : si32 iter_args(%566 = %25, %567 = %26, %568 = %27, %569 = %28, %570 = %29, %571 = %556) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %572 = x86.dmk.vmovupd[%568], %571 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %573 = x86.dmk.vmovupd[%568 + 272], %571 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %574 = x86.dmk.vmovupd[%568 + 544], %571 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %575 = x86.dmk.vmovupd[%568 + 816], %571 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %576 = x86.dmk.vmovupd[%568 + 1088], %571 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %577 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-NEXT:        %578 = x86.dss.vpxord %577, %577 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %579 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-NEXT:        %580 = x86.dss.vpxord %579, %579 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %581 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-NEXT:        %582 = x86.dss.vpxord %581, %581 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %583 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-NEXT:        %584 = x86.dss.vpxord %583, %583 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %585 = x86.get_avx_register : !x86.avx512reg<zmm26>
// CHECK-NEXT:        %586 = x86.dss.vpxord %585, %585 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %587 = x86.get_avx_register : !x86.avx512reg<zmm17>
// CHECK-NEXT:        %588 = x86.dss.vpxord %587, %587 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm17>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %589 = x86.get_avx_register : !x86.avx512reg<zmm18>
// CHECK-NEXT:        %590 = x86.dss.vpxord %589, %589 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm18>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %591 = x86.get_avx_register : !x86.avx512reg<zmm19>
// CHECK-NEXT:        %592 = x86.dss.vpxord %591, %591 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm19>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %593 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-NEXT:        %594 = x86.dss.vpxord %593, %593 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %595 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-NEXT:        %596 = x86.dss.vpxord %595, %595 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %597 = x86.get_avx_register : !x86.avx512reg<zmm12>
// CHECK-NEXT:        %598 = x86.dss.vpxord %597, %597 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm12>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %599 = x86.get_avx_register : !x86.avx512reg<zmm13>
// CHECK-NEXT:        %600 = x86.dss.vpxord %599, %599 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm13>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %601 = x86.get_avx_register : !x86.avx512reg<zmm14>
// CHECK-NEXT:        %602 = x86.dss.vpxord %601, %601 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm14>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %603 = x86.get_avx_register : !x86.avx512reg<zmm15>
// CHECK-NEXT:        %604 = x86.dss.vpxord %603, %603 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm15>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %605 = x86.get_avx_register : !x86.avx512reg<zmm16>
// CHECK-NEXT:        %606 = x86.dss.vpxord %605, %605 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm16>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %607 = x86.dmk.vmovupd[%566], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %608 = x86.dmk.vmovupd[%566 + 272], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %609 = x86.rsm.vfmadd231pd %572, %607, [%567] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %610 = x86.rsm.vfmadd231pd %573, %607, [%567 + 128] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %611 = x86.rsm.vfmadd231pd %574, %607, [%567 + 256] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %612 = x86.rsm.vfmadd231pd %575, %607, [%567 + 384] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %613 = x86.rsm.vfmadd231pd %576, %607, [%567 + 512] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %614 = x86.dmk.vmovupd[%566 + 544], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %615 = x86.rsm.vfmadd231pd %578, %608, [%567 + 8] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %616 = x86.rsm.vfmadd231pd %580, %608, [%567 + 136] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %617 = x86.rsm.vfmadd231pd %582, %608, [%567 + 264] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %618 = x86.rsm.vfmadd231pd %584, %608, [%567 + 392] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %619 = x86.rsm.vfmadd231pd %586, %608, [%567 + 520] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %620 = x86.dmk.vmovupd[%566 + 816], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %621 = x86.rsm.vfmadd231pd %588, %614, [%567 + 16] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %622 = x86.rsm.vfmadd231pd %590, %614, [%567 + 144] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %623 = x86.rsm.vfmadd231pd %592, %614, [%567 + 272] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %624 = x86.rsm.vfmadd231pd %594, %614, [%567 + 400] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %625 = x86.rsm.vfmadd231pd %596, %614, [%567 + 528] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %626 = x86.dmk.vmovupd[%566 + 1088], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %627 = x86.rsm.vfmadd231pd %598, %620, [%567 + 24] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %628 = x86.rsm.vfmadd231pd %600, %620, [%567 + 152] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %629 = x86.rsm.vfmadd231pd %602, %620, [%567 + 280] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %630 = x86.rsm.vfmadd231pd %604, %620, [%567 + 408] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %631 = x86.rsm.vfmadd231pd %606, %620, [%567 + 536] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %632 = x86.dmk.vmovupd[%566 + 1360], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %633 = x86.rsm.vfmadd231pd %609, %626, [%567 + 32] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %634 = x86.rsm.vfmadd231pd %610, %626, [%567 + 160] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %635 = x86.rsm.vfmadd231pd %611, %626, [%567 + 288] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %636 = x86.rsm.vfmadd231pd %612, %626, [%567 + 416] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %637 = x86.rsm.vfmadd231pd %613, %626, [%567 + 544] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %638 = x86.dmk.vmovupd[%566 + 1632], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %639 = x86.rsm.vfmadd231pd %615, %632, [%567 + 40] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %640 = x86.rsm.vfmadd231pd %616, %632, [%567 + 168] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %641 = x86.rsm.vfmadd231pd %617, %632, [%567 + 296] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %642 = x86.rsm.vfmadd231pd %618, %632, [%567 + 424] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %643 = x86.rsm.vfmadd231pd %619, %632, [%567 + 552] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %644 = x86.dmk.vmovupd[%566 + 1904], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %645 = x86.rsm.vfmadd231pd %621, %638, [%567 + 48] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %646 = x86.rsm.vfmadd231pd %622, %638, [%567 + 176] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %647 = x86.rsm.vfmadd231pd %623, %638, [%567 + 304] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %648 = x86.rsm.vfmadd231pd %624, %638, [%567 + 432] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %649 = x86.rsm.vfmadd231pd %625, %638, [%567 + 560] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %650 = x86.dmk.vmovupd[%566 + 2176], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %651 = x86.rsm.vfmadd231pd %627, %644, [%567 + 56] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %652 = x86.rsm.vfmadd231pd %628, %644, [%567 + 184] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %653 = x86.rsm.vfmadd231pd %629, %644, [%567 + 312] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %654 = x86.rsm.vfmadd231pd %630, %644, [%567 + 440] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %655 = x86.rsm.vfmadd231pd %631, %644, [%567 + 568] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %656 = x86.dmk.vmovupd[%566 + 2448], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %657 = x86.rsm.vfmadd231pd %633, %650, [%567 + 64] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %658 = x86.rsm.vfmadd231pd %634, %650, [%567 + 192] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %659 = x86.rsm.vfmadd231pd %635, %650, [%567 + 320] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %660 = x86.rsm.vfmadd231pd %636, %650, [%567 + 448] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %661 = x86.rsm.vfmadd231pd %637, %650, [%567 + 576] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %662 = x86.dmk.vmovupd[%566 + 2720], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %663 = x86.rsm.vfmadd231pd %639, %656, [%567 + 72] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %664 = x86.rsm.vfmadd231pd %640, %656, [%567 + 200] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %665 = x86.rsm.vfmadd231pd %641, %656, [%567 + 328] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %666 = x86.rsm.vfmadd231pd %642, %656, [%567 + 456] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %667 = x86.rsm.vfmadd231pd %643, %656, [%567 + 584] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %668 = x86.dmk.vmovupd[%566 + 2992], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %669 = x86.rsm.vfmadd231pd %645, %662, [%567 + 80] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %670 = x86.rsm.vfmadd231pd %646, %662, [%567 + 208] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %671 = x86.rsm.vfmadd231pd %647, %662, [%567 + 336] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %672 = x86.rsm.vfmadd231pd %648, %662, [%567 + 464] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %673 = x86.rsm.vfmadd231pd %649, %662, [%567 + 592] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %674 = x86.dmk.vmovupd[%566 + 3264], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %675 = x86.rsm.vfmadd231pd %651, %668, [%567 + 88] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %676 = x86.rsm.vfmadd231pd %652, %668, [%567 + 216] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %677 = x86.rsm.vfmadd231pd %653, %668, [%567 + 344] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %678 = x86.rsm.vfmadd231pd %654, %668, [%567 + 472] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %679 = x86.rsm.vfmadd231pd %655, %668, [%567 + 600] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %680 = x86.dmk.vmovupd[%566 + 3536], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %681 = x86.rsm.vfmadd231pd %657, %674, [%567 + 96] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %682 = x86.rsm.vfmadd231pd %658, %674, [%567 + 224] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %683 = x86.rsm.vfmadd231pd %659, %674, [%567 + 352] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %684 = x86.rsm.vfmadd231pd %660, %674, [%567 + 480] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %685 = x86.rsm.vfmadd231pd %661, %674, [%567 + 608] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %686 = x86.dmk.vmovupd[%566 + 3808], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %687 = x86.rsm.vfmadd231pd %663, %680, [%567 + 104] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-NEXT:        %688 = x86.rsm.vfmadd231pd %664, %680, [%567 + 232] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-NEXT:        %689 = x86.rsm.vfmadd231pd %665, %680, [%567 + 360] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-NEXT:        %690 = x86.rsm.vfmadd231pd %666, %680, [%567 + 488] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-NEXT:        %691 = x86.rsm.vfmadd231pd %667, %680, [%567 + 616] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %692 = x86.dmk.vmovupd[%566 + 4080], %571 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %693 = x86.rsm.vfmadd231pd %669, %686, [%567 + 112] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-NEXT:        %694 = x86.rsm.vfmadd231pd %670, %686, [%567 + 240] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-NEXT:        %695 = x86.rsm.vfmadd231pd %671, %686, [%567 + 368] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-NEXT:        %696 = x86.rsm.vfmadd231pd %672, %686, [%567 + 496] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-NEXT:        %697 = x86.rsm.vfmadd231pd %673, %686, [%567 + 624] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-NEXT:        %698 = x86.ri.add %566, 4352 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %699 = x86.rsm.vfmadd231pd %675, %692, [%567 + 120] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-NEXT:        %700 = x86.rsm.vfmadd231pd %676, %692, [%567 + 248] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-NEXT:        %701 = x86.rsm.vfmadd231pd %677, %692, [%567 + 376] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-NEXT:        %702 = x86.rsm.vfmadd231pd %678, %692, [%567 + 504] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-NEXT:        %703 = x86.rsm.vfmadd231pd %679, %692, [%567 + 632] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-NEXT:        %704 = x86.ri.add %567, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %705 = x86.dss.vaddpd %687, %681 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %706 = x86.dss.vaddpd %688, %682 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %707 = x86.dss.vaddpd %689, %683 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %708 = x86.dss.vaddpd %690, %684 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %709 = x86.dss.vaddpd %691, %685 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %710 = x86.dss.vaddpd %693, %705 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %711 = x86.dss.vaddpd %694, %706 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %712 = x86.dss.vaddpd %695, %707 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %713 = x86.dss.vaddpd %696, %708 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %714 = x86.dss.vaddpd %697, %709 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %715 = x86.dss.vaddpd %699, %710 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %716 = x86.dss.vaddpd %700, %711 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %717 = x86.dss.vaddpd %701, %712 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %718 = x86.dss.vaddpd %702, %713 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %719 = x86.dss.vaddpd %703, %714 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %720 = x86.ri.sub %704, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovupd[%568], %715, %571 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%568 + 272], %716, %571 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%568 + 544], %717, %571 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%568 + 816], %718, %571 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%568 + 1088], %719, %571 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %721 = x86.ri.add %568, 16 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %722 = x86.ri.sub %698, 4336 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %722, %720, %721, %569, %570, %571 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %723 = x86.ri.add %561, 1088 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %724 = x86.ri.add %560, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %725 = x86.ri.sub %559, 272 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %725, %724, %723, %562, %563 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %726 = x86.ds.mov %15 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %727, %728 = x86.d.pop %726 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-NEXT:      sub rsp, 192
// CHECK-REGALLOC-NEXT:      mov r10, -64
// CHECK-REGALLOC-NEXT:      and rsp, r10
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 32
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm15, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovupd zmm16, [rdx+272]
// CHECK-REGALLOC-NEXT:      vmovupd zmm17, [rdx+336]
// CHECK-REGALLOC-NEXT:      vmovupd zmm18, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovupd zmm19, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdx+544]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdx+608]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdx+672]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdx+736]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdx+816]
// CHECK-REGALLOC-NEXT:      vmovupd zmm25, [rdx+880]
// CHECK-REGALLOC-NEXT:      vmovupd zmm26, [rdx+944]
// CHECK-REGALLOC-NEXT:      vmovupd zmm27, [rdx+1008]
// CHECK-REGALLOC-NEXT:      vmovupd zmm28, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovupd zmm29, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovupd zmm30, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovupd zmm31, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovupd [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+192], zmm15
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+272], zmm16
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+336], zmm17
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+400], zmm18
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+464], zmm19
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+544], zmm20
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+608], zmm21
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+672], zmm22
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+736], zmm23
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+816], zmm24
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+880], zmm25
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+944], zmm26
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1008], zmm27
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1088], zmm28
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1152], zmm29
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1216], zmm30
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1280], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 4096
// CHECK-REGALLOC-NEXT:      cmp rcx, 32
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovb k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 32
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 2
// CHECK-REGALLOC-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+272]
// CHECK-REGALLOC-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+544]
// CHECK-REGALLOC-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+816]
// CHECK-REGALLOC-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+1088]
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
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+272]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+128]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+256]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+384]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+512]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+544]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+8]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+136]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+264]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+392]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+520]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+816]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+16]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+144]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+272]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+400]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+528]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+1088]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+24]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+152]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+280]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+408]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+536]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+1360]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+32]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+160]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+288]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+416]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+544]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+1632]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+40]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+168]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+296]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+424]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+552]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+1904]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+48]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+176]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+304]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+432]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+560]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+2176]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+56]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+184]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+312]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+440]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+568]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+2448]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+64]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+192]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+320]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+448]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+576]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+2720]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+72]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+200]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+328]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+456]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+584]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+2992]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+80]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+208]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+336]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+464]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+592]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+3264]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+88]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+216]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+344]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+472]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+600]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+3536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+96]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+224]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+352]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+480]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+608]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+3808]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+104]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+232]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+360]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+488]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+616]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+4080]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+112]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+240]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+368]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+496]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+624]{1to8}
// CHECK-REGALLOC-NEXT:      add rdi, 4352
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+120]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+248]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+376]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+504]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+632]{1to8}
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm22, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm23, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm24, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm25, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm26, zmm31
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm17, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm18, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm19, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm20, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm21, zmm31
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm12, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm13, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm14, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm15, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm16, zmm31
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+272] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+544] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+816] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1088] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 16
// CHECK-REGALLOC-NEXT:      sub rdi, 4336
// CHECK-REGALLOC-NEXT:      cmp rcx, 34
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 1088
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      sub rdi, 272
// CHECK-REGALLOC-NEXT:      cmp rax, 5
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
