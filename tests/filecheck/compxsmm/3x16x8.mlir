// RUN: compxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 3 : si32 step 3 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 16 : si32 step 16 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovapd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %37 = x86.dm.vmovapd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %38 = x86.dm.vmovapd [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %39 = x86.dm.vmovapd [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %40 = x86.dm.vmovapd [%33 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %41 = x86.dm.vmovapd [%33 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %42 = x86.dm.vmovapd [%31] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %43 = x86.dm.vmovapd [%31 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %44 = x86.dm.vbroadcastsd [%32] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %45 = x86.rss.vfmadd231pd %36, %42, %44 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %46 = x86.rss.vfmadd231pd %37, %43, %44 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %47 = x86.dm.vbroadcastsd [%32 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %48 = x86.rss.vfmadd231pd %38, %42, %47 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %49 = x86.rss.vfmadd231pd %39, %43, %47 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %50 = x86.dm.vbroadcastsd [%32 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %51 = x86.ri.add %32, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %52 = x86.ri.add %31, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %53 = x86.rss.vfmadd231pd %40, %42, %50 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %54 = x86.rss.vfmadd231pd %41, %43, %50 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %55 = x86.dm.vmovapd [%52] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %56 = x86.dm.vmovapd [%52 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %57 = x86.dm.vbroadcastsd [%51] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %58 = x86.rss.vfmadd231pd %45, %55, %57 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %59 = x86.rss.vfmadd231pd %46, %56, %57 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %60 = x86.dm.vbroadcastsd [%51 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %61 = x86.rss.vfmadd231pd %48, %55, %60 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %62 = x86.rss.vfmadd231pd %49, %56, %60 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %63 = x86.dm.vbroadcastsd [%51 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %64 = x86.ri.add %51, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %65 = x86.ri.add %52, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %66 = x86.rss.vfmadd231pd %53, %55, %63 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %67 = x86.rss.vfmadd231pd %54, %56, %63 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %68 = x86.dm.vmovapd [%65] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %69 = x86.dm.vmovapd [%65 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %70 = x86.dm.vbroadcastsd [%64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %71 = x86.rss.vfmadd231pd %58, %68, %70 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %72 = x86.rss.vfmadd231pd %59, %69, %70 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %73 = x86.dm.vbroadcastsd [%64 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %74 = x86.rss.vfmadd231pd %61, %68, %73 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %75 = x86.rss.vfmadd231pd %62, %69, %73 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %76 = x86.dm.vbroadcastsd [%64 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %77 = x86.ri.add %64, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %78 = x86.ri.add %65, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %79 = x86.rss.vfmadd231pd %66, %68, %76 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %80 = x86.rss.vfmadd231pd %67, %69, %76 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %81 = x86.dm.vmovapd [%78] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %82 = x86.dm.vmovapd [%78 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %83 = x86.dm.vbroadcastsd [%77] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %84 = x86.rss.vfmadd231pd %71, %81, %83 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %85 = x86.rss.vfmadd231pd %72, %82, %83 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %86 = x86.dm.vbroadcastsd [%77 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %87 = x86.rss.vfmadd231pd %74, %81, %86 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %88 = x86.rss.vfmadd231pd %75, %82, %86 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %89 = x86.dm.vbroadcastsd [%77 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %90 = x86.ri.add %77, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %91 = x86.ri.add %78, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %92 = x86.rss.vfmadd231pd %79, %81, %89 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %93 = x86.rss.vfmadd231pd %80, %82, %89 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %94 = x86.dm.vmovapd [%91] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %95 = x86.dm.vmovapd [%91 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %96 = x86.dm.vbroadcastsd [%90] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %97 = x86.rss.vfmadd231pd %84, %94, %96 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %98 = x86.rss.vfmadd231pd %85, %95, %96 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %99 = x86.dm.vbroadcastsd [%90 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %100 = x86.rss.vfmadd231pd %87, %94, %99 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %101 = x86.rss.vfmadd231pd %88, %95, %99 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %102 = x86.dm.vbroadcastsd [%90 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %103 = x86.ri.add %90, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %104 = x86.ri.add %91, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %105 = x86.rss.vfmadd231pd %92, %94, %102 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %106 = x86.rss.vfmadd231pd %93, %95, %102 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %107 = x86.dm.vmovapd [%104] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %108 = x86.dm.vmovapd [%104 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %109 = x86.dm.vbroadcastsd [%103] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %110 = x86.rss.vfmadd231pd %97, %107, %109 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %111 = x86.rss.vfmadd231pd %98, %108, %109 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %112 = x86.dm.vbroadcastsd [%103 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %113 = x86.rss.vfmadd231pd %100, %107, %112 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %114 = x86.rss.vfmadd231pd %101, %108, %112 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %115 = x86.dm.vbroadcastsd [%103 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %116 = x86.ri.add %103, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %117 = x86.ri.add %104, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %118 = x86.rss.vfmadd231pd %105, %107, %115 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %119 = x86.rss.vfmadd231pd %106, %108, %115 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %120 = x86.dm.vmovapd [%117] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %121 = x86.dm.vmovapd [%117 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %122 = x86.dm.vbroadcastsd [%116] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %123 = x86.rss.vfmadd231pd %110, %120, %122 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %124 = x86.rss.vfmadd231pd %111, %121, %122 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %125 = x86.dm.vbroadcastsd [%116 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %126 = x86.rss.vfmadd231pd %113, %120, %125 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %127 = x86.rss.vfmadd231pd %114, %121, %125 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %128 = x86.dm.vbroadcastsd [%116 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %129 = x86.ri.add %116, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %130 = x86.ri.add %117, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %131 = x86.rss.vfmadd231pd %118, %120, %128 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %132 = x86.rss.vfmadd231pd %119, %121, %128 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %133 = x86.dm.vmovapd [%130] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %134 = x86.dm.vmovapd [%130 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %135 = x86.dm.vbroadcastsd [%129] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %136 = x86.rss.vfmadd231pd %123, %133, %135 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %137 = x86.rss.vfmadd231pd %124, %134, %135 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %138 = x86.dm.vbroadcastsd [%129 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %139 = x86.rss.vfmadd231pd %126, %133, %138 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %140 = x86.rss.vfmadd231pd %127, %134, %138 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %141 = x86.dm.vbroadcastsd [%129 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %142 = x86.ri.add %129, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %143 = x86.ri.add %130, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %144 = x86.rss.vfmadd231pd %131, %133, %141 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %145 = x86.rss.vfmadd231pd %132, %134, %141 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %146 = x86.ri.sub %142, 64 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%33], %136 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%33 + 64], %137 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%33 + 128], %139 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%33 + 192], %140 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%33 + 256], %144 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%33 + 320], %145 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %147 = x86.ri.add %33, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %148 = x86.ri.sub %143, 896 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %148, %146, %147, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %149 = x86.ri.add %27, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %150 = x86.ri.add %26, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %151 = x86.ri.sub %25, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %151, %150, %149, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %152 = x86.ds.mov %15 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %153, %154 = x86.d.pop %152 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rax, 3
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      sub rsi, 64
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm27
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm28
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm29
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 896
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      add rsi, 192
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp rax, 3
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
