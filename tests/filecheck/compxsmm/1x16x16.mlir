// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,x86-allocate-registers,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 1 : si32 step 1 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 16 : si32 step 16 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovapd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %37 = x86.dm.vmovapd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %38 = x86.dm.vmovapd [%31] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %39 = x86.dm.vmovapd [%31 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %40 = x86.dm.vbroadcastsd [%32] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %41 = x86.ri.add %32, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %42 = x86.ri.add %31, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %43 = x86.rss.vfmadd231pd %36, %38, %40 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %44 = x86.rss.vfmadd231pd %37, %39, %40 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %45 = x86.dm.vmovapd [%42] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %46 = x86.dm.vmovapd [%42 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %47 = x86.dm.vbroadcastsd [%41] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %48 = x86.ri.add %41, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %49 = x86.ri.add %42, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %50 = x86.rss.vfmadd231pd %43, %45, %47 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %51 = x86.rss.vfmadd231pd %44, %46, %47 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %52 = x86.dm.vmovapd [%49] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %53 = x86.dm.vmovapd [%49 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %54 = x86.dm.vbroadcastsd [%48] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %55 = x86.ri.add %48, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %56 = x86.ri.add %49, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %57 = x86.rss.vfmadd231pd %50, %52, %54 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %58 = x86.rss.vfmadd231pd %51, %53, %54 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %59 = x86.dm.vmovapd [%56] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %60 = x86.dm.vmovapd [%56 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %61 = x86.dm.vbroadcastsd [%55] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %62 = x86.ri.add %55, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %63 = x86.ri.add %56, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %64 = x86.rss.vfmadd231pd %57, %59, %61 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %65 = x86.rss.vfmadd231pd %58, %60, %61 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %66 = x86.dm.vmovapd [%63] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %67 = x86.dm.vmovapd [%63 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %68 = x86.dm.vbroadcastsd [%62] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %69 = x86.ri.add %62, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %70 = x86.ri.add %63, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %71 = x86.rss.vfmadd231pd %64, %66, %68 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %72 = x86.rss.vfmadd231pd %65, %67, %68 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %73 = x86.dm.vmovapd [%70] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %74 = x86.dm.vmovapd [%70 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %75 = x86.dm.vbroadcastsd [%69] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %76 = x86.ri.add %69, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %77 = x86.ri.add %70, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %78 = x86.rss.vfmadd231pd %71, %73, %75 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %79 = x86.rss.vfmadd231pd %72, %74, %75 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %80 = x86.dm.vmovapd [%77] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %81 = x86.dm.vmovapd [%77 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %82 = x86.dm.vbroadcastsd [%76] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %83 = x86.ri.add %76, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %84 = x86.ri.add %77, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %85 = x86.rss.vfmadd231pd %78, %80, %82 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %86 = x86.rss.vfmadd231pd %79, %81, %82 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %87 = x86.dm.vmovapd [%84] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %88 = x86.dm.vmovapd [%84 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %89 = x86.dm.vbroadcastsd [%83] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %90 = x86.ri.add %83, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %91 = x86.ri.add %84, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %92 = x86.rss.vfmadd231pd %85, %87, %89 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %93 = x86.rss.vfmadd231pd %86, %88, %89 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %94 = x86.dm.vmovapd [%91] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %95 = x86.dm.vmovapd [%91 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %96 = x86.dm.vbroadcastsd [%90] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %97 = x86.ri.add %90, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %98 = x86.ri.add %91, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %99 = x86.rss.vfmadd231pd %92, %94, %96 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %100 = x86.rss.vfmadd231pd %93, %95, %96 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %101 = x86.dm.vmovapd [%98] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %102 = x86.dm.vmovapd [%98 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %103 = x86.dm.vbroadcastsd [%97] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %104 = x86.ri.add %97, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %105 = x86.ri.add %98, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %106 = x86.rss.vfmadd231pd %99, %101, %103 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %107 = x86.rss.vfmadd231pd %100, %102, %103 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %108 = x86.dm.vmovapd [%105] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %109 = x86.dm.vmovapd [%105 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %110 = x86.dm.vbroadcastsd [%104] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %111 = x86.ri.add %104, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %112 = x86.ri.add %105, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %113 = x86.rss.vfmadd231pd %106, %108, %110 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %114 = x86.rss.vfmadd231pd %107, %109, %110 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %115 = x86.dm.vmovapd [%112] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %116 = x86.dm.vmovapd [%112 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %117 = x86.dm.vbroadcastsd [%111] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %118 = x86.ri.add %111, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %119 = x86.ri.add %112, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %120 = x86.rss.vfmadd231pd %113, %115, %117 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %121 = x86.rss.vfmadd231pd %114, %116, %117 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %122 = x86.dm.vmovapd [%119] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %123 = x86.dm.vmovapd [%119 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %124 = x86.dm.vbroadcastsd [%118] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %125 = x86.ri.add %118, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %126 = x86.ri.add %119, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %127 = x86.rss.vfmadd231pd %120, %122, %124 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %128 = x86.rss.vfmadd231pd %121, %123, %124 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %129 = x86.dm.vmovapd [%126] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %130 = x86.dm.vmovapd [%126 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %131 = x86.dm.vbroadcastsd [%125] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %132 = x86.ri.add %125, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %133 = x86.ri.add %126, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %134 = x86.rss.vfmadd231pd %127, %129, %131 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %135 = x86.rss.vfmadd231pd %128, %130, %131 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %136 = x86.dm.vmovapd [%133] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %137 = x86.dm.vmovapd [%133 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %138 = x86.dm.vbroadcastsd [%132] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %139 = x86.ri.add %132, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %140 = x86.ri.add %133, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %141 = x86.rss.vfmadd231pd %134, %136, %138 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %142 = x86.rss.vfmadd231pd %135, %137, %138 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %143 = x86.dm.vmovapd [%140] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %144 = x86.dm.vmovapd [%140 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %145 = x86.dm.vbroadcastsd [%139] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %146 = x86.ri.add %139, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %147 = x86.ri.add %140, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %148 = x86.rss.vfmadd231pd %141, %143, %145 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %149 = x86.rss.vfmadd231pd %142, %144, %145 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %150 = x86.ri.sub %146, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%33], %148 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%33 + 64], %149 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %151 = x86.ri.add %33, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %152 = x86.ri.sub %147, 1920 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %152, %150, %151, %34, %35 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %153 = x86.ri.add %27, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %154 = x86.ri.add %26, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %155 = x86.ri.sub %25, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %155, %154, %153, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %156 = x86.ds.mov %15 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %157, %158 = x86.d.pop %156 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      mov [[STACK_ALIGN:\S+]], -64
// CHECK-REGALLOC-NEXT:      and rsp, [[STACK_ALIGN]]
// CHECK-REGALLOC-NEXT:      mov [[N:\S+]], 0
// CHECK-REGALLOC-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-REGALLOC-NEXT:      add [[N]], 1
// CHECK-REGALLOC-NEXT:      mov [[M:\S+]], 0
// CHECK-REGALLOC-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-REGALLOC-NEXT:      add [[M]], 16
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_ACC0:\S+]], [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_ACC1:\S+]], [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0:\S+]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1:\S+]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X2:\S+]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X0]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X1]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X0]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X1]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X2]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X1]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X2]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X0]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X2]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X0]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X1]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X0]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X1]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X2]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X1]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X2]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X0]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X2]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X0]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X1]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X0]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X1]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X2]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X1]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X2]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X0]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X2]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X0]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X1]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X0]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X1]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X2]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X1]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X2]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X0]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X2]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X0]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X1]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X0]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X1]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X2]], [[VEC_X0]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X2]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X1]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X2]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X0]], [[VEC_X1]]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X0]], [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd [[VEC_X1]], [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd [[VEC_X2]], [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC0]], [[VEC_X0]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      vfmadd231pd [[VEC_ACC1]], [[VEC_X1]], [[VEC_X2]]
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], [[VEC_ACC0]]
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], [[VEC_ACC1]]
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-NEXT:      cmp [[M]], 16
// CHECK-REGALLOC-NEXT:      jl [[SCF_M_BODY]]
// CHECK-REGALLOC-NEXT:      add rdx, 0
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp [[N]], 1
// CHECK-REGALLOC-NEXT:      jl [[SCF_N_BODY]]
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
