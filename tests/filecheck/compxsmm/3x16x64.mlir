// RUN: compxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    x86.fallthrough ^bb0(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-NEXT:  ^bb0(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-NEXT:    x86.label "l33"
// CHECK-NEXT:    %17 = x86.ri.add %16, 3 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:    %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %19, %20, %21, %22, %23, %24 = x86_scf.for %25 : !x86.reg64<r10>  = %18 to 16 : si32 step 16 : si32 iter_args(%26 = %11, %27 = %12, %28 = %13, %29 = %14, %30 = %15, %31 = %17) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r11>) {
// CHECK-NEXT:      %32 = x86.dm.vmovapd [%28] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %33 = x86.dm.vmovapd [%28 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %34 = x86.dm.vmovapd [%28 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %35 = x86.dm.vmovapd [%28 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %36 = x86.dm.vmovapd [%28 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %37 = x86.dm.vmovapd [%28 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %38 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50 = x86_scf.for %51 : !x86.reg64<r12>  = %38 to 64 : si32 step 4 : si32 iter_args(%52 = %26, %53 = %27, %54 = %28, %55 = %29, %56 = %30, %57 = %31, %58 = %32, %59 = %33, %60 = %34, %61 = %35, %62 = %36, %63 = %37) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r11>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:        %64 = x86.dm.vmovapd [%52] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %65 = x86.dm.vmovapd [%52 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %66 = x86.dm.vbroadcastsd [%53] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %67 = x86.rss.vfmadd231pd %58, %64, %66 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %68 = x86.rss.vfmadd231pd %59, %65, %66 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %69 = x86.dm.vbroadcastsd [%53 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %70 = x86.rss.vfmadd231pd %60, %64, %69 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %71 = x86.rss.vfmadd231pd %61, %65, %69 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %72 = x86.dm.vbroadcastsd [%53 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %73 = x86.ri.add %53, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %74 = x86.ri.add %52, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %75 = x86.rss.vfmadd231pd %62, %64, %72 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %76 = x86.rss.vfmadd231pd %63, %65, %72 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %77 = x86.dm.vmovapd [%74] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %78 = x86.dm.vmovapd [%74 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %79 = x86.dm.vbroadcastsd [%73] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %80 = x86.rss.vfmadd231pd %67, %77, %79 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %81 = x86.rss.vfmadd231pd %68, %78, %79 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %82 = x86.dm.vbroadcastsd [%73 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %83 = x86.rss.vfmadd231pd %70, %77, %82 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %84 = x86.rss.vfmadd231pd %71, %78, %82 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %85 = x86.dm.vbroadcastsd [%73 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %86 = x86.ri.add %73, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %87 = x86.ri.add %74, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %88 = x86.rss.vfmadd231pd %75, %77, %85 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %89 = x86.rss.vfmadd231pd %76, %78, %85 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %90 = x86.dm.vmovapd [%87] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %91 = x86.dm.vmovapd [%87 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %92 = x86.dm.vbroadcastsd [%86] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %93 = x86.rss.vfmadd231pd %80, %90, %92 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %94 = x86.rss.vfmadd231pd %81, %91, %92 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %95 = x86.dm.vbroadcastsd [%86 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %96 = x86.rss.vfmadd231pd %83, %90, %95 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %97 = x86.rss.vfmadd231pd %84, %91, %95 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %98 = x86.dm.vbroadcastsd [%86 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %99 = x86.ri.add %86, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %100 = x86.ri.add %87, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %101 = x86.rss.vfmadd231pd %88, %90, %98 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %102 = x86.rss.vfmadd231pd %89, %91, %98 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %103 = x86.dm.vmovapd [%100] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:        %104 = x86.dm.vmovapd [%100 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:        %105 = x86.dm.vbroadcastsd [%99] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %106 = x86.rss.vfmadd231pd %93, %103, %105 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %107 = x86.rss.vfmadd231pd %94, %104, %105 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %108 = x86.dm.vbroadcastsd [%99 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %109 = x86.rss.vfmadd231pd %96, %103, %108 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %110 = x86.rss.vfmadd231pd %97, %104, %108 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %111 = x86.dm.vbroadcastsd [%99 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:        %112 = x86.ri.add %99, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        %113 = x86.ri.add %100, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        %114 = x86.rss.vfmadd231pd %101, %103, %111 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %115 = x86.rss.vfmadd231pd %102, %104, %111 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        x86_scf.yield %113, %112, %54, %55, %56, %57, %106, %107, %109, %110, %114, %115 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r11>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:      }
// CHECK-NEXT:      %116 = x86.ri.sub %40, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%41], %45 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%41 + 64], %46 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%41 + 128], %47 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%41 + 192], %48 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%41 + 256], %49 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%41 + 320], %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %117 = x86.ri.add %41, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %118 = x86.ri.sub %39, 8064 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %118, %116, %117, %42, %43, %44 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r11>
// CHECK-NEXT:    }
// CHECK-NEXT:    %119 = x86.ri.add %21, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %120 = x86.ri.add %20, 1536 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %121 = x86.ri.sub %19, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    %122 = x86.si.cmp %24, 3 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %122 : !x86.rflags<rflags>, ^bb0(%121 : !x86.reg64<rdi>, %120 : !x86.reg64<rsi>, %119 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>), ^bb1(%121 : !x86.reg64<rdi>, %120 : !x86.reg64<rsi>, %119 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>)
// CHECK-NEXT:  ^bb1(%123: !x86.reg64<rdi>, %124: !x86.reg64<rsi>, %125: !x86.reg64<rdx>, %126: !x86.reg64<rbp>, %127: !x86.reg64<rsp>, %128: !x86.reg64<r11>):
// CHECK-NEXT:    %129 = x86.ds.mov %126 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %130, %131 = x86.d.pop %129 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
