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
// CHECK-NEXT:    x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-NEXT:  ^bb1(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-NEXT:    x86.label "l34"
// CHECK-NEXT:    %26 = x86.ri.add %25, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:    %27 = x86.dm.vmovapd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %28 = x86.dm.vmovapd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %29 = x86.dm.vmovapd [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %30 = x86.dm.vmovapd [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %31 = x86.dm.vmovapd [%21 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %32 = x86.dm.vmovapd [%21 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %33 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:    %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46 = x86_scf.for %47 : !x86.reg64<r12>  = %33 to 64 : si32 step 4 : si32 iter_args(%48 = %19, %49 = %20, %50 = %21, %51 = %22, %52 = %23, %53 = %24, %54 = %26, %55 = %27, %56 = %28, %57 = %29, %58 = %30, %59 = %31, %60 = %32) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r11>, !x86.reg64<r10>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) {
// CHECK-NEXT:      %61 = x86.dm.vmovapd [%48] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %62 = x86.dm.vmovapd [%48 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %63 = x86.dm.vbroadcastsd [%49] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %64 = x86.rss.vfmadd231pd %55, %61, %63 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %65 = x86.rss.vfmadd231pd %56, %62, %63 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %66 = x86.dm.vbroadcastsd [%49 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %67 = x86.rss.vfmadd231pd %57, %61, %66 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %68 = x86.rss.vfmadd231pd %58, %62, %66 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %69 = x86.dm.vbroadcastsd [%49 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %70 = x86.ri.add %49, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %71 = x86.ri.add %48, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %72 = x86.rss.vfmadd231pd %59, %61, %69 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %73 = x86.rss.vfmadd231pd %60, %62, %69 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %74 = x86.dm.vmovapd [%71] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %75 = x86.dm.vmovapd [%71 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %76 = x86.dm.vbroadcastsd [%70] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %77 = x86.rss.vfmadd231pd %64, %74, %76 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %78 = x86.rss.vfmadd231pd %65, %75, %76 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %79 = x86.dm.vbroadcastsd [%70 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %80 = x86.rss.vfmadd231pd %67, %74, %79 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %81 = x86.rss.vfmadd231pd %68, %75, %79 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %82 = x86.dm.vbroadcastsd [%70 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %83 = x86.ri.add %70, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %84 = x86.ri.add %71, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %85 = x86.rss.vfmadd231pd %72, %74, %82 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %86 = x86.rss.vfmadd231pd %73, %75, %82 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %87 = x86.dm.vmovapd [%84] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %88 = x86.dm.vmovapd [%84 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %89 = x86.dm.vbroadcastsd [%83] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %90 = x86.rss.vfmadd231pd %77, %87, %89 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231pd %78, %88, %89 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %92 = x86.dm.vbroadcastsd [%83 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %93 = x86.rss.vfmadd231pd %80, %87, %92 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %94 = x86.rss.vfmadd231pd %81, %88, %92 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %95 = x86.dm.vbroadcastsd [%83 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %96 = x86.ri.add %83, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %97 = x86.ri.add %84, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %98 = x86.rss.vfmadd231pd %85, %87, %95 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %99 = x86.rss.vfmadd231pd %86, %88, %95 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %100 = x86.dm.vmovapd [%97] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %101 = x86.dm.vmovapd [%97 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %102 = x86.dm.vbroadcastsd [%96] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %103 = x86.rss.vfmadd231pd %90, %100, %102 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %104 = x86.rss.vfmadd231pd %91, %101, %102 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %105 = x86.dm.vbroadcastsd [%96 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %106 = x86.rss.vfmadd231pd %93, %100, %105 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %107 = x86.rss.vfmadd231pd %94, %101, %105 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %108 = x86.dm.vbroadcastsd [%96 + 1024] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %109 = x86.ri.add %96, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %110 = x86.ri.add %97, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %111 = x86.rss.vfmadd231pd %98, %100, %108 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %112 = x86.rss.vfmadd231pd %99, %101, %108 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      x86_scf.yield %110, %109, %50, %51, %52, %53, %54, %103, %104, %106, %107, %111, %112 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.reg64<r11>, !x86.reg64<r10>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>
// CHECK-NEXT:    }
// CHECK-NEXT:    %113 = x86.ri.sub %35, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    x86.ms.vmovapd [%36], %41 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:    x86.ms.vmovapd [%36 + 64], %42 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:    x86.ms.vmovapd [%36 + 128], %43 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:    x86.ms.vmovapd [%36 + 192], %44 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:    x86.ms.vmovapd [%36 + 256], %45 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:    x86.ms.vmovapd [%36 + 320], %46 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:    %114 = x86.ri.add %36, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %115 = x86.ri.sub %34, 8064 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    %116 = x86.si.cmp %40, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %116 : !x86.rflags<rflags>, ^bb1(%115 : !x86.reg64<rdi>, %113 : !x86.reg64<rsi>, %114 : !x86.reg64<rdx>, %37 : !x86.reg64<rbp>, %38 : !x86.reg64<rsp>, %39 : !x86.reg64<r11>, %40 : !x86.reg64<r10>), ^bb2(%115 : !x86.reg64<rdi>, %113 : !x86.reg64<rsi>, %114 : !x86.reg64<rdx>, %37 : !x86.reg64<rbp>, %38 : !x86.reg64<rsp>, %39 : !x86.reg64<r11>, %40 : !x86.reg64<r10>)
// CHECK-NEXT:  ^bb2(%117: !x86.reg64<rdi>, %118: !x86.reg64<rsi>, %119: !x86.reg64<rdx>, %120: !x86.reg64<rbp>, %121: !x86.reg64<rsp>, %122: !x86.reg64<r11>, %123: !x86.reg64<r10>):
// CHECK-NEXT:    %124 = x86.ri.add %119, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %125 = x86.ri.add %118, 1536 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %126 = x86.ri.sub %117, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    %127 = x86.si.cmp %122, 3 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %127 : !x86.rflags<rflags>, ^bb0(%126 : !x86.reg64<rdi>, %125 : !x86.reg64<rsi>, %124 : !x86.reg64<rdx>, %120 : !x86.reg64<rbp>, %121 : !x86.reg64<rsp>, %122 : !x86.reg64<r11>), ^bb3(%126 : !x86.reg64<rdi>, %125 : !x86.reg64<rsi>, %124 : !x86.reg64<rdx>, %120 : !x86.reg64<rbp>, %121 : !x86.reg64<rsp>, %122 : !x86.reg64<r11>)
// CHECK-NEXT:  ^bb3(%128: !x86.reg64<rdi>, %129: !x86.reg64<rsi>, %130: !x86.reg64<rdx>, %131: !x86.reg64<rbp>, %132: !x86.reg64<rsp>, %133: !x86.reg64<r11>):
// CHECK-NEXT:    %134 = x86.ds.mov %131 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %135, %136 = x86.d.pop %134 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
