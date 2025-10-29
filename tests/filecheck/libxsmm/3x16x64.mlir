// RUN: libxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// CHECK:       x86_func.func public @matmul_bac(%0 : !x86.reg<rdi>, %1 : !x86.reg<rsi>, %2 : !x86.reg<rdx>) {
// CHECK-NEXT:    %3 = x86.ds.mov %0 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %4 = x86.ds.mov %1 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %5 = x86.ds.mov %2 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %6 = x86.get_register : () -> !x86.reg<rbp>
// CHECK-NEXT:    %7 = x86.get_register : () -> !x86.reg<rsp>
// CHECK-NEXT:    %8 = x86.s.push %7, %6 : (!x86.reg<rsp>, !x86.reg<rbp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %9 = x86.ds.mov %8 : (!x86.reg<rsp>) -> !x86.reg<rbp>
// CHECK-NEXT:    %10 = x86.ri.sub %8, 192 : (!x86.reg<rsp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %11 = x86.di.mov -64 : () -> !x86.reg<r10>
// CHECK-NEXT:    %12 = x86.rs.and %10, %11 : (!x86.reg<rsp>, !x86.reg<r10>) -> !x86.reg<rsp>
// CHECK-NEXT:    %13 = x86.di.mov 0 : () -> !x86.reg<r11>
// CHECK-NEXT:    x86.c.jmp ^bb0(%13 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb0(%14 : !x86.reg<rdi>, %15 : !x86.reg<rsi>, %16 : !x86.reg<rdx>, %17 : !x86.reg<rbp>, %18 : !x86.reg<rsp>, %19 : !x86.reg<r10>, %20 : !x86.reg<r11>):
// CHECK-NEXT:    x86.label "33"
// CHECK-NEXT:    %21 = x86.ri.add %20, 3 : (!x86.reg<r11>) -> !x86.reg<r11>
// CHECK-NEXT:    %22 = x86.di.mov 0 : () -> !x86.reg<r10>
// CHECK-NEXT:    x86.c.jmp ^bb1(%22 : !x86.reg<r10>)
// CHECK-NEXT:  ^bb1(%23 : !x86.reg<rdi>, %24 : !x86.reg<rsi>, %25 : !x86.reg<rdx>, %26 : !x86.reg<rbp>, %27 : !x86.reg<rsp>, %28 : !x86.reg<r10>, %29 : !x86.reg<r11>):
// CHECK-NEXT:    x86.label "34"
// CHECK-NEXT:    %30 = x86.ri.add %28, 16 : (!x86.reg<r10>) -> !x86.reg<r10>
// CHECK-NEXT:    %31 = x86.dm.vmovapd %25, 0 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:    %32 = x86.dm.vmovapd %25, 64 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:    %33 = x86.dm.vmovapd %25, 128 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:    %34 = x86.dm.vmovapd %25, 192 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:    %35 = x86.dm.vmovapd %25, 256 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:    %36 = x86.dm.vmovapd %25, 320 : (!x86.reg<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:    %37 = x86.di.mov 0 : () -> !x86.reg<r12>
// CHECK-NEXT:    x86.c.jmp ^bb2(%37 : !x86.reg<r12>)
// CHECK-NEXT:  ^bb2(%38 : !x86.reg<rdi>, %39 : !x86.reg<rsi>, %40 : !x86.reg<rdx>, %41 : !x86.reg<rbp>, %42 : !x86.reg<rsp>, %43 : !x86.reg<r10>, %44 : !x86.reg<r11>, %45 : !x86.avx512reg<zmm26>, %46 : !x86.avx512reg<zmm27>, %47 : !x86.avx512reg<zmm28>, %48 : !x86.avx512reg<zmm29>, %49 : !x86.avx512reg<zmm30>, %50 : !x86.avx512reg<zmm31>, %51 : !x86.reg<r12>):
// CHECK-NEXT:    x86.label "35"
// CHECK-NEXT:    %52 = x86.ri.add %51, 4 : (!x86.reg<r12>) -> !x86.reg<r12>
// CHECK-NEXT:    %53 = x86.dm.vbroadcastsd %39, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %54 = x86.dm.vbroadcastsd %39, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %55 = x86.dm.vbroadcastsd %39, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %56 = x86.ri.add %39, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %57 = x86.ri.add %38, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %58 = x86.dm.vbroadcastsd %56, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %59 = x86.dm.vbroadcastsd %56, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %60 = x86.dm.vbroadcastsd %56, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %61 = x86.ri.add %56, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %62 = x86.ri.add %57, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %63 = x86.dm.vbroadcastsd %61, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %64 = x86.dm.vbroadcastsd %61, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %65 = x86.dm.vbroadcastsd %61, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %66 = x86.ri.add %61, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %67 = x86.ri.add %62, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %68 = x86.dm.vbroadcastsd %66, 0 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %69 = x86.dm.vbroadcastsd %66, 512 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %70 = x86.dm.vbroadcastsd %66, 1024 : (!x86.reg<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:    %71 = x86.ri.add %66, 8 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %72 = x86.ri.add %67, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %73 = x86.si.cmp %52, 64 : (!x86.reg<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %73 : !x86.rflags<rflags>, ^bb2(%72 : !x86.reg<rdi>, %71 : !x86.reg<rsi>, %40 : !x86.reg<rdx>, %41 : !x86.reg<rbp>, %42 : !x86.reg<rsp>, %43 : !x86.reg<r10>, %44 : !x86.reg<r11>, %45 : !x86.avx512reg<zmm26>, %46 : !x86.avx512reg<zmm27>, %47 : !x86.avx512reg<zmm28>, %48 : !x86.avx512reg<zmm29>, %49 : !x86.avx512reg<zmm30>, %50 : !x86.avx512reg<zmm31>, %52 : !x86.reg<r12>), ^bb3(%72 : !x86.reg<rdi>, %71 : !x86.reg<rsi>, %40 : !x86.reg<rdx>, %41 : !x86.reg<rbp>, %42 : !x86.reg<rsp>, %43 : !x86.reg<r10>, %44 : !x86.reg<r11>, %45 : !x86.avx512reg<zmm26>, %46 : !x86.avx512reg<zmm27>, %47 : !x86.avx512reg<zmm28>, %48 : !x86.avx512reg<zmm29>, %49 : !x86.avx512reg<zmm30>, %50 : !x86.avx512reg<zmm31>, %52 : !x86.reg<r12>)
// CHECK-NEXT:  ^bb3(%74 : !x86.reg<rdi>, %75 : !x86.reg<rsi>, %76 : !x86.reg<rdx>, %77 : !x86.reg<rbp>, %78 : !x86.reg<rsp>, %79 : !x86.reg<r10>, %80 : !x86.reg<r11>, %81 : !x86.avx512reg<zmm26>, %82 : !x86.avx512reg<zmm27>, %83 : !x86.avx512reg<zmm28>, %84 : !x86.avx512reg<zmm29>, %85 : !x86.avx512reg<zmm30>, %86 : !x86.avx512reg<zmm31>, %87 : !x86.reg<r12>):
// CHECK-NEXT:    %88 = x86.ri.sub %75, 512 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %89 = x86.ri.add %76, 128 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %90 = x86.ri.sub %74, 8064 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %91 = x86.si.cmp %79, 16 : (!x86.reg<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %91 : !x86.rflags<rflags>, ^bb1(%90 : !x86.reg<rdi>, %88 : !x86.reg<rsi>, %89 : !x86.reg<rdx>, %77 : !x86.reg<rbp>, %78 : !x86.reg<rsp>, %79 : !x86.reg<r10>, %80 : !x86.reg<r11>), ^bb4(%90 : !x86.reg<rdi>, %88 : !x86.reg<rsi>, %89 : !x86.reg<rdx>, %77 : !x86.reg<rbp>, %78 : !x86.reg<rsp>, %79 : !x86.reg<r10>, %80 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb4(%92 : !x86.reg<rdi>, %93 : !x86.reg<rsi>, %94 : !x86.reg<rdx>, %95 : !x86.reg<rbp>, %96 : !x86.reg<rsp>, %97 : !x86.reg<r10>, %98 : !x86.reg<r11>):
// CHECK-NEXT:    %99 = x86.ri.add %94, 256 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %100 = x86.ri.add %93, 1536 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %101 = x86.ri.sub %92, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %102 = x86.si.cmp %98, 3 : (!x86.reg<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %102 : !x86.rflags<rflags>, ^bb0(%101 : !x86.reg<rdi>, %100 : !x86.reg<rsi>, %99 : !x86.reg<rdx>, %95 : !x86.reg<rbp>, %96 : !x86.reg<rsp>, %97 : !x86.reg<r10>, %98 : !x86.reg<r11>), ^bb5(%101 : !x86.reg<rdi>, %100 : !x86.reg<rsi>, %99 : !x86.reg<rdx>, %95 : !x86.reg<rbp>, %96 : !x86.reg<rsp>, %97 : !x86.reg<r10>, %98 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb5(%103 : !x86.reg<rdi>, %104 : !x86.reg<rsi>, %105 : !x86.reg<rdx>, %106 : !x86.reg<rbp>, %107 : !x86.reg<rsp>, %108 : !x86.reg<r10>, %109 : !x86.reg<r11>):
// CHECK-NEXT:    %110 = x86.ds.mov %106 : (!x86.reg<rbp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %111, %112 = x86.d.pop %110 : (!x86.reg<rsp>) -> (!x86.reg<rbp>, !x86.reg<rsp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
