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
// CHECK-NEXT:    %31 = x86.di.mov 0 : () -> !x86.reg<r12>
// CHECK-NEXT:    x86.c.jmp ^bb2(%31 : !x86.reg<r12>)
// CHECK-NEXT:  ^bb2(%32 : !x86.reg<rdi>, %33 : !x86.reg<rsi>, %34 : !x86.reg<rdx>, %35 : !x86.reg<rbp>, %36 : !x86.reg<rsp>, %37 : !x86.reg<r10>, %38 : !x86.reg<r11>, %39 : !x86.reg<r12>):
// CHECK-NEXT:    x86.label "35"
// CHECK-NEXT:    %40 = x86.ri.add %39, 4 : (!x86.reg<r12>) -> !x86.reg<r12>
// CHECK-NEXT:    %41 = x86.si.cmp %40, 64 : (!x86.reg<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %41 : !x86.rflags<rflags>, ^bb2(%32 : !x86.reg<rdi>, %33 : !x86.reg<rsi>, %34 : !x86.reg<rdx>, %35 : !x86.reg<rbp>, %36 : !x86.reg<rsp>, %37 : !x86.reg<r10>, %38 : !x86.reg<r11>, %40 : !x86.reg<r12>), ^bb3(%32 : !x86.reg<rdi>, %33 : !x86.reg<rsi>, %34 : !x86.reg<rdx>, %35 : !x86.reg<rbp>, %36 : !x86.reg<rsp>, %37 : !x86.reg<r10>, %38 : !x86.reg<r11>, %40 : !x86.reg<r12>)
// CHECK-NEXT:  ^bb3(%42 : !x86.reg<rdi>, %43 : !x86.reg<rsi>, %44 : !x86.reg<rdx>, %45 : !x86.reg<rbp>, %46 : !x86.reg<rsp>, %47 : !x86.reg<r10>, %48 : !x86.reg<r11>, %49 : !x86.reg<r12>):
// CHECK-NEXT:    %50 = x86.ri.sub %43, 512 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %51 = x86.ri.add %44, 128 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %52 = x86.ri.sub %42, 8064 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %53 = x86.si.cmp %47, 16 : (!x86.reg<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %53 : !x86.rflags<rflags>, ^bb1(%52 : !x86.reg<rdi>, %50 : !x86.reg<rsi>, %51 : !x86.reg<rdx>, %45 : !x86.reg<rbp>, %46 : !x86.reg<rsp>, %47 : !x86.reg<r10>, %48 : !x86.reg<r11>), ^bb4(%52 : !x86.reg<rdi>, %50 : !x86.reg<rsi>, %51 : !x86.reg<rdx>, %45 : !x86.reg<rbp>, %46 : !x86.reg<rsp>, %47 : !x86.reg<r10>, %48 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb4(%54 : !x86.reg<rdi>, %55 : !x86.reg<rsi>, %56 : !x86.reg<rdx>, %57 : !x86.reg<rbp>, %58 : !x86.reg<rsp>, %59 : !x86.reg<r10>, %60 : !x86.reg<r11>):
// CHECK-NEXT:    %61 = x86.ri.add %56, 256 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %62 = x86.ri.add %55, 1536 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %63 = x86.ri.sub %54, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %64 = x86.si.cmp %60, 3 : (!x86.reg<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %64 : !x86.rflags<rflags>, ^bb0(%63 : !x86.reg<rdi>, %62 : !x86.reg<rsi>, %61 : !x86.reg<rdx>, %57 : !x86.reg<rbp>, %58 : !x86.reg<rsp>, %59 : !x86.reg<r10>, %60 : !x86.reg<r11>), ^bb5(%63 : !x86.reg<rdi>, %62 : !x86.reg<rsi>, %61 : !x86.reg<rdx>, %57 : !x86.reg<rbp>, %58 : !x86.reg<rsp>, %59 : !x86.reg<r10>, %60 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb5(%65 : !x86.reg<rdi>, %66 : !x86.reg<rsi>, %67 : !x86.reg<rdx>, %68 : !x86.reg<rbp>, %69 : !x86.reg<rsp>, %70 : !x86.reg<r10>, %71 : !x86.reg<r11>):
// CHECK-NEXT:    %72 = x86.ds.mov %68 : (!x86.reg<rbp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %73, %74 = x86.d.pop %72 : (!x86.reg<rsp>) -> (!x86.reg<rbp>, !x86.reg<rsp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
