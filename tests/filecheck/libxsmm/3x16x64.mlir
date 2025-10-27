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
// CHECK-NEXT:    %31 = x86.ri.add %25, 128 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %32 = x86.ri.sub %23, 8064 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %33 = x86.si.cmp %30, 16 : (!x86.reg<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %33 : !x86.rflags<rflags>, ^bb1(%32 : !x86.reg<rdi>, %24 : !x86.reg<rsi>, %31 : !x86.reg<rdx>, %26 : !x86.reg<rbp>, %27 : !x86.reg<rsp>, %30 : !x86.reg<r10>, %29 : !x86.reg<r11>), ^bb2(%32 : !x86.reg<rdi>, %24 : !x86.reg<rsi>, %31 : !x86.reg<rdx>, %26 : !x86.reg<rbp>, %27 : !x86.reg<rsp>, %30 : !x86.reg<r10>, %29 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb2(%34 : !x86.reg<rdi>, %35 : !x86.reg<rsi>, %36 : !x86.reg<rdx>, %37 : !x86.reg<rbp>, %38 : !x86.reg<rsp>, %39 : !x86.reg<r10>, %40 : !x86.reg<r11>):
// CHECK-NEXT:    %41 = x86.ri.add %36, 256 : (!x86.reg<rdx>) -> !x86.reg<rdx>
// CHECK-NEXT:    %42 = x86.ri.add %35, 1536 : (!x86.reg<rsi>) -> !x86.reg<rsi>
// CHECK-NEXT:    %43 = x86.ri.sub %34, 128 : (!x86.reg<rdi>) -> !x86.reg<rdi>
// CHECK-NEXT:    %44 = x86.si.cmp %40, 3 : (!x86.reg<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:    x86.c.jl %44 : !x86.rflags<rflags>, ^bb0(%43 : !x86.reg<rdi>, %42 : !x86.reg<rsi>, %41 : !x86.reg<rdx>, %37 : !x86.reg<rbp>, %38 : !x86.reg<rsp>, %39 : !x86.reg<r10>, %40 : !x86.reg<r11>), ^bb3(%43 : !x86.reg<rdi>, %42 : !x86.reg<rsi>, %41 : !x86.reg<rdx>, %37 : !x86.reg<rbp>, %38 : !x86.reg<rsp>, %39 : !x86.reg<r10>, %40 : !x86.reg<r11>)
// CHECK-NEXT:  ^bb3(%45 : !x86.reg<rdi>, %46 : !x86.reg<rsi>, %47 : !x86.reg<rdx>, %48 : !x86.reg<rbp>, %49 : !x86.reg<rsp>, %50 : !x86.reg<r10>, %51 : !x86.reg<r11>):
// CHECK-NEXT:    %52 = x86.ds.mov %48 : (!x86.reg<rbp>) -> !x86.reg<rsp>
// CHECK-NEXT:    %53, %54 = x86.d.pop %52 : (!x86.reg<rsp>) -> (!x86.reg<rbp>, !x86.reg<rsp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
