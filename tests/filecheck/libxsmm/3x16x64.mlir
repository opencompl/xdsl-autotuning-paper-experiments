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
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
