// RUN: libxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.ds.mov %0 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:    %4 = x86.ds.mov %1 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:    %5 = x86.ds.mov %2 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }
