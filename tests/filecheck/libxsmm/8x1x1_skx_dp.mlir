// RUN: libxsmm-gemm dense %t matmul_bac 8 1 1 8 1 8 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 8 1 1 8 1 8 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 8 1 1 8 1 8 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      mov rbp, rsp
// CHECK-MANUAL-NEXT:      sub rsp, 192
// CHECK-MANUAL-NEXT:      mov r10, -64
// CHECK-MANUAL-NEXT:      and rsp, r10
// CHECK-LIBXSMM-NEXT:      mov r11, 0
// CHECK-LIBXSMM-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-LIBXSMM-NEXT:      add r11, 1
// CHECK-LIBXSMM-NEXT:      mov r10, 0
// CHECK-LIBXSMM-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-LIBXSMM-NEXT:      add r10, 8
// CHECK-MANUAL-NEXT:      vmovapd zmm31, [rdx]
// CHECK-MANUAL-NEXT:      vmovapd zmm0, [rdi]
// CHECK-MANUAL-NEXT:      add rdi, 64
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm0, [rsi]{1to8}
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-LIBXSMM-NEXT:     sub rsi, 8
// CHECK-MANUAL-NEXT:      vmovapd [rdx], zmm31
// CHECK-COMPXSMM-NEXT:    add rdi, 0
// CHECK-COMPXSMM-NEXT:    sub rsi, 8
// CHECK-MANUAL-NEXT:      add rdx, 64
// CHECK-LIBXSMM-NEXT:     sub rdi, 0
// CHECK-LIBXSMM-NEXT:      cmp r10, 8
// CHECK-LIBXSMM-NEXT:      jl [[SCF_M_BODY]]
// CHECK-LIBXSMM-NEXT:     add rdx, 0
// CHECK-COMPXSMM-NEXT:    sub rdi, 64
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-LIBXSMM-NEXT:     sub rdi, 64
// CHECK-COMPXSMM-NEXT:    add rdx, 0
// CHECK-LIBXSMM-NEXT:      cmp r11, 1
// CHECK-LIBXSMM-NEXT:      jl [[SCF_N_BODY]]
// CHECK-MANUAL-NEXT:      mov rsp, rbp
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      ret

// CHECK:       builtin.module {
// CHECK-NEXT:    x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:      %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:      %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:      %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:      %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:      x86.fallthrough ^bb1(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb1(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-NEXT:      x86.label "l33"
// CHECK-NEXT:      %17 = x86.ri.add %16, 1 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %26 = x86.ri.add %25, 8 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %27 = x86.dm.vmovapd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %28 = x86.dm.vmovapd [%19] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %29 = x86.ri.add %19, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %30 = x86.rsm.vfmadd231pd %27, %28, [%20] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %31 = x86.ri.add %20, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %32 = x86.ri.sub %31, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%21], %30 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %33 = x86.ri.add %21, 64 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %34 = x86.ri.sub %29, 0 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %35 = x86.si.cmp %26, 8 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %35 : !x86.rflags<rflags>, ^bb2(%34 : !x86.reg64<rdi>, %32 : !x86.reg64<rsi>, %33 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>), ^bb3(%34 : !x86.reg64<rdi>, %32 : !x86.reg64<rsi>, %33 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb3(%36: !x86.reg64<rdi>, %37: !x86.reg64<rsi>, %38: !x86.reg64<rdx>, %39: !x86.reg64<rbp>, %40: !x86.reg64<rsp>, %41: !x86.reg64<r11>, %42: !x86.reg64<r10>):
// CHECK-NEXT:      %43 = x86.ri.add %38, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %44 = x86.ri.add %37, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %45 = x86.ri.sub %36, 64 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %46 = x86.si.cmp %41, 1 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %46 : !x86.rflags<rflags>, ^bb1(%45 : !x86.reg64<rdi>, %44 : !x86.reg64<rsi>, %43 : !x86.reg64<rdx>, %39 : !x86.reg64<rbp>, %40 : !x86.reg64<rsp>, %41 : !x86.reg64<r11>), ^bb4(%45 : !x86.reg64<rdi>, %44 : !x86.reg64<rsi>, %43 : !x86.reg64<rdx>, %39 : !x86.reg64<rbp>, %40 : !x86.reg64<rsp>, %41 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb4(%47: !x86.reg64<rdi>, %48: !x86.reg64<rsi>, %49: !x86.reg64<rdx>, %50: !x86.reg64<rbp>, %51: !x86.reg64<rsp>, %52: !x86.reg64<r11>):
// CHECK-NEXT:      %53 = x86.ds.mov %50 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %54, %55 = x86.d.pop %53 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
