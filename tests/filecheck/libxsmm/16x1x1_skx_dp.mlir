// RUN: libxsmm-gemm dense %t matmul_bac 16 1 1 16 1 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 16 1 1 16 1 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 1 16 1 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[ASM_LABEL_33:^\S+]]:
// CHECK-NEXT:      add r11, 1
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[ASM_LABEL_34:^\S+]]:
// CHECK-NEXT:      add r10, 16
// CHECK-NEXT:      vmovapd zmm30, [rdx]
// CHECK-NEXT:      vmovapd zmm31, [rdx+64]
// CHECK-NEXT:      vmovapd zmm1, [rdi]
// CHECK-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      sub rsi, 8
// CHECK-NEXT:      vmovapd [rdx], zmm30
// CHECK-NEXT:      vmovapd [rdx+64], zmm31
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      sub rdi, 0
// CHECK-NEXT:      cmp r10, 16
// CHECK-NEXT:      jl [[ASM_LABEL_34]]
// CHECK-NEXT:      add rdx, 0
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      sub rdi, 128
// CHECK-NEXT:      cmp r11, 1
// CHECK-NEXT:      jl [[ASM_LABEL_33]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret

// CHECK-IR-LIBXSMM:       builtin.module {
// CHECK-IR-LIBXSMM-NEXT:    x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-IR-LIBXSMM-NEXT:      %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-IR-LIBXSMM-NEXT:      %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-IR-LIBXSMM-NEXT:      %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l33"
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 1 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %26 = x86.ri.add %25, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %27 = x86.dm.vmovapd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %28 = x86.dm.vmovapd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %29 = x86.dm.vmovapd [%19] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.dm.vmovapd [%19 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vbroadcastsd [%20] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.ri.add %20, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.ri.add %19, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.rss.vfmadd231pd %27, %29, %31 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.rss.vfmadd231pd %28, %30, %31 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.ri.sub %32, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%21], %34 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovapd [%21 + 64], %35 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.ri.add %21, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.ri.sub %33, 0 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.si.cmp %26, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %39 : !x86.rflags<rflags>, ^bb2(%38 : !x86.reg64<rdi>, %36 : !x86.reg64<rsi>, %37 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>), ^bb3(%38 : !x86.reg64<rdi>, %36 : !x86.reg64<rsi>, %37 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%40: !x86.reg64<rdi>, %41: !x86.reg64<rsi>, %42: !x86.reg64<rdx>, %43: !x86.reg64<rbp>, %44: !x86.reg64<rsp>, %45: !x86.reg64<r11>, %46: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.ri.add %42, 0 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.ri.add %41, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.ri.sub %40, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.si.cmp %45, 1 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %50 : !x86.rflags<rflags>, ^bb1(%49 : !x86.reg64<rdi>, %48 : !x86.reg64<rsi>, %47 : !x86.reg64<rdx>, %43 : !x86.reg64<rbp>, %44 : !x86.reg64<rsp>, %45 : !x86.reg64<r11>), ^bb4(%49 : !x86.reg64<rdi>, %48 : !x86.reg64<rsi>, %47 : !x86.reg64<rdx>, %43 : !x86.reg64<rbp>, %44 : !x86.reg64<rsp>, %45 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb4(%51: !x86.reg64<rdi>, %52: !x86.reg64<rsi>, %53: !x86.reg64<rdx>, %54: !x86.reg64<rbp>, %55: !x86.reg64<rsp>, %56: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %57 = x86.ds.mov %54 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %58, %59 = x86.d.pop %57 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }
