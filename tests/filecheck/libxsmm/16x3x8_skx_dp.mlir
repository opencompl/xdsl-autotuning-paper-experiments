// RUN: libxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-LIBXSMM-NEXT:      push rbp
// CHECK-LIBXSMM-NEXT:      push rbp
// CHECK-LIBXSMM-NEXT:      mov rbp, rsp
// CHECK-LIBXSMM-NEXT:      sub rsp, 192
// CHECK-LIBXSMM-NEXT:      mov r10, -64
// CHECK-LIBXSMM-NEXT:      and rsp, r10
// CHECK-LIBXSMM-NEXT:      mov r11, 0
// CHECK-LIBXSMM-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-LIBXSMM-NEXT:      add r11, 3
// CHECK-LIBXSMM-NEXT:      mov r10, 0
// CHECK-LIBXSMM-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-LIBXSMM-NEXT:      add r10, 16
// CHECK-MANUAL-NEXT:      vmovapd zmm26, [rdx]
// CHECK-MANUAL-NEXT:      vmovapd zmm27, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovapd zmm28, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovapd zmm29, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovapd zmm30, [rdx+256]
// CHECK-MANUAL-NEXT:      vmovapd zmm31, [rdx+320]
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+64]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-LIBXSMM-NEXT:     sub rsi, 64
// CHECK-MANUAL-NEXT:      vmovapd [rdx], zmm26
// CHECK-MANUAL-NEXT:      vmovapd [rdx+64], zmm27
// CHECK-MANUAL-NEXT:      vmovapd [rdx+128], zmm28
// CHECK-MANUAL-NEXT:      vmovapd [rdx+192], zmm29
// CHECK-MANUAL-NEXT:      vmovapd [rdx+256], zmm30
// CHECK-MANUAL-NEXT:      vmovapd [rdx+320], zmm31
// CHECK-COMPXSMM-NEXT:    sub rdi, 896
// CHECK-COMPXSMM-NEXT:    sub rsi, 64
// CHECK-MANUAL-NEXT:      add rdx, 128
// CHECK-LIBXSMM-NEXT:     sub rdi, 896
// CHECK-LIBXSMM-NEXT:      cmp r10, 16
// CHECK-LIBXSMM-NEXT:      jl [[SCF_M_BODY]]
// CHECK-LIBXSMM-NEXT:     add rdx, 256
// CHECK-COMPXSMM-NEXT:    sub rdi, 128
// CHECK-MANUAL-NEXT:      add rsi, 192
// CHECK-LIBXSMM-NEXT:     sub rdi, 128
// CHECK-COMPXSMM-NEXT:    add rdx, 256
// CHECK-LIBXSMM-NEXT:      cmp r11, 3
// CHECK-LIBXSMM-NEXT:      jl [[SCF_N_BODY]]
// CHECK-LIBXSMM-NEXT:      mov rsp, rbp
// CHECK-LIBXSMM-NEXT:      pop rbp
// CHECK-LIBXSMM-NEXT:      pop rbp
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
// CHECK-NEXT:      %17 = x86.ri.add %16, 3 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-NEXT:      %18 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      x86.fallthrough ^bb2(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb2(%19: !x86.reg64<rdi>, %20: !x86.reg64<rsi>, %21: !x86.reg64<rdx>, %22: !x86.reg64<rbp>, %23: !x86.reg64<rsp>, %24: !x86.reg64<r11>, %25: !x86.reg64<r10>):
// CHECK-NEXT:      x86.label "l34"
// CHECK-NEXT:      %26 = x86.ri.add %25, 16 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-NEXT:      %27 = x86.dm.vmovapd [%21] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %28 = x86.dm.vmovapd [%21 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %29 = x86.dm.vmovapd [%21 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %30 = x86.dm.vmovapd [%21 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %31 = x86.dm.vmovapd [%21 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %32 = x86.dm.vmovapd [%21 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %33 = x86.dm.vmovapd [%19] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %34 = x86.dm.vmovapd [%19 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %35 = x86.dm.vbroadcastsd [%20] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %36 = x86.rss.vfmadd231pd %27, %33, %35 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %37 = x86.rss.vfmadd231pd %28, %34, %35 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %38 = x86.dm.vbroadcastsd [%20 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %39 = x86.rss.vfmadd231pd %29, %33, %38 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %40 = x86.rss.vfmadd231pd %30, %34, %38 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %41 = x86.dm.vbroadcastsd [%20 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %42 = x86.ri.add %20, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %43 = x86.ri.add %19, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %44 = x86.rss.vfmadd231pd %31, %33, %41 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %45 = x86.rss.vfmadd231pd %32, %34, %41 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %46 = x86.dm.vmovapd [%43] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %47 = x86.dm.vmovapd [%43 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %48 = x86.dm.vbroadcastsd [%42] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %49 = x86.rss.vfmadd231pd %36, %46, %48 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %50 = x86.rss.vfmadd231pd %37, %47, %48 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %51 = x86.dm.vbroadcastsd [%42 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %52 = x86.rss.vfmadd231pd %39, %46, %51 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %53 = x86.rss.vfmadd231pd %40, %47, %51 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %54 = x86.dm.vbroadcastsd [%42 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %55 = x86.ri.add %42, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %56 = x86.ri.add %43, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %57 = x86.rss.vfmadd231pd %44, %46, %54 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %58 = x86.rss.vfmadd231pd %45, %47, %54 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %59 = x86.dm.vmovapd [%56] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %60 = x86.dm.vmovapd [%56 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %61 = x86.dm.vbroadcastsd [%55] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %62 = x86.rss.vfmadd231pd %49, %59, %61 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %63 = x86.rss.vfmadd231pd %50, %60, %61 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %64 = x86.dm.vbroadcastsd [%55 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %65 = x86.rss.vfmadd231pd %52, %59, %64 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %66 = x86.rss.vfmadd231pd %53, %60, %64 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %67 = x86.dm.vbroadcastsd [%55 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %68 = x86.ri.add %55, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %69 = x86.ri.add %56, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %70 = x86.rss.vfmadd231pd %57, %59, %67 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %71 = x86.rss.vfmadd231pd %58, %60, %67 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %72 = x86.dm.vmovapd [%69] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %73 = x86.dm.vmovapd [%69 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %74 = x86.dm.vbroadcastsd [%68] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %75 = x86.rss.vfmadd231pd %62, %72, %74 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %76 = x86.rss.vfmadd231pd %63, %73, %74 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %77 = x86.dm.vbroadcastsd [%68 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %78 = x86.rss.vfmadd231pd %65, %72, %77 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %79 = x86.rss.vfmadd231pd %66, %73, %77 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %80 = x86.dm.vbroadcastsd [%68 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %81 = x86.ri.add %68, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %82 = x86.ri.add %69, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %83 = x86.rss.vfmadd231pd %70, %72, %80 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %84 = x86.rss.vfmadd231pd %71, %73, %80 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %85 = x86.dm.vmovapd [%82] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %86 = x86.dm.vmovapd [%82 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %87 = x86.dm.vbroadcastsd [%81] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %88 = x86.rss.vfmadd231pd %75, %85, %87 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %89 = x86.rss.vfmadd231pd %76, %86, %87 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %90 = x86.dm.vbroadcastsd [%81 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231pd %78, %85, %90 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %92 = x86.rss.vfmadd231pd %79, %86, %90 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %93 = x86.dm.vbroadcastsd [%81 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %94 = x86.ri.add %81, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %95 = x86.ri.add %82, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %96 = x86.rss.vfmadd231pd %83, %85, %93 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %97 = x86.rss.vfmadd231pd %84, %86, %93 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %98 = x86.dm.vmovapd [%95] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %99 = x86.dm.vmovapd [%95 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %100 = x86.dm.vbroadcastsd [%94] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %101 = x86.rss.vfmadd231pd %88, %98, %100 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %102 = x86.rss.vfmadd231pd %89, %99, %100 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %103 = x86.dm.vbroadcastsd [%94 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %104 = x86.rss.vfmadd231pd %91, %98, %103 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %105 = x86.rss.vfmadd231pd %92, %99, %103 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %106 = x86.dm.vbroadcastsd [%94 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %107 = x86.ri.add %94, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %108 = x86.ri.add %95, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %109 = x86.rss.vfmadd231pd %96, %98, %106 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %110 = x86.rss.vfmadd231pd %97, %99, %106 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %111 = x86.dm.vmovapd [%108] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %112 = x86.dm.vmovapd [%108 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %113 = x86.dm.vbroadcastsd [%107] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %114 = x86.rss.vfmadd231pd %101, %111, %113 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %115 = x86.rss.vfmadd231pd %102, %112, %113 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %116 = x86.dm.vbroadcastsd [%107 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %117 = x86.rss.vfmadd231pd %104, %111, %116 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %118 = x86.rss.vfmadd231pd %105, %112, %116 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %119 = x86.dm.vbroadcastsd [%107 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %120 = x86.ri.add %107, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %121 = x86.ri.add %108, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %122 = x86.rss.vfmadd231pd %109, %111, %119 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %123 = x86.rss.vfmadd231pd %110, %112, %119 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %124 = x86.dm.vmovapd [%121] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %125 = x86.dm.vmovapd [%121 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %126 = x86.dm.vbroadcastsd [%120] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %127 = x86.rss.vfmadd231pd %114, %124, %126 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %128 = x86.rss.vfmadd231pd %115, %125, %126 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %129 = x86.dm.vbroadcastsd [%120 + 64] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %130 = x86.rss.vfmadd231pd %117, %124, %129 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %131 = x86.rss.vfmadd231pd %118, %125, %129 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %132 = x86.dm.vbroadcastsd [%120 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %133 = x86.ri.add %120, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %134 = x86.ri.add %121, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %135 = x86.rss.vfmadd231pd %122, %124, %132 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %136 = x86.rss.vfmadd231pd %123, %125, %132 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %137 = x86.ri.sub %133, 64 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%21], %127 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 64], %128 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 128], %130 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 192], %131 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 256], %135 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%21 + 320], %136 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %138 = x86.ri.add %21, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %139 = x86.ri.sub %134, 896 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %140 = x86.si.cmp %26, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %140 : !x86.rflags<rflags>, ^bb2(%139 : !x86.reg64<rdi>, %137 : !x86.reg64<rsi>, %138 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>), ^bb3(%139 : !x86.reg64<rdi>, %137 : !x86.reg64<rsi>, %138 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb3(%141: !x86.reg64<rdi>, %142: !x86.reg64<rsi>, %143: !x86.reg64<rdx>, %144: !x86.reg64<rbp>, %145: !x86.reg64<rsp>, %146: !x86.reg64<r11>, %147: !x86.reg64<r10>):
// CHECK-NEXT:      %148 = x86.ri.add %143, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %149 = x86.ri.add %142, 192 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %150 = x86.ri.sub %141, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %151 = x86.si.cmp %146, 3 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %151 : !x86.rflags<rflags>, ^bb1(%150 : !x86.reg64<rdi>, %149 : !x86.reg64<rsi>, %148 : !x86.reg64<rdx>, %144 : !x86.reg64<rbp>, %145 : !x86.reg64<rsp>, %146 : !x86.reg64<r11>), ^bb4(%150 : !x86.reg64<rdi>, %149 : !x86.reg64<rsi>, %148 : !x86.reg64<rdx>, %144 : !x86.reg64<rbp>, %145 : !x86.reg64<rsp>, %146 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb4(%152: !x86.reg64<rdi>, %153: !x86.reg64<rsi>, %154: !x86.reg64<rdx>, %155: !x86.reg64<rbp>, %156: !x86.reg64<rsp>, %157: !x86.reg64<r11>):
// CHECK-NEXT:      %158 = x86.ds.mov %155 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %159, %160 = x86.d.pop %158 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
