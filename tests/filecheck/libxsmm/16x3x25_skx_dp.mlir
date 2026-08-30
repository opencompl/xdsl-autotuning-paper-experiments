// RUN: libxsmm-gemm dense %t matmul_bac 16 3 25 16 25 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 16 3 25 16 25 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 25 16 25 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-MANUAL,CHECK-COMPXSMM

// CHECK-MANUAL:       .intel_syntax noprefix
// CHECK-MANUAL-NEXT:  .text
// CHECK-MANUAL-NEXT:  .globl matmul_bac
// CHECK-MANUAL-NEXT:  matmul_bac:
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      push r12
// CHECK-MANUAL-NEXT:      push rbp
// CHECK-MANUAL-NEXT:      mov rbp, rsp
// CHECK-MANUAL-NEXT:      sub rsp, 192
// CHECK-MANUAL-NEXT:      mov r10, -64
// CHECK-MANUAL-NEXT:      and rsp, r10
// CHECK-MANUAL-NEXT:      mov r11, 0
// CHECK-MANUAL-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-MANUAL-NEXT:      add r11, 3
// CHECK-MANUAL-NEXT:      mov r10, 0
// CHECK-MANUAL-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-MANUAL-NEXT:      add r10, 16
// CHECK-MANUAL-NEXT:      vmovapd zmm26, [rdx]
// CHECK-MANUAL-NEXT:      vmovapd zmm27, [rdx+64]
// CHECK-MANUAL-NEXT:      vmovapd zmm28, [rdx+128]
// CHECK-MANUAL-NEXT:      vmovapd zmm29, [rdx+192]
// CHECK-MANUAL-NEXT:      vmovapd zmm30, [rdx+256]
// CHECK-MANUAL-NEXT:      vmovapd zmm31, [rdx+320]
// CHECK-MANUAL-NEXT:      mov r12, 0
// CHECK-MANUAL-NEXT:  [[SCF_K_BODY:^\S+]]:
// CHECK-MANUAL-NEXT:      add r12, 4
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+200]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+400]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+200]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+400]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+200]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+400]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+200]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+400]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-MANUAL-NEXT:      cmp r12, 24
// CHECK-MANUAL-NEXT:      jl [[SCF_K_BODY]]
// CHECK-MANUAL-NEXT:      vmovapd zmm1, [rdi]
// CHECK-MANUAL-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+200]
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-MANUAL-NEXT:      vbroadcastsd zmm0, [rsi+400]
// CHECK-MANUAL-NEXT:      add rsi, 8
// CHECK-MANUAL-NEXT:      add rdi, 128
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-MANUAL-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-COMPXSMM-NEXT:    sub rdi, 3072
// CHECK-MANUAL-NEXT:      sub rsi, 200
// CHECK-MANUAL-NEXT:      vmovapd [rdx], zmm26
// CHECK-MANUAL-NEXT:      vmovapd [rdx+64], zmm27
// CHECK-MANUAL-NEXT:      vmovapd [rdx+128], zmm28
// CHECK-MANUAL-NEXT:      vmovapd [rdx+192], zmm29
// CHECK-MANUAL-NEXT:      vmovapd [rdx+256], zmm30
// CHECK-MANUAL-NEXT:      vmovapd [rdx+320], zmm31
// CHECK-MANUAL-NEXT:      add rdx, 128
// CHECK-LIBXSMM-NEXT:     sub rdi, 3072
// CHECK-MANUAL-NEXT:      cmp r10, 16
// CHECK-MANUAL-NEXT:      jl [[SCF_M_BODY]]
// CHECK-LIBXSMM-NEXT:     add rdx, 256
// CHECK-COMPXSMM-NEXT:    sub rdi, 128
// CHECK-MANUAL-NEXT:      add rsi, 600
// CHECK-LIBXSMM-NEXT:     sub rdi, 128
// CHECK-COMPXSMM-NEXT:    add rdx, 256
// CHECK-MANUAL-NEXT:      cmp r11, 3
// CHECK-MANUAL-NEXT:      jl [[SCF_N_BODY]]
// CHECK-MANUAL-NEXT:      mov rsp, rbp
// CHECK-MANUAL-NEXT:      pop rbp
// CHECK-MANUAL-NEXT:      pop r12
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
// CHECK-NEXT:      %33 = x86.di.mov 0 : () -> !x86.reg64<r12>
// CHECK-NEXT:      x86.fallthrough ^bb3(%19 : !x86.reg64<rdi>, %20 : !x86.reg64<rsi>, %21 : !x86.reg64<rdx>, %22 : !x86.reg64<rbp>, %23 : !x86.reg64<rsp>, %24 : !x86.reg64<r11>, %26 : !x86.reg64<r10>, %27 : !x86.avx512reg<zmm26>, %28 : !x86.avx512reg<zmm27>, %29 : !x86.avx512reg<zmm28>, %30 : !x86.avx512reg<zmm29>, %31 : !x86.avx512reg<zmm30>, %32 : !x86.avx512reg<zmm31>, %33 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb3(%34: !x86.reg64<rdi>, %35: !x86.reg64<rsi>, %36: !x86.reg64<rdx>, %37: !x86.reg64<rbp>, %38: !x86.reg64<rsp>, %39: !x86.reg64<r11>, %40: !x86.reg64<r10>, %41: !x86.avx512reg<zmm26>, %42: !x86.avx512reg<zmm27>, %43: !x86.avx512reg<zmm28>, %44: !x86.avx512reg<zmm29>, %45: !x86.avx512reg<zmm30>, %46: !x86.avx512reg<zmm31>, %47: !x86.reg64<r12>):
// CHECK-NEXT:      x86.label "l35"
// CHECK-NEXT:      %48 = x86.ri.add %47, 4 : (!x86.reg64<r12>) -> !x86.reg64<r12>
// CHECK-NEXT:      %49 = x86.dm.vmovapd [%34] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %50 = x86.dm.vmovapd [%34 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %51 = x86.dm.vbroadcastsd [%35] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %52 = x86.rss.vfmadd231pd %41, %49, %51 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %53 = x86.rss.vfmadd231pd %42, %50, %51 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %54 = x86.dm.vbroadcastsd [%35 + 200] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %55 = x86.rss.vfmadd231pd %43, %49, %54 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %56 = x86.rss.vfmadd231pd %44, %50, %54 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %57 = x86.dm.vbroadcastsd [%35 + 400] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %58 = x86.ri.add %35, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %59 = x86.ri.add %34, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %60 = x86.rss.vfmadd231pd %45, %49, %57 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %61 = x86.rss.vfmadd231pd %46, %50, %57 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %62 = x86.dm.vmovapd [%59] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %63 = x86.dm.vmovapd [%59 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %64 = x86.dm.vbroadcastsd [%58] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %65 = x86.rss.vfmadd231pd %52, %62, %64 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %66 = x86.rss.vfmadd231pd %53, %63, %64 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %67 = x86.dm.vbroadcastsd [%58 + 200] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %68 = x86.rss.vfmadd231pd %55, %62, %67 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %69 = x86.rss.vfmadd231pd %56, %63, %67 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %70 = x86.dm.vbroadcastsd [%58 + 400] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %71 = x86.ri.add %58, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %72 = x86.ri.add %59, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %73 = x86.rss.vfmadd231pd %60, %62, %70 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %74 = x86.rss.vfmadd231pd %61, %63, %70 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %75 = x86.dm.vmovapd [%72] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %76 = x86.dm.vmovapd [%72 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %77 = x86.dm.vbroadcastsd [%71] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %78 = x86.rss.vfmadd231pd %65, %75, %77 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %79 = x86.rss.vfmadd231pd %66, %76, %77 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %80 = x86.dm.vbroadcastsd [%71 + 200] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %81 = x86.rss.vfmadd231pd %68, %75, %80 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %82 = x86.rss.vfmadd231pd %69, %76, %80 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %83 = x86.dm.vbroadcastsd [%71 + 400] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %84 = x86.ri.add %71, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %85 = x86.ri.add %72, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %86 = x86.rss.vfmadd231pd %73, %75, %83 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %87 = x86.rss.vfmadd231pd %74, %76, %83 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %88 = x86.dm.vmovapd [%85] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %89 = x86.dm.vmovapd [%85 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %90 = x86.dm.vbroadcastsd [%84] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %91 = x86.rss.vfmadd231pd %78, %88, %90 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %92 = x86.rss.vfmadd231pd %79, %89, %90 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %93 = x86.dm.vbroadcastsd [%84 + 200] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %94 = x86.rss.vfmadd231pd %81, %88, %93 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %95 = x86.rss.vfmadd231pd %82, %89, %93 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %96 = x86.dm.vbroadcastsd [%84 + 400] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %97 = x86.ri.add %84, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %98 = x86.ri.add %85, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %99 = x86.rss.vfmadd231pd %86, %88, %96 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %100 = x86.rss.vfmadd231pd %87, %89, %96 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %101 = x86.si.cmp %48, 24 : (!x86.reg64<r12>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %101 : !x86.rflags<rflags>, ^bb3(%98 : !x86.reg64<rdi>, %97 : !x86.reg64<rsi>, %36 : !x86.reg64<rdx>, %37 : !x86.reg64<rbp>, %38 : !x86.reg64<rsp>, %39 : !x86.reg64<r11>, %40 : !x86.reg64<r10>, %91 : !x86.avx512reg<zmm26>, %92 : !x86.avx512reg<zmm27>, %94 : !x86.avx512reg<zmm28>, %95 : !x86.avx512reg<zmm29>, %99 : !x86.avx512reg<zmm30>, %100 : !x86.avx512reg<zmm31>, %48 : !x86.reg64<r12>), ^bb4(%98 : !x86.reg64<rdi>, %97 : !x86.reg64<rsi>, %36 : !x86.reg64<rdx>, %37 : !x86.reg64<rbp>, %38 : !x86.reg64<rsp>, %39 : !x86.reg64<r11>, %40 : !x86.reg64<r10>, %91 : !x86.avx512reg<zmm26>, %92 : !x86.avx512reg<zmm27>, %94 : !x86.avx512reg<zmm28>, %95 : !x86.avx512reg<zmm29>, %99 : !x86.avx512reg<zmm30>, %100 : !x86.avx512reg<zmm31>, %48 : !x86.reg64<r12>)
// CHECK-NEXT:    ^bb4(%102: !x86.reg64<rdi>, %103: !x86.reg64<rsi>, %104: !x86.reg64<rdx>, %105: !x86.reg64<rbp>, %106: !x86.reg64<rsp>, %107: !x86.reg64<r11>, %108: !x86.reg64<r10>, %109: !x86.avx512reg<zmm26>, %110: !x86.avx512reg<zmm27>, %111: !x86.avx512reg<zmm28>, %112: !x86.avx512reg<zmm29>, %113: !x86.avx512reg<zmm30>, %114: !x86.avx512reg<zmm31>, %115: !x86.reg64<r12>):
// CHECK-NEXT:      %116 = x86.dm.vmovapd [%102] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-NEXT:      %117 = x86.dm.vmovapd [%102 + 64] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm2>
// CHECK-NEXT:      %118 = x86.dm.vbroadcastsd [%103] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %119 = x86.rss.vfmadd231pd %109, %116, %118 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:      %120 = x86.rss.vfmadd231pd %110, %117, %118 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:      %121 = x86.dm.vbroadcastsd [%103 + 200] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %122 = x86.rss.vfmadd231pd %111, %116, %121 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:      %123 = x86.rss.vfmadd231pd %112, %117, %121 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:      %124 = x86.dm.vbroadcastsd [%103 + 400] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-NEXT:      %125 = x86.ri.add %103, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %126 = x86.ri.add %102, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %127 = x86.rss.vfmadd231pd %113, %116, %124 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:      %128 = x86.rss.vfmadd231pd %114, %117, %124 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:      %129 = x86.ri.sub %125, 200 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      x86.ms.vmovapd [%104], %119 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%104 + 64], %120 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%104 + 128], %122 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%104 + 192], %123 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%104 + 256], %127 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:      x86.ms.vmovapd [%104 + 320], %128 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:      %130 = x86.ri.add %104, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %131 = x86.ri.sub %126, 3072 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %132 = x86.si.cmp %108, 16 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %132 : !x86.rflags<rflags>, ^bb2(%131 : !x86.reg64<rdi>, %129 : !x86.reg64<rsi>, %130 : !x86.reg64<rdx>, %105 : !x86.reg64<rbp>, %106 : !x86.reg64<rsp>, %107 : !x86.reg64<r11>, %108 : !x86.reg64<r10>), ^bb5(%131 : !x86.reg64<rdi>, %129 : !x86.reg64<rsi>, %130 : !x86.reg64<rdx>, %105 : !x86.reg64<rbp>, %106 : !x86.reg64<rsp>, %107 : !x86.reg64<r11>, %108 : !x86.reg64<r10>)
// CHECK-NEXT:    ^bb5(%133: !x86.reg64<rdi>, %134: !x86.reg64<rsi>, %135: !x86.reg64<rdx>, %136: !x86.reg64<rbp>, %137: !x86.reg64<rsp>, %138: !x86.reg64<r11>, %139: !x86.reg64<r10>):
// CHECK-NEXT:      %140 = x86.ri.add %135, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %141 = x86.ri.add %134, 600 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %142 = x86.ri.sub %133, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      %143 = x86.si.cmp %138, 3 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-NEXT:      x86.c.jl %143 : !x86.rflags<rflags>, ^bb1(%142 : !x86.reg64<rdi>, %141 : !x86.reg64<rsi>, %140 : !x86.reg64<rdx>, %136 : !x86.reg64<rbp>, %137 : !x86.reg64<rsp>, %138 : !x86.reg64<r11>), ^bb6(%142 : !x86.reg64<rdi>, %141 : !x86.reg64<rsi>, %140 : !x86.reg64<rdx>, %136 : !x86.reg64<rbp>, %137 : !x86.reg64<rsp>, %138 : !x86.reg64<r11>)
// CHECK-NEXT:    ^bb6(%144: !x86.reg64<rdi>, %145: !x86.reg64<rsi>, %146: !x86.reg64<rdx>, %147: !x86.reg64<rbp>, %148: !x86.reg64<rsp>, %149: !x86.reg64<r11>):
// CHECK-NEXT:      %150 = x86.ds.mov %147 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:      %151, %152 = x86.d.pop %150 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:      x86_func.ret
// CHECK-NEXT:    }
// CHECK-NEXT:  }
