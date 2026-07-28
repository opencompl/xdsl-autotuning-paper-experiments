// RUN: libxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM
// RUN: compxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r15, 127
// CHECK-NEXT:      kmovb k1, r15d
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 7
// CHECK-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+56]
// CHECK-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+112]
// CHECK-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+168]
// CHECK-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+224]
// CHECK-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi]
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+56]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+128]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+256]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+384]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+512]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+112]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+8]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+136]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+264]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+392]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+520]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+168]
// CHECK-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+16]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+144]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+272]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+400]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+528]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+224]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+24]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+152]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+280]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+408]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+536]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+280]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+32]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+160]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+288]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+416]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+544]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+336]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+40]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+168]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+296]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+424]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+552]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+392]
// CHECK-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+48]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+176]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+304]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+432]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+560]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+448]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+56]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+184]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+312]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+440]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+568]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+504]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+64]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+192]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+320]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+448]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+576]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+560]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+72]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+200]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+328]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+456]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+584]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+616]
// CHECK-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+80]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+208]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+336]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+464]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+592]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+672]
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+88]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+216]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+344]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+472]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+600]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+728]
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+96]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+224]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+352]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+480]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+608]{1to8}
// CHECK-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+784]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+104]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+232]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, [rsi+360]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm1, [rsi+488]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, [rsi+616]{1to8}
// CHECK-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+840]
// CHECK-NEXT:      vfmadd231pd zmm17, zmm0, [rsi+112]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm0, [rsi+240]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm0, [rsi+368]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm0, [rsi+496]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm0, [rsi+624]{1to8}
// CHECK-NEXT:      add rdi, 896
// CHECK-NEXT:      vfmadd231pd zmm12, zmm1, [rsi+120]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm13, zmm1, [rsi+248]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm14, zmm1, [rsi+376]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm15, zmm1, [rsi+504]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+632]{1to8}
// CHECK-NEXT:      add rsi, 128
// CHECK-NEXT:      vaddpd zmm27, zmm22, zmm27
// CHECK-NEXT:      vaddpd zmm28, zmm23, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm24, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm25, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm26, zmm31
// CHECK-NEXT:      vaddpd zmm27, zmm17, zmm27
// CHECK-NEXT:      vaddpd zmm28, zmm18, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm19, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm20, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm21, zmm31
// CHECK-NEXT:      vaddpd zmm27, zmm12, zmm27
// CHECK-NEXT:      vaddpd zmm28, zmm13, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm14, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm15, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm16, zmm31
// CHECK-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-NEXT:      vmovupd [rdx+56] {k1}, zmm28
// CHECK-NEXT:      vmovupd [rdx+112] {k1}, zmm29
// CHECK-NEXT:      vmovupd [rdx+168] {k1}, zmm30
// CHECK-NEXT:      vmovupd [rdx+224] {k1}, zmm31
// CHECK-NEXT:      add rdx, 56
// CHECK-NEXT:      sub rdi, 840
// CHECK-NEXT:      cmp r10, 7
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      add rdx, 224
// CHECK-NEXT:      add rsi, 640
// CHECK-NEXT:      sub rdi, 56
// CHECK-NEXT:      cmp r11, 5
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r15
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
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb0(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb0(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l33"
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 127 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %19 = x86.ks.kmovb %18 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %20 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %20 : !x86.reg64<r10>, %19 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%21: !x86.reg64<rdi>, %22: !x86.reg64<rsi>, %23: !x86.reg64<rdx>, %24: !x86.reg64<rbp>, %25: !x86.reg64<rsp>, %26: !x86.reg64<r11>, %27: !x86.reg64<r10>, %28: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %29 = x86.ri.add %27, 7 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.dmk.vmovupd[%23], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dmk.vmovupd[%23 + 56], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dmk.vmovupd[%23 + 112], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dmk.vmovupd[%23 + 168], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dmk.vmovupd[%23 + 224], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.get_avx_register : !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dss.vpxord %35, %35 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm22>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.get_avx_register : !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dss.vpxord %37, %37 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm23>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.get_avx_register : !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dss.vpxord %39, %39 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm24>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.get_avx_register : !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dss.vpxord %41, %41 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm25>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.get_avx_register : !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.dss.vpxord %43, %43 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm26>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.get_avx_register : !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dss.vpxord %45, %45 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm17>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.get_avx_register : !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.dss.vpxord %47, %47 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm18>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.get_avx_register : !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.dss.vpxord %49, %49 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm19>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.get_avx_register : !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %52 = x86.dss.vpxord %51, %51 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm20>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %53 = x86.get_avx_register : !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %54 = x86.dss.vpxord %53, %53 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm21>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %55 = x86.get_avx_register : !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %56 = x86.dss.vpxord %55, %55 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm12>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %57 = x86.get_avx_register : !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %58 = x86.dss.vpxord %57, %57 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm13>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %59 = x86.get_avx_register : !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %60 = x86.dss.vpxord %59, %59 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm14>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %61 = x86.get_avx_register : !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %62 = x86.dss.vpxord %61, %61 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm15>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %63 = x86.get_avx_register : !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %64 = x86.dss.vpxord %63, %63 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm16>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %65 = x86.dmk.vmovupd[%21], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %66 = x86.dmk.vmovupd[%21 + 56], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %67 = x86.rsm.vfmadd231pd %30, %65, [%22] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %68 = x86.rsm.vfmadd231pd %31, %65, [%22 + 128] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %69 = x86.rsm.vfmadd231pd %32, %65, [%22 + 256] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %70 = x86.rsm.vfmadd231pd %33, %65, [%22 + 384] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %71 = x86.rsm.vfmadd231pd %34, %65, [%22 + 512] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %72 = x86.dmk.vmovupd[%21 + 112], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %73 = x86.rsm.vfmadd231pd %36, %66, [%22 + 8] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %74 = x86.rsm.vfmadd231pd %38, %66, [%22 + 136] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %75 = x86.rsm.vfmadd231pd %40, %66, [%22 + 264] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %76 = x86.rsm.vfmadd231pd %42, %66, [%22 + 392] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %77 = x86.rsm.vfmadd231pd %44, %66, [%22 + 520] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %78 = x86.dmk.vmovupd[%21 + 168], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %79 = x86.rsm.vfmadd231pd %46, %72, [%22 + 16] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %80 = x86.rsm.vfmadd231pd %48, %72, [%22 + 144] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %81 = x86.rsm.vfmadd231pd %50, %72, [%22 + 272] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %82 = x86.rsm.vfmadd231pd %52, %72, [%22 + 400] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %83 = x86.rsm.vfmadd231pd %54, %72, [%22 + 528] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %84 = x86.dmk.vmovupd[%21 + 224], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %85 = x86.rsm.vfmadd231pd %56, %78, [%22 + 24] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %86 = x86.rsm.vfmadd231pd %58, %78, [%22 + 152] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %87 = x86.rsm.vfmadd231pd %60, %78, [%22 + 280] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %88 = x86.rsm.vfmadd231pd %62, %78, [%22 + 408] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %89 = x86.rsm.vfmadd231pd %64, %78, [%22 + 536] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %90 = x86.dmk.vmovupd[%21 + 280], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %91 = x86.rsm.vfmadd231pd %67, %84, [%22 + 32] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %92 = x86.rsm.vfmadd231pd %68, %84, [%22 + 160] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %93 = x86.rsm.vfmadd231pd %69, %84, [%22 + 288] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %94 = x86.rsm.vfmadd231pd %70, %84, [%22 + 416] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %95 = x86.rsm.vfmadd231pd %71, %84, [%22 + 544] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %96 = x86.dmk.vmovupd[%21 + 336], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %97 = x86.rsm.vfmadd231pd %73, %90, [%22 + 40] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %98 = x86.rsm.vfmadd231pd %74, %90, [%22 + 168] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %99 = x86.rsm.vfmadd231pd %75, %90, [%22 + 296] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %100 = x86.rsm.vfmadd231pd %76, %90, [%22 + 424] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %101 = x86.rsm.vfmadd231pd %77, %90, [%22 + 552] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %102 = x86.dmk.vmovupd[%21 + 392], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %103 = x86.rsm.vfmadd231pd %79, %96, [%22 + 48] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %104 = x86.rsm.vfmadd231pd %80, %96, [%22 + 176] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %105 = x86.rsm.vfmadd231pd %81, %96, [%22 + 304] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %106 = x86.rsm.vfmadd231pd %82, %96, [%22 + 432] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %107 = x86.rsm.vfmadd231pd %83, %96, [%22 + 560] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %108 = x86.dmk.vmovupd[%21 + 448], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %109 = x86.rsm.vfmadd231pd %85, %102, [%22 + 56] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %110 = x86.rsm.vfmadd231pd %86, %102, [%22 + 184] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %111 = x86.rsm.vfmadd231pd %87, %102, [%22 + 312] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %112 = x86.rsm.vfmadd231pd %88, %102, [%22 + 440] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %113 = x86.rsm.vfmadd231pd %89, %102, [%22 + 568] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %114 = x86.dmk.vmovupd[%21 + 504], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %115 = x86.rsm.vfmadd231pd %91, %108, [%22 + 64] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.rsm.vfmadd231pd %92, %108, [%22 + 192] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.rsm.vfmadd231pd %93, %108, [%22 + 320] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.rsm.vfmadd231pd %94, %108, [%22 + 448] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.rsm.vfmadd231pd %95, %108, [%22 + 576] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.dmk.vmovupd[%21 + 560], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.rsm.vfmadd231pd %97, %114, [%22 + 72] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.rsm.vfmadd231pd %98, %114, [%22 + 200] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.rsm.vfmadd231pd %99, %114, [%22 + 328] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.rsm.vfmadd231pd %100, %114, [%22 + 456] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %125 = x86.rsm.vfmadd231pd %101, %114, [%22 + 584] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %126 = x86.dmk.vmovupd[%21 + 616], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %127 = x86.rsm.vfmadd231pd %103, %120, [%22 + 80] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %128 = x86.rsm.vfmadd231pd %104, %120, [%22 + 208] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %129 = x86.rsm.vfmadd231pd %105, %120, [%22 + 336] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %130 = x86.rsm.vfmadd231pd %106, %120, [%22 + 464] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %131 = x86.rsm.vfmadd231pd %107, %120, [%22 + 592] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %132 = x86.dmk.vmovupd[%21 + 672], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %133 = x86.rsm.vfmadd231pd %109, %126, [%22 + 88] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %134 = x86.rsm.vfmadd231pd %110, %126, [%22 + 216] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %135 = x86.rsm.vfmadd231pd %111, %126, [%22 + 344] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %136 = x86.rsm.vfmadd231pd %112, %126, [%22 + 472] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %137 = x86.rsm.vfmadd231pd %113, %126, [%22 + 600] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %138 = x86.dmk.vmovupd[%21 + 728], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %139 = x86.rsm.vfmadd231pd %115, %132, [%22 + 96] {broadcast} : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %140 = x86.rsm.vfmadd231pd %116, %132, [%22 + 224] {broadcast} : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %141 = x86.rsm.vfmadd231pd %117, %132, [%22 + 352] {broadcast} : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %142 = x86.rsm.vfmadd231pd %118, %132, [%22 + 480] {broadcast} : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %143 = x86.rsm.vfmadd231pd %119, %132, [%22 + 608] {broadcast} : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %144 = x86.dmk.vmovupd[%21 + 784], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %145 = x86.rsm.vfmadd231pd %121, %138, [%22 + 104] {broadcast} : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %146 = x86.rsm.vfmadd231pd %122, %138, [%22 + 232] {broadcast} : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %147 = x86.rsm.vfmadd231pd %123, %138, [%22 + 360] {broadcast} : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %148 = x86.rsm.vfmadd231pd %124, %138, [%22 + 488] {broadcast} : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %149 = x86.rsm.vfmadd231pd %125, %138, [%22 + 616] {broadcast} : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %150 = x86.dmk.vmovupd[%21 + 840], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %151 = x86.rsm.vfmadd231pd %127, %144, [%22 + 112] {broadcast} : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm17>
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.rsm.vfmadd231pd %128, %144, [%22 + 240] {broadcast} : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm18>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.rsm.vfmadd231pd %129, %144, [%22 + 368] {broadcast} : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm19>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.rsm.vfmadd231pd %130, %144, [%22 + 496] {broadcast} : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm20>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.rsm.vfmadd231pd %131, %144, [%22 + 624] {broadcast} : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm0>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm21>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.ri.add %21, 896 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.rsm.vfmadd231pd %133, %150, [%22 + 120] {broadcast} : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm12>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.rsm.vfmadd231pd %134, %150, [%22 + 248] {broadcast} : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm13>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.rsm.vfmadd231pd %135, %150, [%22 + 376] {broadcast} : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm14>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.rsm.vfmadd231pd %136, %150, [%22 + 504] {broadcast} : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm15>
// CHECK-IR-LIBXSMM-NEXT:      %161 = x86.rsm.vfmadd231pd %137, %150, [%22 + 632] {broadcast} : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm1>, !x86.reg64<rsi>) -> !x86.avx512reg<zmm16>
// CHECK-IR-LIBXSMM-NEXT:      %162 = x86.ri.add %22, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %163 = x86.dss.vaddpd %145, %139 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %164 = x86.dss.vaddpd %146, %140 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %165 = x86.dss.vaddpd %147, %141 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %166 = x86.dss.vaddpd %148, %142 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %167 = x86.dss.vaddpd %149, %143 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %168 = x86.dss.vaddpd %151, %163 : (!x86.avx512reg<zmm17>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %169 = x86.dss.vaddpd %152, %164 : (!x86.avx512reg<zmm18>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %170 = x86.dss.vaddpd %153, %165 : (!x86.avx512reg<zmm19>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %171 = x86.dss.vaddpd %154, %166 : (!x86.avx512reg<zmm20>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %172 = x86.dss.vaddpd %155, %167 : (!x86.avx512reg<zmm21>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %173 = x86.dss.vaddpd %157, %168 : (!x86.avx512reg<zmm12>, !x86.avx512reg<zmm27>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %174 = x86.dss.vaddpd %158, %169 : (!x86.avx512reg<zmm13>, !x86.avx512reg<zmm28>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %175 = x86.dss.vaddpd %159, %170 : (!x86.avx512reg<zmm14>, !x86.avx512reg<zmm29>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %176 = x86.dss.vaddpd %160, %171 : (!x86.avx512reg<zmm15>, !x86.avx512reg<zmm30>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %177 = x86.dss.vaddpd %161, %172 : (!x86.avx512reg<zmm16>, !x86.avx512reg<zmm31>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23], %173, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 56], %174, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 112], %175, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 168], %176, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 224], %177, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %178 = x86.ri.add %23, 56 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %179 = x86.ri.sub %156, 840 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %180 = x86.si.cmp %29, 7 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %180 : !x86.rflags<rflags>, ^bb1(%179 : !x86.reg64<rdi>, %162 : !x86.reg64<rsi>, %178 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %29 : !x86.reg64<r10>, %28 : !x86.avx512maskreg<k1>), ^bb2(%179 : !x86.reg64<rdi>, %162 : !x86.reg64<rsi>, %178 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %29 : !x86.reg64<r10>, %28 : !x86.avx512maskreg<k1>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%181: !x86.reg64<rdi>, %182: !x86.reg64<rsi>, %183: !x86.reg64<rdx>, %184: !x86.reg64<rbp>, %185: !x86.reg64<rsp>, %186: !x86.reg64<r11>, %187: !x86.reg64<r10>, %188: !x86.avx512maskreg<k1>):
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.ri.add %183, 224 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.ri.add %182, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.ri.sub %181, 56 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.si.cmp %186, 5 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %192 : !x86.rflags<rflags>, ^bb0(%191 : !x86.reg64<rdi>, %190 : !x86.reg64<rsi>, %189 : !x86.reg64<rdx>, %184 : !x86.reg64<rbp>, %185 : !x86.reg64<rsp>, %186 : !x86.reg64<r11>), ^bb3(%191 : !x86.reg64<rdi>, %190 : !x86.reg64<rsi>, %189 : !x86.reg64<rdx>, %184 : !x86.reg64<rbp>, %185 : !x86.reg64<rsp>, %186 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%193: !x86.reg64<rdi>, %194: !x86.reg64<rsi>, %195: !x86.reg64<rdx>, %196: !x86.reg64<rbp>, %197: !x86.reg64<rsp>, %198: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %199 = x86.ds.mov %196 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %200, %201 = x86.d.pop %199 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }
// CHECK-IR-LIBXSMM-NEXT:  
