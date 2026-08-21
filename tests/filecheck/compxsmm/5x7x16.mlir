// RUN: compxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 7 5 16 7 16 7 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// The manual pipeline saves one extra GPR for mask setup, so the shared structure
// starts at frame setup and ends before register restoration. Within that range,
// named captures verify identical dataflow and instruction structure.

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 5 : si32 step 5 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 127 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %24 = x86.ks.kmovb %23 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %25 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %26, %27, %28, %29, %30, %31, %32 = x86_scf.for %33 : !x86.reg64<r10>  = %25 to 7 : si32 step 7 : si32 iter_args(%34 = %18, %35 = %19, %36 = %20, %37 = %21, %38 = %22, %39 = %24) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %40 = x86.dmk.vmovupd[%36], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %41 = x86.dmk.vmovupd[%36 + 56], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %42 = x86.dmk.vmovupd[%36 + 112], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %43 = x86.dmk.vmovupd[%36 + 168], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %44 = x86.dmk.vmovupd[%36 + 224], %39 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55 = "xsmm.matmul_k"(%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44) <{m_blocking = 7 : i64, n_blocking = 5 : i64, k_blocking = 16 : i64, lda = 7 : i64, ldb = 16 : i64, datatype = f64, aligned_a = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 5>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %56 = x86.ri.sub %46, 128 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.msk.vmovupd[%47], %51, %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%47 + 56], %52, %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%47 + 112], %53, %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%47 + 168], %54, %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        x86.msk.vmovupd[%47 + 224], %55, %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-NEXT:        %57 = x86.ri.add %47, 56 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %58 = x86.ri.sub %45, 840 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %58, %56, %57, %48, %49, %50 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %59 = x86.ri.add %29, 224 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %60 = x86.ri.add %28, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %61 = x86.ri.sub %27, 56 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %61, %60, %59, %30, %31 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %62 = x86.ds.mov %15 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %63, %64 = x86.d.pop %62 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-NEXT:      sub rsp, 192
// CHECK-REGALLOC-NEXT:      mov r10, -64
// CHECK-REGALLOC-NEXT:      and rsp, r10
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 127
// CHECK-REGALLOC-NEXT:      kmovb k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 7
// CHECK-REGALLOC-NEXT:      vmovupd zmm27 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm28 {k1}{z}, [rdx+56]
// CHECK-REGALLOC-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+112]
// CHECK-REGALLOC-NEXT:      vmovupd zmm30 {k1}{z}, [rdx+168]
// CHECK-REGALLOC-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+224]
// CHECK-REGALLOC-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-REGALLOC-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-REGALLOC-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-REGALLOC-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-REGALLOC-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-REGALLOC-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-REGALLOC-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-REGALLOC-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-REGALLOC-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-REGALLOC-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-REGALLOC-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-REGALLOC-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-REGALLOC-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-REGALLOC-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-REGALLOC-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+56]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+128]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+256]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+384]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+512]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+8]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+136]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+264]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+392]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+520]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+168]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+16]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+144]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+272]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+400]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+528]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+224]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+24]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+152]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+280]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+408]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+536]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+32]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+160]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+288]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+416]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+544]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+336]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+40]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+168]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+296]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+424]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+552]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+392]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+48]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+176]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+304]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+432]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+560]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+448]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+56]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+184]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+312]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+440]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+568]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+504]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+64]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+192]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+320]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+448]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+576]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+72]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+200]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+328]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+456]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+584]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+616]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+80]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+208]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+336]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+464]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+592]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+672]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+88]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+216]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+344]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+472]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+600]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, [rsi+96]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, [rsi+224]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, [rsi+352]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, [rsi+480]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, [rsi+608]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdi+784]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, [rsi+104]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, [rsi+232]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+360]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+488]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+616]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+112]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+240]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+368]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+496]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+624]{1to8}
// CHECK-REGALLOC-NEXT:      add rdi, 896
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, [rsi+120]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, [rsi+248]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, [rsi+376]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, [rsi+504]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, [rsi+632]{1to8}
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm22, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm23, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm24, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm25, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm26, zmm31
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm17, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm18, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm19, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm20, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm21, zmm31
// CHECK-REGALLOC-NEXT:      vaddpd zmm27, zmm12, zmm27
// CHECK-REGALLOC-NEXT:      vaddpd zmm28, zmm13, zmm28
// CHECK-REGALLOC-NEXT:      vaddpd zmm29, zmm14, zmm29
// CHECK-REGALLOC-NEXT:      vaddpd zmm30, zmm15, zmm30
// CHECK-REGALLOC-NEXT:      vaddpd zmm31, zmm16, zmm31
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovupd [rdx] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+56] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+112] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+168] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+224] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 56
// CHECK-REGALLOC-NEXT:      sub rdi, 840
// CHECK-REGALLOC-NEXT:      cmp rcx, 7
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      add rdx, 224
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      sub rdi, 56
// CHECK-REGALLOC-NEXT:      cmp rax, 5
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret

// CHECK-REGALLOC-STRUCTURE:           mov rbp, rsp
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsp, 192
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[STACK_ALIGN:\S+]], -64
// CHECK-REGALLOC-STRUCTURE-NEXT:      and rsp, [[STACK_ALIGN]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[N:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[N]], 5
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[MASK_VALUE:\S+]], 127
// CHECK-REGALLOC-STRUCTURE-NEXT:      kmovb [[MASK:\S+]], [[MASK_VALUE_32:\S+]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[M:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[M]], 7
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[ACC_G0_N0:\S+]] {[[MASK]]}{z}, [rdx]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[ACC_G0_N1:\S+]] {[[MASK]]}{z}, [rdx+56]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[ACC_G0_N2:\S+]] {[[MASK]]}{z}, [rdx+112]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[ACC_G0_N3:\S+]] {[[MASK]]}{z}, [rdx+168]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[ACC_G0_N4:\S+]] {[[MASK]]}{z}, [rdx+224]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G1_N0:\S+]], [[ACC_G1_N0]], [[ACC_G1_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G1_N1:\S+]], [[ACC_G1_N1]], [[ACC_G1_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G1_N2:\S+]], [[ACC_G1_N2]], [[ACC_G1_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G1_N3:\S+]], [[ACC_G1_N3]], [[ACC_G1_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G1_N4:\S+]], [[ACC_G1_N4]], [[ACC_G1_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G2_N0:\S+]], [[ACC_G2_N0]], [[ACC_G2_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G2_N1:\S+]], [[ACC_G2_N1]], [[ACC_G2_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G2_N2:\S+]], [[ACC_G2_N2]], [[ACC_G2_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G2_N3:\S+]], [[ACC_G2_N3]], [[ACC_G2_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G2_N4:\S+]], [[ACC_G2_N4]], [[ACC_G2_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G3_N0:\S+]], [[ACC_G3_N0]], [[ACC_G3_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G3_N1:\S+]], [[ACC_G3_N1]], [[ACC_G3_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G3_N2:\S+]], [[ACC_G3_N2]], [[ACC_G3_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G3_N3:\S+]], [[ACC_G3_N3]], [[ACC_G3_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC_G3_N4:\S+]], [[ACC_G3_N4]], [[ACC_G3_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K0:\S+]] {[[MASK]]}{z}, [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K1:\S+]] {[[MASK]]}{z}, [rdi+56]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N0]], [[A_K0]], [rsi]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N1]], [[A_K0]], [rsi+128]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N2]], [[A_K0]], [rsi+256]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N3]], [[A_K0]], [rsi+384]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N4]], [[A_K0]], [rsi+512]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K2:\S+]] {[[MASK]]}{z}, [rdi+112]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N0]], [[A_K1]], [rsi+8]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N1]], [[A_K1]], [rsi+136]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N2]], [[A_K1]], [rsi+264]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N3]], [[A_K1]], [rsi+392]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N4]], [[A_K1]], [rsi+520]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K3:\S+]] {[[MASK]]}{z}, [rdi+168]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N0]], [[A_K2]], [rsi+16]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N1]], [[A_K2]], [rsi+144]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N2]], [[A_K2]], [rsi+272]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N3]], [[A_K2]], [rsi+400]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N4]], [[A_K2]], [rsi+528]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K4:\S+]] {[[MASK]]}{z}, [rdi+224]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N0]], [[A_K3]], [rsi+24]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N1]], [[A_K3]], [rsi+152]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N2]], [[A_K3]], [rsi+280]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N3]], [[A_K3]], [rsi+408]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N4]], [[A_K3]], [rsi+536]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K5:\S+]] {[[MASK]]}{z}, [rdi+280]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N0]], [[A_K4]], [rsi+32]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N1]], [[A_K4]], [rsi+160]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N2]], [[A_K4]], [rsi+288]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N3]], [[A_K4]], [rsi+416]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N4]], [[A_K4]], [rsi+544]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K6:\S+]] {[[MASK]]}{z}, [rdi+336]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N0]], [[A_K5]], [rsi+40]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N1]], [[A_K5]], [rsi+168]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N2]], [[A_K5]], [rsi+296]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N3]], [[A_K5]], [rsi+424]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N4]], [[A_K5]], [rsi+552]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K7:\S+]] {[[MASK]]}{z}, [rdi+392]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N0]], [[A_K6]], [rsi+48]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N1]], [[A_K6]], [rsi+176]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N2]], [[A_K6]], [rsi+304]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N3]], [[A_K6]], [rsi+432]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N4]], [[A_K6]], [rsi+560]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K8:\S+]] {[[MASK]]}{z}, [rdi+448]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N0]], [[A_K7]], [rsi+56]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N1]], [[A_K7]], [rsi+184]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N2]], [[A_K7]], [rsi+312]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N3]], [[A_K7]], [rsi+440]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N4]], [[A_K7]], [rsi+568]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K9:\S+]] {[[MASK]]}{z}, [rdi+504]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N0]], [[A_K8]], [rsi+64]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N1]], [[A_K8]], [rsi+192]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N2]], [[A_K8]], [rsi+320]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N3]], [[A_K8]], [rsi+448]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N4]], [[A_K8]], [rsi+576]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K10:\S+]] {[[MASK]]}{z}, [rdi+560]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N0]], [[A_K9]], [rsi+72]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N1]], [[A_K9]], [rsi+200]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N2]], [[A_K9]], [rsi+328]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N3]], [[A_K9]], [rsi+456]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N4]], [[A_K9]], [rsi+584]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K11:\S+]] {[[MASK]]}{z}, [rdi+616]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N0]], [[A_K10]], [rsi+80]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N1]], [[A_K10]], [rsi+208]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N2]], [[A_K10]], [rsi+336]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N3]], [[A_K10]], [rsi+464]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N4]], [[A_K10]], [rsi+592]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K12:\S+]] {[[MASK]]}{z}, [rdi+672]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N0]], [[A_K11]], [rsi+88]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N1]], [[A_K11]], [rsi+216]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N2]], [[A_K11]], [rsi+344]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N3]], [[A_K11]], [rsi+472]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N4]], [[A_K11]], [rsi+600]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K13:\S+]] {[[MASK]]}{z}, [rdi+728]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N0]], [[A_K12]], [rsi+96]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N1]], [[A_K12]], [rsi+224]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N2]], [[A_K12]], [rsi+352]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N3]], [[A_K12]], [rsi+480]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G0_N4]], [[A_K12]], [rsi+608]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K14:\S+]] {[[MASK]]}{z}, [rdi+784]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N0]], [[A_K13]], [rsi+104]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N1]], [[A_K13]], [rsi+232]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N2]], [[A_K13]], [rsi+360]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N3]], [[A_K13]], [rsi+488]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G1_N4]], [[A_K13]], [rsi+616]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [[A_K15:\S+]] {[[MASK]]}{z}, [rdi+840]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N0]], [[A_K14]], [rsi+112]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N1]], [[A_K14]], [rsi+240]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N2]], [[A_K14]], [rsi+368]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N3]], [[A_K14]], [rsi+496]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G2_N4]], [[A_K14]], [rsi+624]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 896
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N0]], [[A_K15]], [rsi+120]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N1]], [[A_K15]], [rsi+248]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N2]], [[A_K15]], [rsi+376]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N3]], [[A_K15]], [rsi+504]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC_G3_N4]], [[A_K15]], [rsi+632]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N0]], [[ACC_G1_N0]], [[ACC_G0_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N1]], [[ACC_G1_N1]], [[ACC_G0_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N2]], [[ACC_G1_N2]], [[ACC_G0_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N3]], [[ACC_G1_N3]], [[ACC_G0_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N4]], [[ACC_G1_N4]], [[ACC_G0_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N0]], [[ACC_G2_N0]], [[ACC_G0_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N1]], [[ACC_G2_N1]], [[ACC_G0_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N2]], [[ACC_G2_N2]], [[ACC_G0_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N3]], [[ACC_G2_N3]], [[ACC_G0_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N4]], [[ACC_G2_N4]], [[ACC_G0_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N0]], [[ACC_G3_N0]], [[ACC_G0_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N1]], [[ACC_G3_N1]], [[ACC_G0_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N2]], [[ACC_G3_N2]], [[ACC_G0_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N3]], [[ACC_G3_N3]], [[ACC_G0_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC_G0_N4]], [[ACC_G3_N4]], [[ACC_G0_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [rdx] {[[MASK]]}, [[ACC_G0_N0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [rdx+56] {[[MASK]]}, [[ACC_G0_N1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [rdx+112] {[[MASK]]}, [[ACC_G0_N2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [rdx+168] {[[MASK]]}, [[ACC_G0_N3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovupd [rdx+224] {[[MASK]]}, [[ACC_G0_N4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 56
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 840
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[M]], 7
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 224
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 640
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 56
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[N]], 5
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rsp, rbp
