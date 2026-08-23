// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 18 : si32 step 6 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %24 = x86.ks.kmovw %23 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %25 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %26, %27, %28, %29, %30, %31, %32 = x86_scf.for %33 : !x86.reg64<r10>  = %25 to 50 : si32 step 50 : si32 iter_args(%34 = %18, %35 = %19, %36 = %20, %37 = %21, %38 = %22, %39 = %24) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %40, %41, %42, %43, %44, %45 = "xsmm.matmul_m"(%34, %35, %36, %37, %38, %39) <{m_blocking = 50 : i64, n_blocking = 6 : i64, k = 128 : i64, lda = 50 : i64, ldb = 128 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>)
// CHECK-NEXT:        x86_scf.yield %40, %41, %42, %43, %44, %45 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %46 = x86.ri.add %29, 1000 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %47 = x86.ri.add %28, 3072 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %48 = x86.ri.sub %27, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %48, %47, %46, %30, %31 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %49 = x86.di.mov 18 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %50, %51, %52, %53, %54, %55 = x86_scf.for %56 : !x86.reg64<r11>  = %49 to 38 : si32 step 5 : si32 iter_args(%57 = %12, %58 = %13, %59 = %14, %60 = %15, %61 = %16) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %62 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-NEXT:      %63 = x86.ks.kmovw %62 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-NEXT:      %64 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %65, %66, %67, %68, %69, %70, %71 = x86_scf.for %72 : !x86.reg64<r10>  = %64 to 50 : si32 step 50 : si32 iter_args(%73 = %57, %74 = %58, %75 = %59, %76 = %60, %77 = %61, %78 = %63) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) {
// CHECK-NEXT:        %79, %80, %81, %82, %83, %84 = "xsmm.matmul_m"(%73, %74, %75, %76, %77, %78) <{m_blocking = 50 : i64, n_blocking = 5 : i64, k = 128 : i64, lda = 50 : i64, ldb = 128 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>)
// CHECK-NEXT:        x86_scf.yield %79, %80, %81, %82, %83, %84 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512maskreg<k1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %85 = x86.ri.add %68, 800 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %86 = x86.ri.add %67, 2560 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %87 = x86.ri.sub %66, 200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %87, %86, %85, %69, %70 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %88 = x86.ds.mov %54 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %89, %90 = x86.d.pop %88 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      push rbx
// CHECK-REGALLOC-NEXT:      push rbp
// CHECK-REGALLOC-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-NEXT:      sub rsp, 192
// CHECK-REGALLOC-NEXT:      mov r10, -64
// CHECK-REGALLOC-NEXT:      and rsp, r10
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 50
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm11 {k1}{z}, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+200]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+264]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+328]
// CHECK-REGALLOC-NEXT:      vmovups zmm15 {k1}{z}, [rdx+392]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+528]
// CHECK-REGALLOC-NEXT:      vmovups zmm19 {k1}{z}, [rdx+592]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+600]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+664]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+728]
// CHECK-REGALLOC-NEXT:      vmovups zmm23 {k1}{z}, [rdx+792]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+800]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+864]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+928]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx+992]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+1000]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+1064]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+1128]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1192]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192] {k1}, zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+200], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+264], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+328], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+392] {k1}, zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+400], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+464], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+528], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+592] {k1}, zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+600], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+664], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+728], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+792] {k1}, zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+800], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+864], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+928], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+992] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1000], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1064], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1128], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1192] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 200
// CHECK-REGALLOC-NEXT:      sub rdi, 25400
// CHECK-REGALLOC-NEXT:      cmp rcx, 50
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 1000
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      sub rdi, 200
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 18
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 50
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm15 {k1}{z}, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+200]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+264]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+328]
// CHECK-REGALLOC-NEXT:      vmovups zmm19 {k1}{z}, [rdx+392]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+528]
// CHECK-REGALLOC-NEXT:      vmovups zmm23 {k1}{z}, [rdx+592]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+600]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+664]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+728]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx+792]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+800]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+864]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+928]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+992]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192] {k1}, zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+200], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+264], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+328], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+392] {k1}, zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+400], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+464], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+528], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+592] {k1}, zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+600], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+664], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+728], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+792] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+800], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+864], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+928], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+992] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 200
// CHECK-REGALLOC-NEXT:      sub rdi, 25400
// CHECK-REGALLOC-NEXT:      cmp rcx, 50
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      add rdx, 800
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      sub rdi, 200
// CHECK-REGALLOC-NEXT:      cmp rax, 38
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
