// RUN: compxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 64 16 64 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK-REGALLOC-STRUCTURE names logical values while abstracting their physical
// registers. CHECK-REGALLOC below pins the automatic allocator's exact choices.

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-NEXT:    %11, %12, %13, %14, %15, %16 = x86_scf.for %17 : !x86.reg64<r11>  = %10 to 3 : si32 step 3 : si32 iter_args(%18 = %0, %19 = %1, %20 = %2, %21 = %6, %22 = %9) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:      %23 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-NEXT:      %24, %25, %26, %27, %28, %29 = x86_scf.for %30 : !x86.reg64<r10>  = %23 to 16 : si32 step 16 : si32 iter_args(%31 = %18, %32 = %19, %33 = %20, %34 = %21, %35 = %22) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) {
// CHECK-NEXT:        %36 = x86.dm.vmovapd [%33] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-NEXT:        %37 = x86.dm.vmovapd [%33 + 64] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm27>
// CHECK-NEXT:        %38 = x86.dm.vmovapd [%33 + 128] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-NEXT:        %39 = x86.dm.vmovapd [%33 + 192] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm29>
// CHECK-NEXT:        %40 = x86.dm.vmovapd [%33 + 256] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-NEXT:        %41 = x86.dm.vmovapd [%33 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm31>
// CHECK-NEXT:        %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52 = "xsmm.matmul_k"(%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41) <{m_blocking = 16 : i64, n_blocking = 3 : i64, k_blocking = 64 : i64, lda = 16 : i64, ldb = 64 : i64, datatype = f64, aligned_a = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 6>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 6>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>, !x86.avx512reg<zmm26>, !x86.avx512reg<zmm27>, !x86.avx512reg<zmm28>, !x86.avx512reg<zmm29>, !x86.avx512reg<zmm30>, !x86.avx512reg<zmm31>)
// CHECK-NEXT:        %53 = x86.ri.sub %43, 512 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:        x86.ms.vmovapd [%44], %47 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%44 + 64], %48 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%44 + 128], %49 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%44 + 192], %50 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%44 + 256], %51 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-NEXT:        x86.ms.vmovapd [%44 + 320], %52 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>) -> ()
// CHECK-NEXT:        %54 = x86.ri.add %44, 128 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:        %55 = x86.ri.sub %42, 8064 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:        x86_scf.yield %55, %53, %54, %45, %46 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:      }
// CHECK-NEXT:      %56 = x86.ri.add %27, 256 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-NEXT:      %57 = x86.ri.add %26, 1536 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-NEXT:      %58 = x86.ri.sub %25, 128 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-NEXT:      x86_scf.yield %58, %57, %56, %28, %29 : !x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>
// CHECK-NEXT:    }
// CHECK-NEXT:    %59 = x86.ds.mov %15 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %60, %61 = x86.d.pop %59 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      add rax, 3
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+320]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 64
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm27
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm28
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm29
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 8064
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      add rsi, 1536
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp rax, 3
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret

// CHECK-REGALLOC-STRUCTURE:       .intel_syntax noprefix
// CHECK-REGALLOC-STRUCTURE-NEXT:  .text
// CHECK-REGALLOC-STRUCTURE-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-STRUCTURE-NEXT:  matmul_bac:
// CHECK-REGALLOC-STRUCTURE-NEXT:      push rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      push [[K:\S+]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      push rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsp, 192
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[STACK_ALIGN:\S+]], -64
// CHECK-REGALLOC-STRUCTURE-NEXT:      and rsp, [[STACK_ALIGN]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[N:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[N]], 3
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[M:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC0:\S+]], [rdx]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC1:\S+]], [rdx+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC2:\S+]], [rdx+128]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC3:\S+]], [rdx+192]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC4:\S+]], [rdx+256]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC5:\S+]], [rdx+320]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[K]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[K]], 4
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K0:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K0:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B0_K0:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K0]], [[B0_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K0]], [[B0_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B1_K0:\S+]], [rsi+512]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC2]], [[A0_K0]], [[B1_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC3]], [[A1_K0]], [[B1_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B2_K0:\S+]], [rsi+1024]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC4]], [[A0_K0]], [[B2_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC5]], [[A1_K0]], [[B2_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K1:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K1:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B0_K1:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K1]], [[B0_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K1]], [[B0_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B1_K1:\S+]], [rsi+512]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC2]], [[A0_K1]], [[B1_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC3]], [[A1_K1]], [[B1_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B2_K1:\S+]], [rsi+1024]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC4]], [[A0_K1]], [[B2_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC5]], [[A1_K1]], [[B2_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K2:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K2:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B0_K2:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K2]], [[B0_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K2]], [[B0_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B1_K2:\S+]], [rsi+512]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC2]], [[A0_K2]], [[B1_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC3]], [[A1_K2]], [[B1_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B2_K2:\S+]], [rsi+1024]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC4]], [[A0_K2]], [[B2_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC5]], [[A1_K2]], [[B2_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K3:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K3:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B0_K3:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K3]], [[B0_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K3]], [[B0_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B1_K3:\S+]], [rsi+512]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC2]], [[A0_K3]], [[B1_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC3]], [[A1_K3]], [[B1_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B2_K3:\S+]], [rsi+1024]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC4]], [[A0_K3]], [[B2_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC5]], [[A1_K3]], [[B2_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[K]], 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 512
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+64], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+128], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+192], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+256], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+320], [[ACC5]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 8064
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 256
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 1536
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[N]], 3
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop [[K]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      ret
