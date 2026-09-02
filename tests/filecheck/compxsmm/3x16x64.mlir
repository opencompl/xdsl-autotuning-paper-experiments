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
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul"(%0, %1, %2, %6, %9) <{m = 16 : i64, n = 3 : i64, k = 64 : i64, lda = 16 : i64, ldb = 64 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:    %15 = x86.ds.mov %13 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %16, %17 = x86.d.pop %15 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+320]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      cmp rbx, 64
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm2
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 8064
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 1536
// CHECK-REGALLOC-NEXT:      add rdx, 256
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
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+64], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+128], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+192], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+256], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+320], [[ACC5]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 8064
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 512
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 1536
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 256
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[N]], 3
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop [[K]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      ret
