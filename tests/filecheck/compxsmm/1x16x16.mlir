// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK-REGALLOC-STRUCTURE abstracts only allocatable register names. Running
// it against both pipelines verifies that register allocation does not otherwise
// change the instruction structure. CHECK-REGALLOC pins the automatic choices.

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul"(%0, %1, %2, %6, %9) <{m = 16 : i64, n = 1 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:    %15 = x86.ds.mov %13 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %16, %17 = x86.d.pop %15 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      add rax, 1
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 0
// CHECK-REGALLOC-NEXT:      cmp rax, 1
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret

// CHECK-REGALLOC-STRUCTURE:       .intel_syntax noprefix
// CHECK-REGALLOC-STRUCTURE-NEXT:  .text
// CHECK-REGALLOC-STRUCTURE-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-STRUCTURE-NEXT:  matmul_bac:
// CHECK-REGALLOC-STRUCTURE-NEXT:      push rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      push rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsp, 192
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[STACK_ALIGN:\S+]], -64
// CHECK-REGALLOC-STRUCTURE-NEXT:      and rsp, [[STACK_ALIGN]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[N:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[N]], 1
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[M:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC0:\S+]], [rdx]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC1:\S+]], [rdx+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K0:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K0:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K0:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K0]], [[B_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K0]], [[B_K0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K1:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K1:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K1:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K1]], [[B_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K1]], [[B_K1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K2:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K2:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K2:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K2]], [[B_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K2]], [[B_K2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K3:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K3:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K3:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K3]], [[B_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K3]], [[B_K3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K4:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K4:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K4:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K4]], [[B_K4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K4]], [[B_K4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K5:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K5:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K5:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K5]], [[B_K5]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K5]], [[B_K5]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K6:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K6:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K6:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K6]], [[B_K6]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K6]], [[B_K6]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K7:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K7:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K7:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K7]], [[B_K7]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K7]], [[B_K7]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K8:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K8:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K8:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K8]], [[B_K8]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K8]], [[B_K8]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K9:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K9:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K9:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K9]], [[B_K9]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K9]], [[B_K9]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K10:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K10:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K10:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K10]], [[B_K10]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K10]], [[B_K10]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K11:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K11:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K11:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K11]], [[B_K11]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K11]], [[B_K11]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K12:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K12:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K12:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K12]], [[B_K12]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K12]], [[B_K12]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K13:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K13:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K13:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K13]], [[B_K13]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K13]], [[B_K13]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K14:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K14:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K14:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K14]], [[B_K14]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K14]], [[B_K14]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0_K15:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1_K15:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vbroadcastsd [[B_K15:\S+]], [rsi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 8
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0_K15]], [[B_K15]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A1_K15]], [[B_K15]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+64], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 0
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[N]], 1
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      ret
