// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 1 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK-REGALLOC-STRUCTURE abstracts only allocatable register names. Running
// it against both pipelines verifies that register allocation does not otherwise
// change the instruction structure. CHECK-REGALLOC pins the automatic choices.

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3, %4, %5 = "xsmm.matmul"(%0, %1, %2) <{m = 16 : i64, n = 1 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm3, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm4, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm3, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm4, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm3, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm4, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm3, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm4, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm3, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm4, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 0
// CHECK-REGALLOC-NEXT:      ret

// CHECK-REGALLOC-STRUCTURE:       .intel_syntax noprefix
// CHECK-REGALLOC-STRUCTURE-NEXT:  .text
// CHECK-REGALLOC-STRUCTURE-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-STRUCTURE-NEXT:  matmul_bac:
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
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+64], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 128
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 0
// CHECK-REGALLOC-STRUCTURE-NEXT:      ret
