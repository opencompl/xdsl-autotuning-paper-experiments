// RUN: compxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 3 8 16 8 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 16 : i64, n = 3 : i64, k = 8 : i64, lda = 16 : i64, ldb = 8 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm8, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm6, zmm8
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm7, zmm8
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm6, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm7, zmm6
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm8, zmm6
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+64]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm7, [rsi+128]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm8, zmm7
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm6, zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm2
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 896
// CHECK-REGALLOC-NEXT:      sub rsi, 64
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 192
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      ret
