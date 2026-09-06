// RUN: compxsmm-gemm dense %t matmul_bac 16 5 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 5 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 16 : i64, n = 5 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm12, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm10, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm11, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm2
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      add rdx, 512
// CHECK-REGALLOC-NEXT:      ret
