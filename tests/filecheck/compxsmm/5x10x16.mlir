// RUN: compxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 10 : i64, n = 5 : i64, k = 16 : i64, lda = 10 : i64, ldb = 16 : i64, ldc = 10 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n"} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// CHECK-REGALLOC:       .intel_syntax noprefix
// CHECK-REGALLOC-NEXT:  .text
// CHECK-REGALLOC-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-NEXT:  matmul_bac:
// CHECK-REGALLOC-NEXT:      mov rax, 3
// CHECK-REGALLOC-NEXT:      kmovb k1, eax
// CHECK-REGALLOC-NEXT:      vmovupd zmm9, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm8 {k1}{z}, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm7, [rdx+80]
// CHECK-REGALLOC-NEXT:      vmovupd zmm6 {k1}{z}, [rdx+144]
// CHECK-REGALLOC-NEXT:      vmovupd zmm5, [rdx+160]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4 {k1}{z}, [rdx+224]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdx+240]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2 {k1}{z}, [rdx+304]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovupd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm11 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovupd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovupd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm11 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovupd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovupd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm11 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovupd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovupd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm11 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovupd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovupd zmm10, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm11 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm10, zmm12
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm11, zmm12
// CHECK-REGALLOC-NEXT:      vmovupd zmm11, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm11, zmm10
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm12, zmm10
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10 {k1}{z}, [rdi+64]
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
// CHECK-REGALLOC-NEXT:      add rdi, 80
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm12, zmm11
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm10, zmm11
// CHECK-REGALLOC-NEXT:      vmovupd [rdx], zmm9
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+64] {k1}, zmm8
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+80], zmm7
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+144] {k1}, zmm6
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+160], zmm5
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+224] {k1}, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+240], zmm3
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+304] {k1}, zmm2
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+320], zmm1
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+384] {k1}, zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 1200
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 80
// CHECK-REGALLOC-NEXT:      sub rdi, 80
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      add rdx, 320
// CHECK-REGALLOC-NEXT:      ret
