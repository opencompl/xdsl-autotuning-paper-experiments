// RUN: compxsmm-gemm dense %t matmul_bac 16 5 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 5 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul"(%0, %1, %2, %6, %9) <{m = 16 : i64, n = 5 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
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
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
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
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      add rdx, 512
// CHECK-REGALLOC-NEXT:      cmp rax, 5
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
