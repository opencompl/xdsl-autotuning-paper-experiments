// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 34 5 16 34 16 34 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul"(%0, %1, %2, %6, %9) <{m = 34 : i64, n = 5 : i64, k = 16 : i64, lda = 34 : i64, ldb = 16 : i64, ldc = 34 : i64, datatype = f64, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
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
// CHECK-REGALLOC-NEXT:      vmovupd zmm5, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm6, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm7, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm8, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovupd zmm9, [rdx+272]
// CHECK-REGALLOC-NEXT:      vmovupd zmm10, [rdx+336]
// CHECK-REGALLOC-NEXT:      vmovupd zmm11, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovupd zmm12, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovupd zmm13, [rdx+544]
// CHECK-REGALLOC-NEXT:      vmovupd zmm14, [rdx+608]
// CHECK-REGALLOC-NEXT:      vmovupd zmm15, [rdx+672]
// CHECK-REGALLOC-NEXT:      vmovupd zmm16, [rdx+736]
// CHECK-REGALLOC-NEXT:      vmovupd zmm17, [rdx+816]
// CHECK-REGALLOC-NEXT:      vmovupd zmm18, [rdx+880]
// CHECK-REGALLOC-NEXT:      vmovupd zmm19, [rdx+944]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0, [rdx+1008]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovupd zmm4, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vmovupd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovupd zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovupd zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovupd zmm21, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 272
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vmovupd [rdx], zmm5
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+64], zmm6
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+128], zmm7
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+192], zmm8
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+272], zmm9
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+336], zmm10
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+400], zmm11
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+464], zmm12
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+544], zmm13
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+608], zmm14
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+672], zmm15
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+736], zmm16
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+816], zmm17
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+880], zmm18
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+944], zmm19
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1008], zmm0
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1088], zmm1
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1152], zmm2
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1216], zmm3
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1280], zmm4
// CHECK-REGALLOC-NEXT:      sub rdi, 4096
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      mov rax, 3
// CHECK-REGALLOC-NEXT:      kmovb k1, eax
// CHECK-REGALLOC-NEXT:      vmovupd zmm4 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovupd zmm3 {k1}{z}, [rdx+272]
// CHECK-REGALLOC-NEXT:      vmovupd zmm2 {k1}{z}, [rdx+544]
// CHECK-REGALLOC-NEXT:      vmovupd zmm1 {k1}{z}, [rdx+816]
// CHECK-REGALLOC-NEXT:      vmovupd zmm0 {k1}{z}, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-REGALLOC-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-REGALLOC-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-REGALLOC-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-REGALLOC-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-REGALLOC-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-REGALLOC-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-REGALLOC-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-REGALLOC-NEXT:      vpxord zmm11, zmm11, zmm11
// CHECK-REGALLOC-NEXT:      vpxord zmm10, zmm10, zmm10
// CHECK-REGALLOC-NEXT:      vpxord zmm9, zmm9, zmm9
// CHECK-REGALLOC-NEXT:      vpxord zmm8, zmm8, zmm8
// CHECK-REGALLOC-NEXT:      vpxord zmm7, zmm7, zmm7
// CHECK-REGALLOC-NEXT:      vpxord zmm6, zmm6, zmm6
// CHECK-REGALLOC-NEXT:      vpxord zmm5, zmm5, zmm5
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+272]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, [rsi]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, [rsi+128]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, [rsi+256]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, [rsi+384]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, [rsi+512]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+544]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm20, [rsi+8]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, [rsi+136]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, [rsi+264]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, [rsi+392]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, [rsi+520]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+816]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, [rsi+16]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, [rsi+144]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, [rsi+272]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, [rsi+400]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, [rsi+528]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+1088]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, [rsi+24]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, [rsi+152]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, [rsi+280]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, [rsi+408]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, [rsi+536]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+1360]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, [rsi+32]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, [rsi+160]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, [rsi+288]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, [rsi+416]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, [rsi+544]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+1632]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm20, [rsi+40]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, [rsi+168]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, [rsi+296]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, [rsi+424]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, [rsi+552]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+1904]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, [rsi+48]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, [rsi+176]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, [rsi+304]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, [rsi+432]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, [rsi+560]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+2176]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, [rsi+56]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, [rsi+184]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, [rsi+312]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, [rsi+440]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, [rsi+568]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+2448]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, [rsi+64]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, [rsi+192]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, [rsi+320]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, [rsi+448]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, [rsi+576]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+2720]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm20, [rsi+72]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, [rsi+200]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, [rsi+328]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, [rsi+456]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, [rsi+584]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+2992]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, [rsi+80]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, [rsi+208]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, [rsi+336]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, [rsi+464]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, [rsi+592]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+3264]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, [rsi+88]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, [rsi+216]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, [rsi+344]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, [rsi+472]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, [rsi+600]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+3536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, [rsi+96]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, [rsi+224]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, [rsi+352]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, [rsi+480]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, [rsi+608]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm21 {k1}{z}, [rdi+3808]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm20, [rsi+104]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm20, [rsi+232]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, [rsi+360]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, [rsi+488]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, [rsi+616]{1to8}
// CHECK-REGALLOC-NEXT:      vmovupd zmm20 {k1}{z}, [rdi+4080]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, [rsi+112]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, [rsi+240]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, [rsi+368]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, [rsi+496]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, [rsi+624]{1to8}
// CHECK-REGALLOC-NEXT:      add rdi, 4352
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, [rsi+120]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, [rsi+248]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, [rsi+376]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, [rsi+504]{1to8}
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, [rsi+632]{1to8}
// CHECK-REGALLOC-NEXT:      add rsi, 128
// CHECK-REGALLOC-NEXT:      vaddpd zmm4, zmm19, zmm4
// CHECK-REGALLOC-NEXT:      vaddpd zmm3, zmm18, zmm3
// CHECK-REGALLOC-NEXT:      vaddpd zmm2, zmm17, zmm2
// CHECK-REGALLOC-NEXT:      vaddpd zmm1, zmm16, zmm1
// CHECK-REGALLOC-NEXT:      vaddpd zmm0, zmm15, zmm0
// CHECK-REGALLOC-NEXT:      vaddpd zmm4, zmm14, zmm4
// CHECK-REGALLOC-NEXT:      vaddpd zmm3, zmm13, zmm3
// CHECK-REGALLOC-NEXT:      vaddpd zmm2, zmm12, zmm2
// CHECK-REGALLOC-NEXT:      vaddpd zmm1, zmm11, zmm1
// CHECK-REGALLOC-NEXT:      vaddpd zmm0, zmm10, zmm0
// CHECK-REGALLOC-NEXT:      vaddpd zmm4, zmm9, zmm4
// CHECK-REGALLOC-NEXT:      vaddpd zmm3, zmm8, zmm3
// CHECK-REGALLOC-NEXT:      vaddpd zmm2, zmm7, zmm2
// CHECK-REGALLOC-NEXT:      vaddpd zmm1, zmm6, zmm1
// CHECK-REGALLOC-NEXT:      vaddpd zmm0, zmm5, zmm0
// CHECK-REGALLOC-NEXT:      vmovupd [rdx] {k1}, zmm4
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+272] {k1}, zmm3
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+544] {k1}, zmm2
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+816] {k1}, zmm1
// CHECK-REGALLOC-NEXT:      vmovupd [rdx+1088] {k1}, zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 4336
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 16
// CHECK-REGALLOC-NEXT:      sub rdi, 272
// CHECK-REGALLOC-NEXT:      add rsi, 640
// CHECK-REGALLOC-NEXT:      add rdx, 1088
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
