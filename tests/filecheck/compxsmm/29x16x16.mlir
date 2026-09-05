// RUN: compxsmm-gemm dense %t matmul_bac 16 29 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 29 16 16 16 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul"(%0, %1, %2, %6, %9) <{m = 16 : i64, n = 29 : i64, k = 16 : i64, lda = 16 : i64, ldb = 16 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
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
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rax, 10
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm22, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm22
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm21, zmm22
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm21, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm22, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm21, [rsi+1152]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm0
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm2
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1152], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1216], zmm17
// CHECK-REGALLOC-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 1280
// CHECK-REGALLOC-NEXT:      add rdx, 1152
// CHECK-REGALLOC-NEXT:      cmp rax, 20
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm18, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm19, zmm20
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm18, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm19, zmm18
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm20, zmm18
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+128]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+256]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+640]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+896]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm19, [rsi+1024]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm20, zmm19
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm18, zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm17
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm2
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 1920
// CHECK-REGALLOC-NEXT:      sub rsi, 128
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 1152
// CHECK-REGALLOC-NEXT:      add rdx, 1024
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
