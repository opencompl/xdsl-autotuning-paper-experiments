// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 16 66 24 16 24 16 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12 = "xsmm.matmul"(%0, %1, %2) <{m = 16 : i64, n = 66 : i64, k = 24 : i64, lda = 16 : i64, ldb = 24 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    %13 = x86.ds.mov %6 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %14, %15 = x86.d.pop %13 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+64]
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
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdx+1344]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+1408]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdx+1472]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovapd zmm23, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovapd zmm24, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovapd zmm25, [rdx+1728]
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rax, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm30, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm27, zmm30
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm29, zmm30
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm29, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm30, zmm27
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm29, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm30, zmm29
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm27, zmm29
// CHECK-REGALLOC-NEXT:      cmp rax, 24
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm28
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
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1280], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1344], zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1408], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1472], zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1536], zmm22
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1600], zmm23
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1664], zmm24
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1728], zmm25
// CHECK-REGALLOC-NEXT:      sub rdi, 2944
// CHECK-REGALLOC-NEXT:      sub rsi, 192
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 2688
// CHECK-REGALLOC-NEXT:      add rdx, 1664
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rax, 13
// CHECK-REGALLOC-NEXT:      vmovapd zmm25, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm24, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm23, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx+1344]
// CHECK-REGALLOC-NEXT:      vmovapd zmm3, [rdx+1408]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdx+1472]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdx+1600]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm28, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm26, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm27, zmm26
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm28, zmm26
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm3, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm2, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm27, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm1, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm0, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      cmp rcx, 24
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm25
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm24
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm23
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm22
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm17
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1152], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1216], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1280], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1344], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1408], zmm3
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1472], zmm2
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1536], zmm1
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1600], zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 2944
// CHECK-REGALLOC-NEXT:      sub rsi, 192
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      add rsi, 2496
// CHECK-REGALLOC-NEXT:      add rdx, 1536
// CHECK-REGALLOC-NEXT:      cmp rax, 52
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
