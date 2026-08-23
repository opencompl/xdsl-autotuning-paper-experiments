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
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul_n"(%0, %1, %2, %6, %9) <{m = 16 : i64, n_start = 0 : i64, n_blocking = 14 : i64, k = 24 : i64, lda = 16 : i64, ldb = 24 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:    %15, %16, %17, %18, %19 = "xsmm.matmul_n"(%10, %11, %12, %13, %14) <{m = 16 : i64, n_start = 14 : i64, n_blocking = 52 : i64, k = 24 : i64, lda = 16 : i64, ldb = 24 : i64, ldc = 16 : i64, datatype = f64, aligned_a = true, aligned_c = true}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:    %20 = x86.ds.mov %18 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %21, %22 = x86.d.pop %20 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:      add rax, 14
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm4, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm5, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovapd zmm23, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovapd zmm24, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovapd zmm25, [rdx+1344]
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx+1408]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdx+1472]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdx+1600]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx+1664]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+1728]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm4, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm5, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2496]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      cmp rbx, 24
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 192
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm4
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm5
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm17
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1152], zmm22
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1216], zmm23
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1280], zmm24
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1344], zmm25
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1408], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1472], zmm27
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1536], zmm28
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1600], zmm29
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1664], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1728], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 2944
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      add rdx, 1664
// CHECK-REGALLOC-NEXT:      add rsi, 2688
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp rax, 14
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      mov rax, 14
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rax, 13
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rcx, 16
// CHECK-REGALLOC-NEXT:      vmovapd zmm6, [rdx]
// CHECK-REGALLOC-NEXT:      vmovapd zmm7, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovapd zmm8, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovapd zmm9, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovapd zmm10, [rdx+256]
// CHECK-REGALLOC-NEXT:      vmovapd zmm11, [rdx+320]
// CHECK-REGALLOC-NEXT:      vmovapd zmm12, [rdx+384]
// CHECK-REGALLOC-NEXT:      vmovapd zmm13, [rdx+448]
// CHECK-REGALLOC-NEXT:      vmovapd zmm14, [rdx+512]
// CHECK-REGALLOC-NEXT:      vmovapd zmm15, [rdx+576]
// CHECK-REGALLOC-NEXT:      vmovapd zmm16, [rdx+640]
// CHECK-REGALLOC-NEXT:      vmovapd zmm17, [rdx+704]
// CHECK-REGALLOC-NEXT:      vmovapd zmm18, [rdx+768]
// CHECK-REGALLOC-NEXT:      vmovapd zmm19, [rdx+832]
// CHECK-REGALLOC-NEXT:      vmovapd zmm20, [rdx+896]
// CHECK-REGALLOC-NEXT:      vmovapd zmm21, [rdx+960]
// CHECK-REGALLOC-NEXT:      vmovapd zmm22, [rdx+1024]
// CHECK-REGALLOC-NEXT:      vmovapd zmm23, [rdx+1088]
// CHECK-REGALLOC-NEXT:      vmovapd zmm24, [rdx+1152]
// CHECK-REGALLOC-NEXT:      vmovapd zmm25, [rdx+1216]
// CHECK-REGALLOC-NEXT:      vmovapd zmm26, [rdx+1280]
// CHECK-REGALLOC-NEXT:      vmovapd zmm27, [rdx+1344]
// CHECK-REGALLOC-NEXT:      vmovapd zmm28, [rdx+1408]
// CHECK-REGALLOC-NEXT:      vmovapd zmm29, [rdx+1472]
// CHECK-REGALLOC-NEXT:      vmovapd zmm30, [rdx+1536]
// CHECK-REGALLOC-NEXT:      vmovapd zmm31, [rdx+1600]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm2, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm0, zmm2
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm1, zmm2
// CHECK-REGALLOC-NEXT:      vmovapd zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm0, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vmovapd zmm2, [rdi]
// CHECK-REGALLOC-NEXT:      vmovapd zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm6, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm7, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+192]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm8, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm9, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+384]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+576]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm12, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm13, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+768]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+960]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm16, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm17, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1152]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1344]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm20, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm21, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1728]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm24, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm25, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+1920]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2112]
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm28, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm29, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastsd zmm1, [rsi+2304]
// CHECK-REGALLOC-NEXT:      add rsi, 8
// CHECK-REGALLOC-NEXT:      add rdi, 128
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231pd zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 24
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      sub rsi, 192
// CHECK-REGALLOC-NEXT:      vmovapd [rdx], zmm6
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+64], zmm7
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+128], zmm8
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+192], zmm9
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+256], zmm10
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+320], zmm11
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+384], zmm12
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+448], zmm13
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+512], zmm14
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+576], zmm15
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+640], zmm16
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+704], zmm17
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+768], zmm18
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+832], zmm19
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+896], zmm20
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+960], zmm21
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1024], zmm22
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1088], zmm23
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1152], zmm24
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1216], zmm25
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1280], zmm26
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1344], zmm27
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1408], zmm28
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1472], zmm29
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1536], zmm30
// CHECK-REGALLOC-NEXT:      vmovapd [rdx+1600], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 128
// CHECK-REGALLOC-NEXT:      sub rdi, 2944
// CHECK-REGALLOC-NEXT:      cmp rcx, 16
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      add rdx, 1536
// CHECK-REGALLOC-NEXT:      add rsi, 2496
// CHECK-REGALLOC-NEXT:      sub rdi, 128
// CHECK-REGALLOC-NEXT:      cmp rax, 66
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
