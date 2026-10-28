// RUN: compxsmm-gemm dense %t matmul 34 34 34 34 34 34 1 1 1 1 spr nopf BF16F32_VNNI_A && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul 34 34 34 34 34 34 1 1 1 1 spr nopf BF16F32_VNNI_A && xdsl-opt %t -f mlir -p 'xsmm-apply-schedule{strategy=libxsmm-spr-bf16-amx},x86-regalloc-verify-liveness,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion' -t x86-asm | filecheck %s --check-prefix CHECK-ASM

// On SPR the GEMM is the one the AVX-512 kernels compute; the AMX strategy
// lowers it to tiles the way LIBXSMM's AMX kernel does. M = 34 has no good
// tile divisor, so it is cut into a block of 32 rows and one of 2, each with
// its own tile configuration; N = 34 is blocked by 17, two tiles of 9 and 8
// columns, and K = 34 by pairs, the largest even divisor up to 32.

// CHECK:       x86_func.func public @matmul(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:  %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 34 : i64, n = 34 : i64, k = 34 : i64, lda = 34 : i64, ldb = 34 : i64, ldc = 34 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", ab_datatype = bf16, vnni_a} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// The caller's tile configuration is saved, then the row strides of A, B and C
// are set for the tile loads to scale.
// CHECK-ASM:       matmul:
// CHECK-ASM:           sub rsp, 64
// CHECK-ASM-NEXT:      sttilecfg [rsp]
// CHECK-ASM-NEXT:      mov r13, 34
// CHECK-ASM-NEXT:      mov r14, 34
// CHECK-ASM-NEXT:      mov r15, 34
// CHECK-ASM-NEXT:  .pushsection .rodata
// CHECK-ASM-NEXT:  .p2align 6
// CHECK-ASM-NEXT:  .Lmatmul_tileconfig0:
// CHECK-ASM-NEXT:  .byte 0x01, 0x00, 0x00, 0x00
// CHECK-ASM:       .byte 0x40, 0x00, 0x40, 0x00
// CHECK-ASM-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-ASM-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-ASM-NEXT:  .byte 0x04, 0x00, 0x04, 0x00
// CHECK-ASM:       .byte 0x09, 0x08, 0x09, 0x08
// CHECK-ASM-NEXT:  .byte 0x01, 0x01, 0x09, 0x08
// CHECK-ASM:       .popsection
// CHECK-ASM-NEXT:      ldtilecfg [rip+.Lmatmul_tileconfig0]
// CHECK-ASM-NEXT:      mov r11, 0
// CHECK-ASM-NEXT:  [[N_BODY:^\S+]]:
// CHECK-ASM-NEXT:      add r11, 17

// Two by two accumulator tiles; per K pair, A tiles in tmm4 and tmm5 and B
// tiles in tmm6 and tmm7, in LIBXSMM's order.
// CHECK-ASM-NEXT:      tileloadd tmm0, [rdx+r15*4]
// CHECK-ASM-NEXT:      tileloadd tmm1, [rdx+r15*4+1224]
// CHECK-ASM-NEXT:      tileloadd tmm2, [rdx+r15*4+64]
// CHECK-ASM-NEXT:      tileloadd tmm3, [rdx+r15*4+1288]
// CHECK-ASM-NEXT:      tileloadd tmm4, [rdi+r13*4]
// CHECK-ASM-NEXT:      tileloadd tmm6, [rsi+r14*2]
// CHECK-ASM-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-ASM-NEXT:      tileloadd tmm5, [rdi+r13*4+64]
// CHECK-ASM-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-ASM-NEXT:      tileloadd tmm7, [rsi+r14*2+612]
// CHECK-ASM-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-ASM-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-ASM-NEXT:      tileloadd tmm4, [rdi+r13*4+136]
// CHECK-ASM-NEXT:      tileloadd tmm6, [rsi+r14*2+4]

// The last of the 17 K pairs, then the tiles of each N tile in turn are stored.
// CHECK-ASM:           tileloadd tmm4, [rdi+r13*4+2176]
// CHECK-ASM-NEXT:      tileloadd tmm6, [rsi+r14*2+64]
// CHECK-ASM-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-ASM-NEXT:      tileloadd tmm5, [rdi+r13*4+2240]
// CHECK-ASM-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-ASM-NEXT:      tileloadd tmm7, [rsi+r14*2+676]
// CHECK-ASM-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-ASM-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-ASM:           tilestored [rdx+r15*4], tmm0
// CHECK-ASM-NEXT:      tilestored [rdx+r15*4+64], tmm2
// CHECK-ASM-NEXT:      tilestored [rdx+r15*4+1224], tmm1
// CHECK-ASM-NEXT:      tilestored [rdx+r15*4+1288], tmm3
// CHECK-ASM:           jl [[N_BODY]]

// The remainder of 2 rows: one tile of M, 8 bytes wide.
// CHECK-ASM:       .Lmatmul_tileconfig1:
// CHECK-ASM-NEXT:  .byte 0x01, 0x00, 0x00, 0x00
// CHECK-ASM:       .byte 0x08, 0x00, 0x08, 0x00
// CHECK-ASM-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-ASM-NEXT:  .byte 0x08, 0x00, 0x40, 0x00
// CHECK-ASM-NEXT:  .byte 0x04, 0x00, 0x04, 0x00
// CHECK-ASM:       .byte 0x09, 0x08, 0x10, 0x10
// CHECK-ASM-NEXT:  .byte 0x01, 0x10, 0x09, 0x08
// CHECK-ASM:           ldtilecfg [rip+.Lmatmul_tileconfig1]
// CHECK-ASM:           tileloadd tmm0, [rdx+r15*4]
// CHECK-ASM-NEXT:      tileloadd tmm1, [rdx+r15*4+1224]
// CHECK-ASM-NEXT:      tileloadd tmm4, [rdi+r13*4]
// CHECK-ASM-NEXT:      tileloadd tmm6, [rsi+r14*2]
// CHECK-ASM-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-ASM-NEXT:      tileloadd tmm7, [rsi+r14*2+612]
// CHECK-ASM-NEXT:      tdpbf16ps tmm1, tmm7, tmm4

// The caller's configuration comes back before the callee-saved registers do.
// CHECK-ASM:           ldtilecfg [rsp]
// CHECK-ASM-NEXT:      add rsp, 64
// CHECK-ASM-NEXT:      pop r15
// CHECK-ASM-NEXT:      pop r14
// CHECK-ASM-NEXT:      pop r13
// CHECK-ASM-NEXT:      ret
