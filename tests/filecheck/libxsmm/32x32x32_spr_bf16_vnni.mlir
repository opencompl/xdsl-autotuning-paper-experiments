// RUN: libxsmm-gemm dense %t matmul 32 32 32 32 32 32 1 1 1 1 spr nopf BF16F32_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK,CHECK-BETA1
// RUN: libxsmm-gemm dense %t matmul 32 32 32 32 32 32 1 0 1 1 spr nopf BF16F32_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK,CHECK-BETA0

// The AMX kernel LIBXSMM emits for SPR when A is packed in K pairs, for one
// 32x32 block of C: four 16x16 F32 tiles, loaded from C (or zeroed for beta 0)
// and stored back, accumulate one K block of 32 from two A tiles (tmm4, tmm5)
// and two B tiles (tmm6, tmm7). It saves the caller's tile configuration below
// its stack frame, which reserves LIBXSMM's GEMM scratch, and restores it at the
// end; its own configuration is in .rodata rather than in the code.

// CHECK:  .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push r14
// CHECK-NEXT:      sub rsp, 64
// CHECK-NEXT:      sttilecfg [rsp]
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 240
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      sub rsp, 12352
// CHECK-NEXT:  .pushsection .rodata
// CHECK-NEXT:  .p2align 6
// CHECK-NEXT:  .Lmatmul_tileconfig0:
// CHECK-NEXT:  .byte 0x01, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x10, 0x10, 0x10, 0x10
// CHECK-NEXT:  .byte 0x10, 0x10, 0x10, 0x10
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .popsection
// CHECK-NEXT:      ldtilecfg [rip+.Lmatmul_tileconfig0]
// CHECK-NEXT:      mov r11, 32
// CHECK-NEXT:      mov r15, 16
// CHECK-NEXT:      mov r14, 32
// CHECK-BETA1-NEXT:      tileloadd tmm0, [rdx+r14*4]
// CHECK-BETA1-NEXT:      tileloadd tmm1, [rdx+r14*4+2048]
// CHECK-BETA1-NEXT:      tileloadd tmm2, [rdx+r14*4+64]
// CHECK-BETA1-NEXT:      tileloadd tmm3, [rdx+r14*4+2112]
// CHECK-BETA0-NEXT:      tilezero tmm0
// CHECK-BETA0-NEXT:      tilezero tmm1
// CHECK-BETA0-NEXT:      tilezero tmm2
// CHECK-BETA0-NEXT:      tilezero tmm3
// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm5, [rdi+r11*4+64]
// CHECK-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-NEXT:      tileloadd tmm7, [rsi+r15*4+1024]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-NEXT:      tilestored [rdx+r14*4], tmm0
// CHECK-NEXT:      tilestored [rdx+r14*4+64], tmm2
// CHECK-NEXT:      tilestored [rdx+r14*4+2048], tmm1
// CHECK-NEXT:      tilestored [rdx+r14*4+2112], tmm3
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ldtilecfg [rsp]
// CHECK-NEXT:      add rsp, 64
// CHECK-NEXT:      pop r14
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
