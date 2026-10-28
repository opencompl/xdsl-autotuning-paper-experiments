// RUN: libxsmm-gemm dense %t matmul 64 64 64 64 64 64 1 1 1 1 spr nopf BF16F32_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// The AMX kernel LIBXSMM emits for SPR when A is packed in K pairs, with loops
// over blocks of 32 in N (r13) and M (r12) around the 2x2 tiles of C, and the
// two K blocks of 32 unrolled.

// CHECK:  .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push r14
// CHECK-NEXT:      push r13
// CHECK-NEXT:      push r12
// CHECK-NEXT:      sub rsp, 64
// CHECK-NEXT:      sttilecfg [rsp]
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 240
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      sub rsp, 12352
// CHECK-NEXT:      mov [rbp-48], rsp
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
// CHECK-NEXT:      mov r11, 64
// CHECK-NEXT:      mov r15, 32
// CHECK-NEXT:      mov r14, 64
// CHECK-NEXT:      mov r13, 0
// CHECK-NEXT:  [[N_BODY:^\S+]]:
// CHECK-NEXT:      add r13, 32
// CHECK-NEXT:      mov r12, 0
// CHECK-NEXT:  [[M_BODY:^\S+]]:
// CHECK-NEXT:      add r12, 32
// CHECK-NEXT:      tileloadd tmm0, [rdx+r14*4]
// CHECK-NEXT:      tileloadd tmm1, [rdx+r14*4+4096]
// CHECK-NEXT:      tileloadd tmm2, [rdx+r14*4+64]
// CHECK-NEXT:      tileloadd tmm3, [rdx+r14*4+4160]
// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm5, [rdi+r11*4+64]
// CHECK-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-NEXT:      tileloadd tmm7, [rsi+r15*4+2048]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4+4096]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4+64]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm5, [rdi+r11*4+4160]
// CHECK-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-NEXT:      tileloadd tmm7, [rsi+r15*4+2112]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-NEXT:      tilestored [rdx+r14*4], tmm0
// CHECK-NEXT:      tilestored [rdx+r14*4+64], tmm2
// CHECK-NEXT:      tilestored [rdx+r14*4+4096], tmm1
// CHECK-NEXT:      tilestored [rdx+r14*4+4160], tmm3
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      cmp r12, 64
// CHECK-NEXT:      jl [[M_BODY]]
// CHECK-NEXT:      add rdx, 7936
// CHECK-NEXT:      add rsi, 4096
// CHECK-NEXT:      sub rdi, 256
// CHECK-NEXT:      cmp r13, 64
// CHECK-NEXT:      jl [[N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ldtilecfg [rsp]
// CHECK-NEXT:      add rsp, 64
// CHECK-NEXT:      pop r12
// CHECK-NEXT:      pop r13
// CHECK-NEXT:      pop r14
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
