// RUN: libxsmm-gemm dense %t matmul 34 34 4 34 4 34 1 1 1 1 spr nopf BF16F32_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// The AMX kernel LIBXSMM emits for SPR when A is packed in K pairs, for an M
// that is not a multiple of 16: a first kernel covers 32 rows of M in 2x2 tiles,
// then a second tile configuration and kernel the 2 left over in 1x2 tiles,
// after moving C, A and B to them. N is blocked by 17, as tiles of 9 and 8
// columns. Before the second kernel LIBXSMM sets up k2 and k1 to mask the 2 rows
// in F32 and in BF16, but an F32 C only moves with tile loads and stores, which
// take no masks: neither is ever read.

// CHECK:  .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push r14
// CHECK-NEXT:      push r13
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
// CHECK-NEXT:  .byte 0x08, 0x00, 0x08, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x09, 0x08, 0x09, 0x08
// CHECK-NEXT:  .byte 0x02, 0x02, 0x09, 0x08
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .popsection
// CHECK-NEXT:      ldtilecfg [rip+.Lmatmul_tileconfig0]
// CHECK-NEXT:      mov r11, 34
// CHECK-NEXT:      mov r15, 2
// CHECK-NEXT:      mov r14, 34
// CHECK-NEXT:      mov r13, 0
// CHECK-NEXT:  [[N_BODY:^\S+]]:
// CHECK-NEXT:      add r13, 17
// CHECK-NEXT:      tileloadd tmm0, [rdx+r14*4]
// CHECK-NEXT:      tileloadd tmm1, [rdx+r14*4+1224]
// CHECK-NEXT:      tileloadd tmm2, [rdx+r14*4+64]
// CHECK-NEXT:      tileloadd tmm3, [rdx+r14*4+1288]
// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm5, [rdi+r11*4+64]
// CHECK-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-NEXT:      tileloadd tmm7, [rsi+r15*4+72]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-NEXT:      tilestored [rdx+r14*4], tmm0
// CHECK-NEXT:      tilestored [rdx+r14*4+64], tmm2
// CHECK-NEXT:      tilestored [rdx+r14*4+1224], tmm1
// CHECK-NEXT:      tilestored [rdx+r14*4+1288], tmm3
// CHECK-NEXT:      add rdx, 2312
// CHECK-NEXT:      add rsi, 136
// CHECK-NEXT:      cmp r13, 34
// CHECK-NEXT:      jl [[N_BODY]]
// CHECK-NEXT:      mov r10, 3
// CHECK-NEXT:      kmovw k2, r10d
// CHECK-NEXT:      mov r10, 262143
// CHECK-NEXT:      kmovd k1, r10d
// CHECK-NEXT:  .pushsection .rodata
// CHECK-NEXT:  .p2align 6
// CHECK-NEXT:  .Lmatmul_tileconfig1:
// CHECK-NEXT:  .byte 0x01, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x08, 0x00, 0x08, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x08, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x08, 0x00, 0x08, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x09, 0x08, 0x10, 0x10
// CHECK-NEXT:  .byte 0x02, 0x10, 0x09, 0x08
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .popsection
// CHECK-NEXT:      ldtilecfg [rip+.Lmatmul_tileconfig1]
// CHECK-NEXT:      add rdx, -4624
// CHECK-NEXT:      add rsi, -272
// CHECK-NEXT:      add rdx, 128
// CHECK-NEXT:      add rdi, 128
// CHECK-NEXT:      mov r11, 34
// CHECK-NEXT:      mov r15, 2
// CHECK-NEXT:      mov r14, 34
// CHECK-NEXT:      mov r13, 0
// CHECK-NEXT:  [[N_TAIL:^\S+]]:
// CHECK-NEXT:      add r13, 17
// CHECK-NEXT:      tileloadd tmm0, [rdx+r14*4]
// CHECK-NEXT:      tileloadd tmm1, [rdx+r14*4+1224]
// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm7, [rsi+r15*4+72]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-NEXT:      tilestored [rdx+r14*4], tmm0
// CHECK-NEXT:      tilestored [rdx+r14*4+1224], tmm1
// CHECK-NEXT:      add rdx, 2312
// CHECK-NEXT:      add rsi, 136
// CHECK-NEXT:      cmp r13, 34
// CHECK-NEXT:      jl [[N_TAIL]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ldtilecfg [rsp]
// CHECK-NEXT:      add rsp, 64
// CHECK-NEXT:      pop r13
// CHECK-NEXT:      pop r14
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
