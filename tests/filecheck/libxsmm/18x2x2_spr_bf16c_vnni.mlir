// RUN: libxsmm-gemm dense %t matmul 18 2 2 18 2 18 1 1 1 1 spr nopf BF16_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK,CHECK-BETA1
// RUN: libxsmm-gemm dense %t matmul 18 2 2 18 2 18 1 0 1 1 spr nopf BF16_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s --check-prefixes CHECK,CHECK-BETA0

// The AMX kernel LIBXSMM emits for SPR when C is BF16, for an 18x2 block: an N
// of one tile ("Mx1"), so two single tiles of M, the second of 2 rows. AMX has
// no BF16 accumulator and SPR no instruction that moves a tile row to a vector
// register, so C goes through LIBXSMM's GEMM scratch on the stack, whose address
// the frame keeps at [rbp-48]: for beta 1 each column of C is widened to F32
// (vpmovsxwd, vpslld) into the scratch, 64 bytes a column, and the tile loaded
// from there; after the multiplication each tile is stored to the scratch and
// its columns rounded back to BF16 (vcvtneps2bf16) into C. k2 masks the 2 rows
// of the partial tile; k1, the mask of a column of two tiles, goes unused.

// CHECK:  .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push r14
// CHECK-BETA1-NEXT:      push r13
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
// CHECK-NEXT:      mov r10, 3
// CHECK-NEXT:      kmovw k2, r10d
// CHECK-NEXT:      mov r10, 262143
// CHECK-NEXT:      kmovd k1, r10d
// CHECK-NEXT:  .pushsection .rodata
// CHECK-NEXT:  .p2align 6
// CHECK-NEXT:  .Lmatmul_tileconfig0:
// CHECK-NEXT:  .byte 0x01, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x08, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x40, 0x00, 0x08, 0x00
// CHECK-NEXT:  .byte 0x04, 0x00, 0x40, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x02, 0x02, 0x10, 0x10
// CHECK-NEXT:  .byte 0x01, 0x01, 0x02, 0x10
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .popsection
// CHECK-NEXT:      ldtilecfg [rip+.Lmatmul_tileconfig0]
// CHECK-NEXT:      mov r11, 18
// CHECK-NEXT:      mov r15, 1
// CHECK-NEXT:      mov r14, 18

// Beta 1: r14 holds the 16 F32 rows of the scratch while the tiles load.
// CHECK-BETA1-NEXT:      mov r14, 16
// CHECK-BETA1-NEXT:      mov r13, [rbp-48]
// CHECK-BETA1-NEXT:      vmovdqu16 ymm0, [rdx]
// CHECK-BETA1-NEXT:      vpmovsxwd zmm0, ymm0
// CHECK-BETA1-NEXT:      vpslld zmm0, zmm0, 16
// CHECK-BETA1-NEXT:      vmovups [r13], zmm0
// CHECK-BETA1-NEXT:      vmovdqu16 ymm1, [rdx+36]
// CHECK-BETA1-NEXT:      vpmovsxwd zmm1, ymm1
// CHECK-BETA1-NEXT:      vpslld zmm1, zmm1, 16
// CHECK-BETA1-NEXT:      vmovups [r13+64], zmm1
// CHECK-BETA1-NEXT:      tileloadd tmm0, [r13+r14*4]
// CHECK-BETA1-NEXT:      vmovdqu16 ymm0 {k2}{z}, [rdx+32]
// CHECK-BETA1-NEXT:      vpmovsxwd zmm0, ymm0
// CHECK-BETA1-NEXT:      vpslld zmm0, zmm0, 16
// CHECK-BETA1-NEXT:      vmovups [r13+64] {k2}, zmm0
// CHECK-BETA1-NEXT:      vmovdqu16 ymm1 {k2}{z}, [rdx+68]
// CHECK-BETA1-NEXT:      vpmovsxwd zmm1, ymm1
// CHECK-BETA1-NEXT:      vpslld zmm1, zmm1, 16
// CHECK-BETA1-NEXT:      vmovups [r13+128] {k2}, zmm1
// CHECK-BETA1-NEXT:      tileloadd tmm1, [r13+r14*4+64]
// CHECK-BETA1-NEXT:      mov r14, 18
// CHECK-BETA0-NEXT:      tilezero tmm0
// CHECK-BETA0-NEXT:      tilezero tmm1

// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm5, [rdi+r11*4+64]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm6, tmm5

// Each tile goes to the scratch on its own and back to C a column at a time.
// CHECK-NEXT:      mov r12, [rbp-48]
// CHECK-NEXT:      mov r14, 16
// CHECK-NEXT:      tilestored [r12+r14*4], tmm0
// CHECK-NEXT:      mov r14, 18
// CHECK-NEXT:      vmovups zmm0, [r12]
// CHECK-NEXT:      vcvtneps2bf16 ymm0, zmm0
// CHECK-NEXT:      vmovdqu16 [rdx], ymm0
// CHECK-NEXT:      vmovups zmm1, [r12+64]
// CHECK-NEXT:      vcvtneps2bf16 ymm1, zmm1
// CHECK-NEXT:      vmovdqu16 [rdx+36], ymm1
// CHECK-NEXT:      mov r12, [rbp-48]
// CHECK-NEXT:      mov r14, 16
// CHECK-NEXT:      tilestored [r12+r14*4], tmm1
// CHECK-NEXT:      mov r14, 18
// CHECK-NEXT:      vmovups zmm0 {k2}{z}, [r12]
// CHECK-NEXT:      vcvtneps2bf16 ymm0, zmm0
// CHECK-NEXT:      vmovdqu16 [rdx+32] {k2}, ymm0
// CHECK-NEXT:      vmovups zmm1 {k2}{z}, [r12+64]
// CHECK-NEXT:      vcvtneps2bf16 ymm1, zmm1
// CHECK-NEXT:      vmovdqu16 [rdx+68] {k2}, ymm1
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ldtilecfg [rsp]
// CHECK-NEXT:      add rsp, 64
// CHECK-NEXT:      pop r12
// CHECK-BETA1-NEXT:      pop r13
// CHECK-NEXT:      pop r14
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
