// RUN: libxsmm-gemm dense %t matmul 32 17 2 32 2 32 1 0 1 1 spr nopf BF16_VNNI_A && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// The AMX kernel LIBXSMM emits for SPR when C is BF16, for a 32x17 block with
// beta 0: two tiles of M (tmm0 over tmm2, tmm1 over tmm3) and N split into tiles
// of 9 and 8 columns. LIBXSMM pairs the stores of the two tiles of a column of
// tiles: both go to the GEMM scratch, 16 F32 rows apart a column, and
// vcvtne2ps2bf16 rounds a column of the upper tile into the lower half of a
// register and the column of the lower tile into its upper half, which stores as
// a whole 32-row column of C.

// CHECK:  .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push r14
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
// CHECK-NEXT:  .byte 0x04, 0x00, 0x04, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x09, 0x08, 0x09, 0x08
// CHECK-NEXT:  .byte 0x01, 0x01, 0x09, 0x08
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .byte 0x00, 0x00, 0x00, 0x00
// CHECK-NEXT:  .popsection
// CHECK-NEXT:      ldtilecfg [rip+.Lmatmul_tileconfig0]
// CHECK-NEXT:      mov r11, 32
// CHECK-NEXT:      mov r15, 1
// CHECK-NEXT:      mov r14, 32
// CHECK-NEXT:      tilezero tmm0
// CHECK-NEXT:      tilezero tmm1
// CHECK-NEXT:      tilezero tmm2
// CHECK-NEXT:      tilezero tmm3
// CHECK-NEXT:      tileloadd tmm4, [rdi+r11*4]
// CHECK-NEXT:      tileloadd tmm6, [rsi+r15*4]
// CHECK-NEXT:      tdpbf16ps tmm0, tmm6, tmm4
// CHECK-NEXT:      tileloadd tmm5, [rdi+r11*4+64]
// CHECK-NEXT:      tdpbf16ps tmm2, tmm6, tmm5
// CHECK-NEXT:      tileloadd tmm7, [rsi+r15*4+36]
// CHECK-NEXT:      tdpbf16ps tmm1, tmm7, tmm4
// CHECK-NEXT:      tdpbf16ps tmm3, tmm7, tmm5
// CHECK-NEXT:      mov r12, [rbp-48]
// CHECK-NEXT:      mov r14, 16
// CHECK-NEXT:      tilestored [r12+r14*4], tmm0
// CHECK-NEXT:      tilestored [r12+r14*4+576], tmm2
// CHECK-NEXT:      mov r14, 32
// CHECK-NEXT:      vmovups zmm0, [r12+576]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm0, zmm0, [r12]
// CHECK-NEXT:      vmovdqu16 [rdx], zmm0
// CHECK-NEXT:      vmovups zmm1, [r12+640]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm1, zmm1, [r12+64]
// CHECK-NEXT:      vmovdqu16 [rdx+64], zmm1
// CHECK-NEXT:      vmovups zmm2, [r12+704]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm2, zmm2, [r12+128]
// CHECK-NEXT:      vmovdqu16 [rdx+128], zmm2
// CHECK-NEXT:      vmovups zmm3, [r12+768]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm3, zmm3, [r12+192]
// CHECK-NEXT:      vmovdqu16 [rdx+192], zmm3
// CHECK-NEXT:      vmovups zmm4, [r12+832]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm4, zmm4, [r12+256]
// CHECK-NEXT:      vmovdqu16 [rdx+256], zmm4
// CHECK-NEXT:      vmovups zmm5, [r12+896]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm5, zmm5, [r12+320]
// CHECK-NEXT:      vmovdqu16 [rdx+320], zmm5
// CHECK-NEXT:      vmovups zmm6, [r12+960]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm6, zmm6, [r12+384]
// CHECK-NEXT:      vmovdqu16 [rdx+384], zmm6
// CHECK-NEXT:      vmovups zmm7, [r12+1024]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm7, zmm7, [r12+448]
// CHECK-NEXT:      vmovdqu16 [rdx+448], zmm7
// CHECK-NEXT:      vmovups zmm8, [r12+1088]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm8, zmm8, [r12+512]
// CHECK-NEXT:      vmovdqu16 [rdx+512], zmm8
// CHECK-NEXT:      mov r12, [rbp-48]
// CHECK-NEXT:      mov r14, 16
// CHECK-NEXT:      tilestored [r12+r14*4], tmm1
// CHECK-NEXT:      tilestored [r12+r14*4+512], tmm3
// CHECK-NEXT:      mov r14, 32
// CHECK-NEXT:      vmovups zmm0, [r12+512]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm0, zmm0, [r12]
// CHECK-NEXT:      vmovdqu16 [rdx+576], zmm0
// CHECK-NEXT:      vmovups zmm1, [r12+576]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm1, zmm1, [r12+64]
// CHECK-NEXT:      vmovdqu16 [rdx+640], zmm1
// CHECK-NEXT:      vmovups zmm2, [r12+640]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm2, zmm2, [r12+128]
// CHECK-NEXT:      vmovdqu16 [rdx+704], zmm2
// CHECK-NEXT:      vmovups zmm3, [r12+704]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm3, zmm3, [r12+192]
// CHECK-NEXT:      vmovdqu16 [rdx+768], zmm3
// CHECK-NEXT:      vmovups zmm4, [r12+768]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm4, zmm4, [r12+256]
// CHECK-NEXT:      vmovdqu16 [rdx+832], zmm4
// CHECK-NEXT:      vmovups zmm5, [r12+832]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm5, zmm5, [r12+320]
// CHECK-NEXT:      vmovdqu16 [rdx+896], zmm5
// CHECK-NEXT:      vmovups zmm6, [r12+896]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm6, zmm6, [r12+384]
// CHECK-NEXT:      vmovdqu16 [rdx+960], zmm6
// CHECK-NEXT:      vmovups zmm7, [r12+960]
// CHECK-NEXT:      vcvtne2ps2bf16 zmm7, zmm7, [r12+448]
// CHECK-NEXT:      vmovdqu16 [rdx+1024], zmm7
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ldtilecfg [rsp]
// CHECK-NEXT:      add rsp, 64
// CHECK-NEXT:      pop r12
// CHECK-NEXT:      pop r14
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
