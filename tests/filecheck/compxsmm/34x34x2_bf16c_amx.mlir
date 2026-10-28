// RUN: compxsmm-gemm dense %t matmul 34 34 2 34 2 34 1 1 1 1 spr nopf BF16_VNNI_A && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul 34 34 2 34 2 34 1 1 1 1 spr nopf BF16_VNNI_A && xdsl-opt %t -f mlir -p 'xsmm-apply-schedule{strategy=libxsmm-spr-bf16-amx},x86-regalloc-verify-liveness,convert-x86-scf-to-x86,x86-prologue-epilogue-insertion' -t x86-asm | filecheck %s --check-prefix CHECK-ASM

// A BF16 C accumulates in F32 tiles all the same; the matmul says so with its
// c_datatype. The AMX strategy moves C through a scratch below the caller's
// tile configuration, as LIBXSMM's AMX kernel does: M = 34 is a block of 32 rows,
// whose two tiles of M store together, and one of 2, whose single tile is masked.

// CHECK:       x86_func.func public @matmul(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:  %3, %4, %5 = xsmm.matmul %0, %1, %2 {m = 34 : i64, n = 34 : i64, k = 2 : i64, lda = 34 : i64, ldb = 2 : i64, ldc = 34 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", ab_datatype = bf16, c_datatype = bf16, vnni_a} : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
// CHECK-NEXT:    x86_func.ret
// CHECK-NEXT:  }

// The scratch is 64-byte aligned below the caller's configuration, whose
// address r12 keeps; r15 strides the tiles over its columns of 16 F32.
// CHECK-ASM:           sub rsp, 64
// CHECK-ASM-NEXT:      sttilecfg [rsp]
// CHECK-ASM-NEXT:      mov r12, rsp
// CHECK-ASM-NEXT:      sub rsp, 2112
// CHECK-ASM-NEXT:      mov r10, -64
// CHECK-ASM-NEXT:      and rsp, r10
// CHECK-ASM-NEXT:      mov r13, 34
// CHECK-ASM-NEXT:      mov r14, 2
// CHECK-ASM-NEXT:      mov r15, 16

// Each column of C is widened to F32 into the scratch before its tile loads.
// CHECK-ASM:           ldtilecfg [rip+.Lmatmul_tileconfig0]
// CHECK-ASM-NEXT:      mov r11, 0
// CHECK-ASM-NEXT:  [[N_BODY:^\S+]]:
// CHECK-ASM-NEXT:      add r11, 17
// CHECK-ASM-NEXT:      vmovdqu16 ymm0, [rdx]
// CHECK-ASM-NEXT:      vpmovsxwd zmm0, ymm0
// CHECK-ASM-NEXT:      vpslld zmm0, zmm0, 16
// CHECK-ASM-NEXT:      vmovups [rsp], zmm0
// CHECK-ASM-NEXT:      vmovdqu16 ymm1, [rdx+68]
// CHECK-ASM:           vmovups [rsp+512], zmm0
// CHECK-ASM-NEXT:      tileloadd tmm0, [rsp+r15*4]
// CHECK-ASM:           tileloadd tmm1, [rsp+r15*4+576]
// CHECK-ASM:           tileloadd tmm2, [rsp+r15*4+64]
// CHECK-ASM:           tileloadd tmm3, [rsp+r15*4+640]
// CHECK-ASM-NEXT:      tileloadd tmm4, [rdi+r13*4]

// The two tiles of M of each tile of N go to the scratch together, and one
// vcvtne2ps2bf16 rounds a column of both into a 32-row column of C.
// CHECK-ASM:           tilestored [rsp+r15*4], tmm0
// CHECK-ASM-NEXT:      tilestored [rsp+r15*4+576], tmm2
// CHECK-ASM-NEXT:      vmovups zmm0, [rsp+576]
// CHECK-ASM-NEXT:      vcvtne2ps2bf16 zmm0, zmm0, [rsp]
// CHECK-ASM-NEXT:      vmovdqu16 [rdx], zmm0
// CHECK-ASM:           vmovdqu16 [rdx+544], zmm8
// CHECK-ASM-NEXT:      tilestored [rsp+r15*4], tmm1
// CHECK-ASM-NEXT:      tilestored [rsp+r15*4+512], tmm3
// CHECK-ASM:           vmovdqu16 [rdx+1088], zmm7
// CHECK-ASM:           jl [[N_BODY]]

// The remainder of 2 rows masks every move of its single tile of M with k2.
// CHECK-ASM:           ldtilecfg [rip+.Lmatmul_tileconfig1]
// CHECK-ASM-NEXT:      mov r10, 3
// CHECK-ASM-NEXT:      kmovw k2, r10d
// CHECK-ASM:           vmovdqu16 ymm0 {k2}{z}, [rdx]
// CHECK-ASM-NEXT:      vpmovsxwd zmm0, ymm0
// CHECK-ASM-NEXT:      vpslld zmm0, zmm0, 16
// CHECK-ASM-NEXT:      vmovups [rsp] {k2}, zmm0
// CHECK-ASM:           tilestored [rsp+r15*4], tmm0
// CHECK-ASM-NEXT:      vmovups zmm0 {k2}{z}, [rsp]
// CHECK-ASM-NEXT:      vcvtneps2bf16 ymm0, zmm0
// CHECK-ASM-NEXT:      vmovdqu16 [rdx] {k2}, ymm0

// RSP comes back from r12 before the caller's configuration does.
// CHECK-ASM:           mov rsp, r12
// CHECK-ASM-NEXT:      ldtilecfg [rsp]
// CHECK-ASM-NEXT:      add rsp, 64
// CHECK-ASM-NEXT:      pop r15
// CHECK-ASM-NEXT:      pop r14
// CHECK-ASM-NEXT:      pop r13
// CHECK-ASM-NEXT:      pop r12
// CHECK-ASM-NEXT:      ret
