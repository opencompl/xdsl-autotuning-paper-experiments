// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 50 38 128 50 128 50 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12 = "xsmm.matmul"(%0, %1, %2) <{m = 50 : i64, n = 38 : i64, k = 128 : i64, lda = 50 : i64, ldb = 128 : i64, ldc = 50 : i64, datatype = f32, aligned_a = false, aligned_c = false, iterator = "n", operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>)
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
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm20 {k1}{z}, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdx+200]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdx+264]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdx+328]
// CHECK-REGALLOC-NEXT:      vmovups zmm3 {k1}{z}, [rdx+392]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovups zmm5, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovups zmm6, [rdx+528]
// CHECK-REGALLOC-NEXT:      vmovups zmm7 {k1}{z}, [rdx+592]
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx+600]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+664]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+728]
// CHECK-REGALLOC-NEXT:      vmovups zmm11 {k1}{z}, [rdx+792]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+800]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+864]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+928]
// CHECK-REGALLOC-NEXT:      vmovups zmm15 {k1}{z}, [rdx+992]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+1000]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+1064]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+1128]
// CHECK-REGALLOC-NEXT:      vmovups zmm19 {k1}{z}, [rdx+1192]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm28, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm24, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm25, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm28
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm27, zmm28
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm25 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm27, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm28, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm25, zmm24
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm28 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm27, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm25, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm24, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm27
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm28, zmm27
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm24 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm25, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm28, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm27, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm26, zmm25
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm24, zmm25
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192] {k1}, zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+200], zmm0
// CHECK-REGALLOC-NEXT:      vmovups [rdx+264], zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+328], zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+392] {k1}, zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+400], zmm4
// CHECK-REGALLOC-NEXT:      vmovups [rdx+464], zmm5
// CHECK-REGALLOC-NEXT:      vmovups [rdx+528], zmm6
// CHECK-REGALLOC-NEXT:      vmovups [rdx+592] {k1}, zmm7
// CHECK-REGALLOC-NEXT:      vmovups [rdx+600], zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+664], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+728], zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+792] {k1}, zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+800], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+864], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+928], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+992] {k1}, zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1000], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1064], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1128], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1192] {k1}, zmm19
// CHECK-REGALLOC-NEXT:      sub rdi, 25400
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 200
// CHECK-REGALLOC-NEXT:      sub rdi, 200
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      add rdx, 1000
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rax, 0
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 3
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm16 {k1}{z}, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+200]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+264]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+328]
// CHECK-REGALLOC-NEXT:      vmovups zmm12 {k1}{z}, [rdx+392]
// CHECK-REGALLOC-NEXT:      vmovups zmm11, [rdx+400]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+464]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+528]
// CHECK-REGALLOC-NEXT:      vmovups zmm8 {k1}{z}, [rdx+592]
// CHECK-REGALLOC-NEXT:      vmovups zmm7, [rdx+600]
// CHECK-REGALLOC-NEXT:      vmovups zmm6, [rdx+664]
// CHECK-REGALLOC-NEXT:      vmovups zmm5, [rdx+728]
// CHECK-REGALLOC-NEXT:      vmovups zmm4 {k1}{z}, [rdx+792]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdx+800]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdx+864]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdx+928]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdx+992]
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rcx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm23 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm24, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm20, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm21, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm24
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm23, zmm24
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm21 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm20, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm23, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm24, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm20
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm21, zmm20
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm24 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm23, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm21, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm20, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm23
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm24, zmm23
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm20 {k1}{z}, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm7, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm6, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm5, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm4, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm21, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 200
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm3, zmm24, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm2, zmm23, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm1, zmm22, zmm21
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm0, zmm20, zmm21
// CHECK-REGALLOC-NEXT:      cmp rcx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192] {k1}, zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+200], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+264], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+328], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+392] {k1}, zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+400], zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+464], zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+528], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+592] {k1}, zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+600], zmm7
// CHECK-REGALLOC-NEXT:      vmovups [rdx+664], zmm6
// CHECK-REGALLOC-NEXT:      vmovups [rdx+728], zmm5
// CHECK-REGALLOC-NEXT:      vmovups [rdx+792] {k1}, zmm4
// CHECK-REGALLOC-NEXT:      vmovups [rdx+800], zmm3
// CHECK-REGALLOC-NEXT:      vmovups [rdx+864], zmm2
// CHECK-REGALLOC-NEXT:      vmovups [rdx+928], zmm1
// CHECK-REGALLOC-NEXT:      vmovups [rdx+992] {k1}, zmm0
// CHECK-REGALLOC-NEXT:      sub rdi, 25400
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      add rdx, 200
// CHECK-REGALLOC-NEXT:      sub rdi, 200
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      add rdx, 800
// CHECK-REGALLOC-NEXT:      cmp rax, 20
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
