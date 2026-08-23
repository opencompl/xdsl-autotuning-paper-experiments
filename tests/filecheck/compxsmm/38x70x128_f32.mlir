// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP && cat %t | filecheck %s
// RUN: compxsmm-gemm dense %t matmul_bac 70 38 128 70 128 70 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC

// CHECK:       x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-NEXT:    %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-NEXT:    %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-NEXT:    %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-NEXT:    %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-NEXT:    %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %10, %11, %12, %13, %14 = "xsmm.matmul_n"(%0, %1, %2, %6, %9) <{m = 70 : i64, n_start = 0 : i64, n_blocking = 38 : i64, k = 128 : i64, lda = 70 : i64, ldb = 128 : i64, ldc = 70 : i64, datatype = f32, aligned_a = false, aligned_c = false}> : (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>) -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64<rdx>, !x86.reg64<rbp>, !x86.reg64<rsp>)
// CHECK-NEXT:    %15 = x86.ds.mov %13 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-NEXT:    %16, %17 = x86.d.pop %15 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
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
// CHECK-REGALLOC-NEXT:  scf_body_4_for:
// CHECK-REGALLOC-NEXT:      add rax, 6
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovups zmm8, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm9, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm10, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm11, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+344]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+408]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+472]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+624]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+688]
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx+752]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+904]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+968]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdx+1032]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+1184]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+1248]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdx+1312]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+1400]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+1464]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+1528]
// CHECK-REGALLOC-NEXT:      vmovups zmm31, [rdx+1592]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm8, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm9, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm10, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm11, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2560]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm8
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm9
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm10
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm11
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+344], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+408], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+472], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+624], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+688], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+752], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+904], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+968], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1032], zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1184], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1248], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1312], zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1400], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1464], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1528], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1592], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 35584
// CHECK-REGALLOC-NEXT:      cmp rcx, 64
// CHECK-REGALLOC-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-NEXT:      mov rcx, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 64
// CHECK-REGALLOC-NEXT:  scf_body_3_for:
// CHECK-REGALLOC-NEXT:      add rcx, 6
// CHECK-REGALLOC-NEXT:      vmovups zmm26 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm28 {k1}{z}, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm29 {k1}{z}, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm30 {k1}{z}, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1400]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-REGALLOC-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-REGALLOC-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-REGALLOC-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-REGALLOC-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-REGALLOC-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm0, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, [rsi+512]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+1024]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1536]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+2048]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2560]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+4]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+516]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+1028]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+1540]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+2052]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+2564]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm0, [rsi+8]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, [rsi+520]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, [rsi+1032]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, [rsi+1544]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm0, [rsi+2056]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, [rsi+2568]{1to16}
// CHECK-REGALLOC-NEXT:      add rdi, 1120
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+12]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+524]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm1, [rsi+1036]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, [rsi+1548]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, [rsi+2060]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, [rsi+2572]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 16
// CHECK-REGALLOC-NEXT:      vaddps zmm26, zmm20, zmm26
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm21, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm22, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm23, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm24, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm25, zmm31
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1400] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 35816
// CHECK-REGALLOC-NEXT:      cmp rcx, 70
// CHECK-REGALLOC-NEXT:      jl scf_body_3_for
// CHECK-REGALLOC-NEXT:      add rdx, 1400
// CHECK-REGALLOC-NEXT:      add rsi, 3072
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      cmp rax, 18
// CHECK-REGALLOC-NEXT:      jl scf_body_4_for
// CHECK-REGALLOC-NEXT:      mov rax, 18
// CHECK-REGALLOC-NEXT:  scf_body_9_for:
// CHECK-REGALLOC-NEXT:      add rax, 5
// CHECK-REGALLOC-NEXT:      mov rcx, 0
// CHECK-REGALLOC-NEXT:  scf_body_6_for:
// CHECK-REGALLOC-NEXT:      add rcx, 64
// CHECK-REGALLOC-NEXT:      vmovups zmm12, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm13, [rdx+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm14, [rdx+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm15, [rdx+192]
// CHECK-REGALLOC-NEXT:      vmovups zmm16, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm17, [rdx+344]
// CHECK-REGALLOC-NEXT:      vmovups zmm18, [rdx+408]
// CHECK-REGALLOC-NEXT:      vmovups zmm19, [rdx+472]
// CHECK-REGALLOC-NEXT:      vmovups zmm20, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm21, [rdx+624]
// CHECK-REGALLOC-NEXT:      vmovups zmm22, [rdx+688]
// CHECK-REGALLOC-NEXT:      vmovups zmm23, [rdx+752]
// CHECK-REGALLOC-NEXT:      vmovups zmm24, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm25, [rdx+904]
// CHECK-REGALLOC-NEXT:      vmovups zmm26, [rdx+968]
// CHECK-REGALLOC-NEXT:      vmovups zmm27, [rdx+1032]
// CHECK-REGALLOC-NEXT:      vmovups zmm28, [rdx+1120]
// CHECK-REGALLOC-NEXT:      vmovups zmm29, [rdx+1184]
// CHECK-REGALLOC-NEXT:      vmovups zmm30, [rdx+1248]
// CHECK-REGALLOC-NEXT:      vmovups zmm31, [rdx+1312]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_5_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm4, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm0, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm4
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm3, zmm4
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm1, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm3, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm4, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm1
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm0, zmm1
// CHECK-REGALLOC-NEXT:      vmovups zmm0, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm3, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm0, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm3
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm4, zmm3
// CHECK-REGALLOC-NEXT:      vmovups zmm4, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm3, [rdi+64]
// CHECK-REGALLOC-NEXT:      vmovups zmm2, [rdi+128]
// CHECK-REGALLOC-NEXT:      vmovups zmm1, [rdi+192]
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+512]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1024]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+1536]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      vbroadcastss zmm0, [rsi+2048]
// CHECK-REGALLOC-NEXT:      add rsi, 4
// CHECK-REGALLOC-NEXT:      add rdi, 280
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm4, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm3, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm2, zmm0
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, zmm0
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_5_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx], zmm12
// CHECK-REGALLOC-NEXT:      vmovups [rdx+64], zmm13
// CHECK-REGALLOC-NEXT:      vmovups [rdx+128], zmm14
// CHECK-REGALLOC-NEXT:      vmovups [rdx+192], zmm15
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280], zmm16
// CHECK-REGALLOC-NEXT:      vmovups [rdx+344], zmm17
// CHECK-REGALLOC-NEXT:      vmovups [rdx+408], zmm18
// CHECK-REGALLOC-NEXT:      vmovups [rdx+472], zmm19
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560], zmm20
// CHECK-REGALLOC-NEXT:      vmovups [rdx+624], zmm21
// CHECK-REGALLOC-NEXT:      vmovups [rdx+688], zmm22
// CHECK-REGALLOC-NEXT:      vmovups [rdx+752], zmm23
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840], zmm24
// CHECK-REGALLOC-NEXT:      vmovups [rdx+904], zmm25
// CHECK-REGALLOC-NEXT:      vmovups [rdx+968], zmm26
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1032], zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120], zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1184], zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1248], zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1312], zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 256
// CHECK-REGALLOC-NEXT:      sub rdi, 35584
// CHECK-REGALLOC-NEXT:      cmp rcx, 64
// CHECK-REGALLOC-NEXT:      jl scf_body_6_for
// CHECK-REGALLOC-NEXT:      mov rcx, 63
// CHECK-REGALLOC-NEXT:      kmovw k1, ecx
// CHECK-REGALLOC-NEXT:      mov rcx, 64
// CHECK-REGALLOC-NEXT:  scf_body_8_for:
// CHECK-REGALLOC-NEXT:      add rcx, 6
// CHECK-REGALLOC-NEXT:      vmovups zmm27 {k1}{z}, [rdx]
// CHECK-REGALLOC-NEXT:      vmovups zmm28 {k1}{z}, [rdx+280]
// CHECK-REGALLOC-NEXT:      vmovups zmm29 {k1}{z}, [rdx+560]
// CHECK-REGALLOC-NEXT:      vmovups zmm30 {k1}{z}, [rdx+840]
// CHECK-REGALLOC-NEXT:      vmovups zmm31 {k1}{z}, [rdx+1120]
// CHECK-REGALLOC-NEXT:      mov rbx, 0
// CHECK-REGALLOC-NEXT:  scf_body_7_for:
// CHECK-REGALLOC-NEXT:      add rbx, 4
// CHECK-REGALLOC-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-REGALLOC-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-REGALLOC-NEXT:      vpxord zmm24, zmm24, zmm24
// CHECK-REGALLOC-NEXT:      vpxord zmm25, zmm25, zmm25
// CHECK-REGALLOC-NEXT:      vpxord zmm26, zmm26, zmm26
// CHECK-REGALLOC-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-REGALLOC-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-REGALLOC-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-REGALLOC-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-REGALLOC-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-REGALLOC-NEXT:      vpxord zmm12, zmm12, zmm12
// CHECK-REGALLOC-NEXT:      vpxord zmm13, zmm13, zmm13
// CHECK-REGALLOC-NEXT:      vpxord zmm14, zmm14, zmm14
// CHECK-REGALLOC-NEXT:      vpxord zmm15, zmm15, zmm15
// CHECK-REGALLOC-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi]
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+280]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm27, zmm1, [rsi]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm28, zmm1, [rsi+512]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm29, zmm1, [rsi+1024]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm30, zmm1, [rsi+1536]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm31, zmm1, [rsi+2048]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm1 {k1}{z}, [rdi+560]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm22, zmm0, [rsi+4]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm23, zmm0, [rsi+516]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm24, zmm0, [rsi+1028]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm25, zmm0, [rsi+1540]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm26, zmm0, [rsi+2052]{1to16}
// CHECK-REGALLOC-NEXT:      vmovups zmm0 {k1}{z}, [rdi+840]
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm17, zmm1, [rsi+8]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm18, zmm1, [rsi+520]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm19, zmm1, [rsi+1032]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm20, zmm1, [rsi+1544]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm21, zmm1, [rsi+2056]{1to16}
// CHECK-REGALLOC-NEXT:      add rdi, 1120
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm12, zmm0, [rsi+12]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm13, zmm0, [rsi+524]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm14, zmm0, [rsi+1036]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm15, zmm0, [rsi+1548]{1to16}
// CHECK-REGALLOC-NEXT:      vfmadd231ps zmm16, zmm0, [rsi+2060]{1to16}
// CHECK-REGALLOC-NEXT:      add rsi, 16
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm22, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm23, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm24, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm25, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm26, zmm31
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm17, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm18, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm19, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm20, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm21, zmm31
// CHECK-REGALLOC-NEXT:      vaddps zmm27, zmm12, zmm27
// CHECK-REGALLOC-NEXT:      vaddps zmm28, zmm13, zmm28
// CHECK-REGALLOC-NEXT:      vaddps zmm29, zmm14, zmm29
// CHECK-REGALLOC-NEXT:      vaddps zmm30, zmm15, zmm30
// CHECK-REGALLOC-NEXT:      vaddps zmm31, zmm16, zmm31
// CHECK-REGALLOC-NEXT:      cmp rbx, 128
// CHECK-REGALLOC-NEXT:      jl scf_body_7_for
// CHECK-REGALLOC-NEXT:      sub rsi, 512
// CHECK-REGALLOC-NEXT:      vmovups [rdx] {k1}, zmm27
// CHECK-REGALLOC-NEXT:      vmovups [rdx+280] {k1}, zmm28
// CHECK-REGALLOC-NEXT:      vmovups [rdx+560] {k1}, zmm29
// CHECK-REGALLOC-NEXT:      vmovups [rdx+840] {k1}, zmm30
// CHECK-REGALLOC-NEXT:      vmovups [rdx+1120] {k1}, zmm31
// CHECK-REGALLOC-NEXT:      add rdx, 24
// CHECK-REGALLOC-NEXT:      sub rdi, 35816
// CHECK-REGALLOC-NEXT:      cmp rcx, 70
// CHECK-REGALLOC-NEXT:      jl scf_body_8_for
// CHECK-REGALLOC-NEXT:      add rdx, 1120
// CHECK-REGALLOC-NEXT:      add rsi, 2560
// CHECK-REGALLOC-NEXT:      sub rdi, 280
// CHECK-REGALLOC-NEXT:      cmp rax, 38
// CHECK-REGALLOC-NEXT:      jl scf_body_9_for
// CHECK-REGALLOC-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      pop rbx
// CHECK-REGALLOC-NEXT:      pop rbp
// CHECK-REGALLOC-NEXT:      ret
