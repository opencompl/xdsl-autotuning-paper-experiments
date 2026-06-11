// RUN: libxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p x86-prologue-epilogue-insertion -t x86-asm | filecheck %s
// RUN: libxsmm-gemm dense %t matmul_bac 10 5 16 10 16 10 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir | filecheck %s --check-prefix CHECK-IR-LIBXSMM

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push r15
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  [[SCF_N_BODY:^\S+]]:
// CHECK-NEXT:      add r11, 5
// CHECK-NEXT:      mov r15, 3
// CHECK-NEXT:      kmovb k1, r15d
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  [[SCF_M_BODY:^\S+]]:
// CHECK-NEXT:      add r10, 10
// CHECK-NEXT:      vmovupd zmm22, [rdx]
// CHECK-NEXT:      vmovupd zmm23 {k1}{z}, [rdx+64]
// CHECK-NEXT:      vmovupd zmm24, [rdx+80]
// CHECK-NEXT:      vmovupd zmm25 {k1}{z}, [rdx+144]
// CHECK-NEXT:      vmovupd zmm26, [rdx+160]
// CHECK-NEXT:      vmovupd zmm27 {k1}{z}, [rdx+224]
// CHECK-NEXT:      vmovupd zmm28, [rdx+240]
// CHECK-NEXT:      vmovupd zmm29 {k1}{z}, [rdx+304]
// CHECK-NEXT:      vmovupd zmm30, [rdx+320]
// CHECK-NEXT:      vmovupd zmm31 {k1}{z}, [rdx+384]
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd zmm1, [rdi]
// CHECK-NEXT:      vmovupd zmm2 {k1}{z}, [rdi+64]
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi]
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm23, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+128]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm25, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+256]
// CHECK-NEXT:      vfmadd231pd zmm26, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm27, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+384]
// CHECK-NEXT:      vfmadd231pd zmm28, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm29, zmm2, zmm0
// CHECK-NEXT:      vbroadcastsd zmm0, [rsi+512]
// CHECK-NEXT:      add rsi, 8
// CHECK-NEXT:      add rdi, 80
// CHECK-NEXT:      vfmadd231pd zmm30, zmm1, zmm0
// CHECK-NEXT:      vfmadd231pd zmm31, zmm2, zmm0
// CHECK-NEXT:      vmovupd [rdx], zmm22
// CHECK-NEXT:      vmovupd [rdx+64] {k1}, zmm23
// CHECK-NEXT:      vmovupd [rdx+80], zmm24
// CHECK-NEXT:      vmovupd [rdx+144] {k1}, zmm25
// CHECK-NEXT:      vmovupd [rdx+160], zmm26
// CHECK-NEXT:      vmovupd [rdx+224] {k1}, zmm27
// CHECK-NEXT:      vmovupd [rdx+240], zmm28
// CHECK-NEXT:      vmovupd [rdx+304] {k1}, zmm29
// CHECK-NEXT:      vmovupd [rdx+320], zmm30
// CHECK-NEXT:      vmovupd [rdx+384] {k1}, zmm31
// CHECK-NEXT:      add rdx, 80
// CHECK-NEXT:      sub rdi, 1200
// CHECK-NEXT:      cmp r10, 10
// CHECK-NEXT:      jl [[SCF_M_BODY]]
// CHECK-NEXT:      add rdx, 320
// CHECK-NEXT:      add rsi, 640
// CHECK-NEXT:      sub rdi, 80
// CHECK-NEXT:      cmp r11, 5
// CHECK-NEXT:      jl [[SCF_N_BODY]]
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop r15
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret

// CHECK-IR-LIBXSMM:       builtin.module {
// CHECK-IR-LIBXSMM-NEXT:    x86_func.func public @matmul_bac(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>, %2: !x86.reg64<rdx>) {
// CHECK-IR-LIBXSMM-NEXT:      %3 = x86.get_register : !x86.reg64<rbp>
// CHECK-IR-LIBXSMM-NEXT:      %4 = x86.get_register : !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %5 = x86.s.push %4, %3 : (!x86.reg64<rsp>, !x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %6 = x86.ds.mov %5 : (!x86.reg64<rsp>) -> !x86.reg64<rbp>
// CHECK-IR-LIBXSMM-NEXT:      %7 = x86.ri.sub %5, 192 : (!x86.reg64<rsp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %8 = x86.di.mov -64 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %9 = x86.rs.and %7, %8 : (!x86.reg64<rsp>, !x86.reg64<r10>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %10 = x86.di.mov 0 : () -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb0(%0 : !x86.reg64<rdi>, %1 : !x86.reg64<rsi>, %2 : !x86.reg64<rdx>, %6 : !x86.reg64<rbp>, %9 : !x86.reg64<rsp>, %10 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb0(%11: !x86.reg64<rdi>, %12: !x86.reg64<rsi>, %13: !x86.reg64<rdx>, %14: !x86.reg64<rbp>, %15: !x86.reg64<rsp>, %16: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l33"
// CHECK-IR-LIBXSMM-NEXT:      %17 = x86.ri.add %16, 5 : (!x86.reg64<r11>) -> !x86.reg64<r11>
// CHECK-IR-LIBXSMM-NEXT:      %18 = x86.di.mov 3 : () -> !x86.reg64<r15>
// CHECK-IR-LIBXSMM-NEXT:      %19 = x86.ks.kmovb %18 : (!x86.reg64<r15>) -> !x86.avx512maskreg<k1>
// CHECK-IR-LIBXSMM-NEXT:      %20 = x86.di.mov 0 : () -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      x86.fallthrough ^bb1(%11 : !x86.reg64<rdi>, %12 : !x86.reg64<rsi>, %13 : !x86.reg64<rdx>, %14 : !x86.reg64<rbp>, %15 : !x86.reg64<rsp>, %17 : !x86.reg64<r11>, %18 : !x86.reg64<r15>, %19 : !x86.avx512maskreg<k1>, %20 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb1(%21: !x86.reg64<rdi>, %22: !x86.reg64<rsi>, %23: !x86.reg64<rdx>, %24: !x86.reg64<rbp>, %25: !x86.reg64<rsp>, %26: !x86.reg64<r11>, %27: !x86.reg64<r15>, %28: !x86.avx512maskreg<k1>, %29: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      x86.label "l34"
// CHECK-IR-LIBXSMM-NEXT:      %30 = x86.ri.add %29, 10 : (!x86.reg64<r10>) -> !x86.reg64<r10>
// CHECK-IR-LIBXSMM-NEXT:      %31 = x86.dm.vmovupd [%23] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %32 = x86.dmk.vmovupd[%23 + 64], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %33 = x86.dm.vmovupd [%23 + 80] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %34 = x86.dmk.vmovupd[%23 + 144], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %35 = x86.dm.vmovupd [%23 + 160] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %36 = x86.dmk.vmovupd[%23 + 224], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %37 = x86.dm.vmovupd [%23 + 240] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %38 = x86.dmk.vmovupd[%23 + 304], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %39 = x86.dm.vmovupd [%23 + 320] : (!x86.reg64<rdx>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %40 = x86.dmk.vmovupd[%23 + 384], %28 {z} : (!x86.reg64<rdx>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %41 = x86.dm.vmovupd [%21] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %42 = x86.dmk.vmovupd[%21 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %43 = x86.dm.vbroadcastsd [%22] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %44 = x86.rss.vfmadd231pd %31, %41, %43 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %45 = x86.rss.vfmadd231pd %32, %42, %43 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %46 = x86.dm.vbroadcastsd [%22 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %47 = x86.rss.vfmadd231pd %33, %41, %46 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %48 = x86.rss.vfmadd231pd %34, %42, %46 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %49 = x86.dm.vbroadcastsd [%22 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %50 = x86.rss.vfmadd231pd %35, %41, %49 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %51 = x86.rss.vfmadd231pd %36, %42, %49 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %52 = x86.dm.vbroadcastsd [%22 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %53 = x86.rss.vfmadd231pd %37, %41, %52 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %54 = x86.rss.vfmadd231pd %38, %42, %52 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %55 = x86.dm.vbroadcastsd [%22 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %56 = x86.ri.add %22, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %57 = x86.ri.add %21, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %58 = x86.rss.vfmadd231pd %39, %41, %55 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %59 = x86.rss.vfmadd231pd %40, %42, %55 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %60 = x86.dm.vmovupd [%57] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %61 = x86.dmk.vmovupd[%57 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %62 = x86.dm.vbroadcastsd [%56] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %63 = x86.rss.vfmadd231pd %44, %60, %62 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %64 = x86.rss.vfmadd231pd %45, %61, %62 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %65 = x86.dm.vbroadcastsd [%56 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %66 = x86.rss.vfmadd231pd %47, %60, %65 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %67 = x86.rss.vfmadd231pd %48, %61, %65 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %68 = x86.dm.vbroadcastsd [%56 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %69 = x86.rss.vfmadd231pd %50, %60, %68 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %70 = x86.rss.vfmadd231pd %51, %61, %68 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %71 = x86.dm.vbroadcastsd [%56 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %72 = x86.rss.vfmadd231pd %53, %60, %71 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %73 = x86.rss.vfmadd231pd %54, %61, %71 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %74 = x86.dm.vbroadcastsd [%56 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %75 = x86.ri.add %56, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %76 = x86.ri.add %57, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %77 = x86.rss.vfmadd231pd %58, %60, %74 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %78 = x86.rss.vfmadd231pd %59, %61, %74 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %79 = x86.dm.vmovupd [%76] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %80 = x86.dmk.vmovupd[%76 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %81 = x86.dm.vbroadcastsd [%75] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %82 = x86.rss.vfmadd231pd %63, %79, %81 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %83 = x86.rss.vfmadd231pd %64, %80, %81 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %84 = x86.dm.vbroadcastsd [%75 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %85 = x86.rss.vfmadd231pd %66, %79, %84 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %86 = x86.rss.vfmadd231pd %67, %80, %84 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %87 = x86.dm.vbroadcastsd [%75 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %88 = x86.rss.vfmadd231pd %69, %79, %87 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %89 = x86.rss.vfmadd231pd %70, %80, %87 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %90 = x86.dm.vbroadcastsd [%75 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %91 = x86.rss.vfmadd231pd %72, %79, %90 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %92 = x86.rss.vfmadd231pd %73, %80, %90 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %93 = x86.dm.vbroadcastsd [%75 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %94 = x86.ri.add %75, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %95 = x86.ri.add %76, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %96 = x86.rss.vfmadd231pd %77, %79, %93 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %97 = x86.rss.vfmadd231pd %78, %80, %93 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %98 = x86.dm.vmovupd [%95] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %99 = x86.dmk.vmovupd[%95 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %100 = x86.dm.vbroadcastsd [%94] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %101 = x86.rss.vfmadd231pd %82, %98, %100 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %102 = x86.rss.vfmadd231pd %83, %99, %100 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %103 = x86.dm.vbroadcastsd [%94 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %104 = x86.rss.vfmadd231pd %85, %98, %103 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %105 = x86.rss.vfmadd231pd %86, %99, %103 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %106 = x86.dm.vbroadcastsd [%94 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %107 = x86.rss.vfmadd231pd %88, %98, %106 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %108 = x86.rss.vfmadd231pd %89, %99, %106 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %109 = x86.dm.vbroadcastsd [%94 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %110 = x86.rss.vfmadd231pd %91, %98, %109 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %111 = x86.rss.vfmadd231pd %92, %99, %109 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %112 = x86.dm.vbroadcastsd [%94 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %113 = x86.ri.add %94, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %114 = x86.ri.add %95, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %115 = x86.rss.vfmadd231pd %96, %98, %112 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %116 = x86.rss.vfmadd231pd %97, %99, %112 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %117 = x86.dm.vmovupd [%114] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %118 = x86.dmk.vmovupd[%114 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %119 = x86.dm.vbroadcastsd [%113] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %120 = x86.rss.vfmadd231pd %101, %117, %119 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %121 = x86.rss.vfmadd231pd %102, %118, %119 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %122 = x86.dm.vbroadcastsd [%113 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %123 = x86.rss.vfmadd231pd %104, %117, %122 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %124 = x86.rss.vfmadd231pd %105, %118, %122 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %125 = x86.dm.vbroadcastsd [%113 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %126 = x86.rss.vfmadd231pd %107, %117, %125 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %127 = x86.rss.vfmadd231pd %108, %118, %125 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %128 = x86.dm.vbroadcastsd [%113 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %129 = x86.rss.vfmadd231pd %110, %117, %128 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %130 = x86.rss.vfmadd231pd %111, %118, %128 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %131 = x86.dm.vbroadcastsd [%113 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %132 = x86.ri.add %113, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %133 = x86.ri.add %114, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %134 = x86.rss.vfmadd231pd %115, %117, %131 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %135 = x86.rss.vfmadd231pd %116, %118, %131 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %136 = x86.dm.vmovupd [%133] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %137 = x86.dmk.vmovupd[%133 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %138 = x86.dm.vbroadcastsd [%132] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %139 = x86.rss.vfmadd231pd %120, %136, %138 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %140 = x86.rss.vfmadd231pd %121, %137, %138 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %141 = x86.dm.vbroadcastsd [%132 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %142 = x86.rss.vfmadd231pd %123, %136, %141 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %143 = x86.rss.vfmadd231pd %124, %137, %141 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %144 = x86.dm.vbroadcastsd [%132 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %145 = x86.rss.vfmadd231pd %126, %136, %144 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %146 = x86.rss.vfmadd231pd %127, %137, %144 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %147 = x86.dm.vbroadcastsd [%132 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %148 = x86.rss.vfmadd231pd %129, %136, %147 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %149 = x86.rss.vfmadd231pd %130, %137, %147 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %150 = x86.dm.vbroadcastsd [%132 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %151 = x86.ri.add %132, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %152 = x86.ri.add %133, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %153 = x86.rss.vfmadd231pd %134, %136, %150 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %154 = x86.rss.vfmadd231pd %135, %137, %150 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %155 = x86.dm.vmovupd [%152] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %156 = x86.dmk.vmovupd[%152 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %157 = x86.dm.vbroadcastsd [%151] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %158 = x86.rss.vfmadd231pd %139, %155, %157 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %159 = x86.rss.vfmadd231pd %140, %156, %157 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %160 = x86.dm.vbroadcastsd [%151 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %161 = x86.rss.vfmadd231pd %142, %155, %160 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %162 = x86.rss.vfmadd231pd %143, %156, %160 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %163 = x86.dm.vbroadcastsd [%151 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %164 = x86.rss.vfmadd231pd %145, %155, %163 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %165 = x86.rss.vfmadd231pd %146, %156, %163 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %166 = x86.dm.vbroadcastsd [%151 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %167 = x86.rss.vfmadd231pd %148, %155, %166 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %168 = x86.rss.vfmadd231pd %149, %156, %166 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %169 = x86.dm.vbroadcastsd [%151 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %170 = x86.ri.add %151, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %171 = x86.ri.add %152, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %172 = x86.rss.vfmadd231pd %153, %155, %169 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %173 = x86.rss.vfmadd231pd %154, %156, %169 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %174 = x86.dm.vmovupd [%171] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %175 = x86.dmk.vmovupd[%171 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %176 = x86.dm.vbroadcastsd [%170] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %177 = x86.rss.vfmadd231pd %158, %174, %176 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %178 = x86.rss.vfmadd231pd %159, %175, %176 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %179 = x86.dm.vbroadcastsd [%170 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %180 = x86.rss.vfmadd231pd %161, %174, %179 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %181 = x86.rss.vfmadd231pd %162, %175, %179 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %182 = x86.dm.vbroadcastsd [%170 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %183 = x86.rss.vfmadd231pd %164, %174, %182 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %184 = x86.rss.vfmadd231pd %165, %175, %182 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %185 = x86.dm.vbroadcastsd [%170 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %186 = x86.rss.vfmadd231pd %167, %174, %185 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %187 = x86.rss.vfmadd231pd %168, %175, %185 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %188 = x86.dm.vbroadcastsd [%170 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %189 = x86.ri.add %170, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %190 = x86.ri.add %171, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %191 = x86.rss.vfmadd231pd %172, %174, %188 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %192 = x86.rss.vfmadd231pd %173, %175, %188 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %193 = x86.dm.vmovupd [%190] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %194 = x86.dmk.vmovupd[%190 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %195 = x86.dm.vbroadcastsd [%189] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %196 = x86.rss.vfmadd231pd %177, %193, %195 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %197 = x86.rss.vfmadd231pd %178, %194, %195 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %198 = x86.dm.vbroadcastsd [%189 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %199 = x86.rss.vfmadd231pd %180, %193, %198 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %200 = x86.rss.vfmadd231pd %181, %194, %198 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %201 = x86.dm.vbroadcastsd [%189 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %202 = x86.rss.vfmadd231pd %183, %193, %201 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %203 = x86.rss.vfmadd231pd %184, %194, %201 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %204 = x86.dm.vbroadcastsd [%189 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %205 = x86.rss.vfmadd231pd %186, %193, %204 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %206 = x86.rss.vfmadd231pd %187, %194, %204 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %207 = x86.dm.vbroadcastsd [%189 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %208 = x86.ri.add %189, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %209 = x86.ri.add %190, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %210 = x86.rss.vfmadd231pd %191, %193, %207 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %211 = x86.rss.vfmadd231pd %192, %194, %207 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %212 = x86.dm.vmovupd [%209] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %213 = x86.dmk.vmovupd[%209 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %214 = x86.dm.vbroadcastsd [%208] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %215 = x86.rss.vfmadd231pd %196, %212, %214 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %216 = x86.rss.vfmadd231pd %197, %213, %214 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %217 = x86.dm.vbroadcastsd [%208 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %218 = x86.rss.vfmadd231pd %199, %212, %217 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %219 = x86.rss.vfmadd231pd %200, %213, %217 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %220 = x86.dm.vbroadcastsd [%208 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %221 = x86.rss.vfmadd231pd %202, %212, %220 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %222 = x86.rss.vfmadd231pd %203, %213, %220 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %223 = x86.dm.vbroadcastsd [%208 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %224 = x86.rss.vfmadd231pd %205, %212, %223 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %225 = x86.rss.vfmadd231pd %206, %213, %223 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %226 = x86.dm.vbroadcastsd [%208 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %227 = x86.ri.add %208, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %228 = x86.ri.add %209, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %229 = x86.rss.vfmadd231pd %210, %212, %226 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %230 = x86.rss.vfmadd231pd %211, %213, %226 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %231 = x86.dm.vmovupd [%228] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %232 = x86.dmk.vmovupd[%228 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %233 = x86.dm.vbroadcastsd [%227] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %234 = x86.rss.vfmadd231pd %215, %231, %233 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %235 = x86.rss.vfmadd231pd %216, %232, %233 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %236 = x86.dm.vbroadcastsd [%227 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %237 = x86.rss.vfmadd231pd %218, %231, %236 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %238 = x86.rss.vfmadd231pd %219, %232, %236 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %239 = x86.dm.vbroadcastsd [%227 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %240 = x86.rss.vfmadd231pd %221, %231, %239 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %241 = x86.rss.vfmadd231pd %222, %232, %239 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %242 = x86.dm.vbroadcastsd [%227 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %243 = x86.rss.vfmadd231pd %224, %231, %242 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %244 = x86.rss.vfmadd231pd %225, %232, %242 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %245 = x86.dm.vbroadcastsd [%227 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %246 = x86.ri.add %227, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %247 = x86.ri.add %228, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %248 = x86.rss.vfmadd231pd %229, %231, %245 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %249 = x86.rss.vfmadd231pd %230, %232, %245 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %250 = x86.dm.vmovupd [%247] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %251 = x86.dmk.vmovupd[%247 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %252 = x86.dm.vbroadcastsd [%246] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %253 = x86.rss.vfmadd231pd %234, %250, %252 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %254 = x86.rss.vfmadd231pd %235, %251, %252 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %255 = x86.dm.vbroadcastsd [%246 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %256 = x86.rss.vfmadd231pd %237, %250, %255 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %257 = x86.rss.vfmadd231pd %238, %251, %255 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %258 = x86.dm.vbroadcastsd [%246 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %259 = x86.rss.vfmadd231pd %240, %250, %258 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %260 = x86.rss.vfmadd231pd %241, %251, %258 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %261 = x86.dm.vbroadcastsd [%246 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %262 = x86.rss.vfmadd231pd %243, %250, %261 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %263 = x86.rss.vfmadd231pd %244, %251, %261 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %264 = x86.dm.vbroadcastsd [%246 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %265 = x86.ri.add %246, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %266 = x86.ri.add %247, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %267 = x86.rss.vfmadd231pd %248, %250, %264 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %268 = x86.rss.vfmadd231pd %249, %251, %264 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %269 = x86.dm.vmovupd [%266] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %270 = x86.dmk.vmovupd[%266 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %271 = x86.dm.vbroadcastsd [%265] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %272 = x86.rss.vfmadd231pd %253, %269, %271 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %273 = x86.rss.vfmadd231pd %254, %270, %271 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %274 = x86.dm.vbroadcastsd [%265 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %275 = x86.rss.vfmadd231pd %256, %269, %274 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %276 = x86.rss.vfmadd231pd %257, %270, %274 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %277 = x86.dm.vbroadcastsd [%265 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %278 = x86.rss.vfmadd231pd %259, %269, %277 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %279 = x86.rss.vfmadd231pd %260, %270, %277 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %280 = x86.dm.vbroadcastsd [%265 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %281 = x86.rss.vfmadd231pd %262, %269, %280 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %282 = x86.rss.vfmadd231pd %263, %270, %280 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %283 = x86.dm.vbroadcastsd [%265 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %284 = x86.ri.add %265, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %285 = x86.ri.add %266, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %286 = x86.rss.vfmadd231pd %267, %269, %283 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %287 = x86.rss.vfmadd231pd %268, %270, %283 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %288 = x86.dm.vmovupd [%285] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %289 = x86.dmk.vmovupd[%285 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %290 = x86.dm.vbroadcastsd [%284] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %291 = x86.rss.vfmadd231pd %272, %288, %290 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %292 = x86.rss.vfmadd231pd %273, %289, %290 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %293 = x86.dm.vbroadcastsd [%284 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %294 = x86.rss.vfmadd231pd %275, %288, %293 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %295 = x86.rss.vfmadd231pd %276, %289, %293 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %296 = x86.dm.vbroadcastsd [%284 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %297 = x86.rss.vfmadd231pd %278, %288, %296 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %298 = x86.rss.vfmadd231pd %279, %289, %296 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %299 = x86.dm.vbroadcastsd [%284 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %300 = x86.rss.vfmadd231pd %281, %288, %299 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %301 = x86.rss.vfmadd231pd %282, %289, %299 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %302 = x86.dm.vbroadcastsd [%284 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %303 = x86.ri.add %284, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %304 = x86.ri.add %285, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %305 = x86.rss.vfmadd231pd %286, %288, %302 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %306 = x86.rss.vfmadd231pd %287, %289, %302 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %307 = x86.dm.vmovupd [%304] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %308 = x86.dmk.vmovupd[%304 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %309 = x86.dm.vbroadcastsd [%303] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %310 = x86.rss.vfmadd231pd %291, %307, %309 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %311 = x86.rss.vfmadd231pd %292, %308, %309 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %312 = x86.dm.vbroadcastsd [%303 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %313 = x86.rss.vfmadd231pd %294, %307, %312 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %314 = x86.rss.vfmadd231pd %295, %308, %312 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %315 = x86.dm.vbroadcastsd [%303 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %316 = x86.rss.vfmadd231pd %297, %307, %315 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %317 = x86.rss.vfmadd231pd %298, %308, %315 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %318 = x86.dm.vbroadcastsd [%303 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %319 = x86.rss.vfmadd231pd %300, %307, %318 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %320 = x86.rss.vfmadd231pd %301, %308, %318 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %321 = x86.dm.vbroadcastsd [%303 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %322 = x86.ri.add %303, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %323 = x86.ri.add %304, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %324 = x86.rss.vfmadd231pd %305, %307, %321 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %325 = x86.rss.vfmadd231pd %306, %308, %321 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      %326 = x86.dm.vmovupd [%323] : (!x86.reg64<rdi>) -> !x86.avx512reg<zmm1>
// CHECK-IR-LIBXSMM-NEXT:      %327 = x86.dmk.vmovupd[%323 + 64], %28 {z} : (!x86.reg64<rdi>, !x86.avx512maskreg<k1>) -> !x86.avx512reg<zmm2>
// CHECK-IR-LIBXSMM-NEXT:      %328 = x86.dm.vbroadcastsd [%322] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %329 = x86.rss.vfmadd231pd %310, %326, %328 : (!x86.avx512reg<zmm22>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm22>
// CHECK-IR-LIBXSMM-NEXT:      %330 = x86.rss.vfmadd231pd %311, %327, %328 : (!x86.avx512reg<zmm23>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm23>
// CHECK-IR-LIBXSMM-NEXT:      %331 = x86.dm.vbroadcastsd [%322 + 128] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %332 = x86.rss.vfmadd231pd %313, %326, %331 : (!x86.avx512reg<zmm24>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm24>
// CHECK-IR-LIBXSMM-NEXT:      %333 = x86.rss.vfmadd231pd %314, %327, %331 : (!x86.avx512reg<zmm25>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm25>
// CHECK-IR-LIBXSMM-NEXT:      %334 = x86.dm.vbroadcastsd [%322 + 256] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %335 = x86.rss.vfmadd231pd %316, %326, %334 : (!x86.avx512reg<zmm26>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm26>
// CHECK-IR-LIBXSMM-NEXT:      %336 = x86.rss.vfmadd231pd %317, %327, %334 : (!x86.avx512reg<zmm27>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm27>
// CHECK-IR-LIBXSMM-NEXT:      %337 = x86.dm.vbroadcastsd [%322 + 384] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %338 = x86.rss.vfmadd231pd %319, %326, %337 : (!x86.avx512reg<zmm28>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm28>
// CHECK-IR-LIBXSMM-NEXT:      %339 = x86.rss.vfmadd231pd %320, %327, %337 : (!x86.avx512reg<zmm29>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm29>
// CHECK-IR-LIBXSMM-NEXT:      %340 = x86.dm.vbroadcastsd [%322 + 512] : (!x86.reg64<rsi>) -> !x86.avx512reg<zmm0>
// CHECK-IR-LIBXSMM-NEXT:      %341 = x86.ri.add %322, 8 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %342 = x86.ri.add %323, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %343 = x86.rss.vfmadd231pd %324, %326, %340 : (!x86.avx512reg<zmm30>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm30>
// CHECK-IR-LIBXSMM-NEXT:      %344 = x86.rss.vfmadd231pd %325, %327, %340 : (!x86.avx512reg<zmm31>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm31>
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%23], %329 : (!x86.reg64<rdx>, !x86.avx512reg<zmm22>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 64], %330, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm23>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%23 + 80], %332 : (!x86.reg64<rdx>, !x86.avx512reg<zmm24>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 144], %333, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm25>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%23 + 160], %335 : (!x86.reg64<rdx>, !x86.avx512reg<zmm26>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 224], %336, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm27>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%23 + 240], %338 : (!x86.reg64<rdx>, !x86.avx512reg<zmm28>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 304], %339, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm29>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.ms.vmovupd [%23 + 320], %343 : (!x86.reg64<rdx>, !x86.avx512reg<zmm30>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      x86.msk.vmovupd[%23 + 384], %344, %28 : (!x86.reg64<rdx>, !x86.avx512reg<zmm31>, !x86.avx512maskreg<k1>) -> ()
// CHECK-IR-LIBXSMM-NEXT:      %345 = x86.ri.add %23, 80 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %346 = x86.ri.sub %342, 1200 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %347 = x86.si.cmp %30, 10 : (!x86.reg64<r10>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %347 : !x86.rflags<rflags>, ^bb1(%346 : !x86.reg64<rdi>, %341 : !x86.reg64<rsi>, %345 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %27 : !x86.reg64<r15>, %28 : !x86.avx512maskreg<k1>, %30 : !x86.reg64<r10>), ^bb2(%346 : !x86.reg64<rdi>, %341 : !x86.reg64<rsi>, %345 : !x86.reg64<rdx>, %24 : !x86.reg64<rbp>, %25 : !x86.reg64<rsp>, %26 : !x86.reg64<r11>, %27 : !x86.reg64<r15>, %28 : !x86.avx512maskreg<k1>, %30 : !x86.reg64<r10>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb2(%348: !x86.reg64<rdi>, %349: !x86.reg64<rsi>, %350: !x86.reg64<rdx>, %351: !x86.reg64<rbp>, %352: !x86.reg64<rsp>, %353: !x86.reg64<r11>, %354: !x86.reg64<r15>, %355: !x86.avx512maskreg<k1>, %356: !x86.reg64<r10>):
// CHECK-IR-LIBXSMM-NEXT:      %357 = x86.ri.add %350, 320 : (!x86.reg64<rdx>) -> !x86.reg64<rdx>
// CHECK-IR-LIBXSMM-NEXT:      %358 = x86.ri.add %349, 640 : (!x86.reg64<rsi>) -> !x86.reg64<rsi>
// CHECK-IR-LIBXSMM-NEXT:      %359 = x86.ri.sub %348, 80 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
// CHECK-IR-LIBXSMM-NEXT:      %360 = x86.si.cmp %353, 5 : (!x86.reg64<r11>) -> !x86.rflags<rflags>
// CHECK-IR-LIBXSMM-NEXT:      x86.c.jl %360 : !x86.rflags<rflags>, ^bb0(%359 : !x86.reg64<rdi>, %358 : !x86.reg64<rsi>, %357 : !x86.reg64<rdx>, %351 : !x86.reg64<rbp>, %352 : !x86.reg64<rsp>, %353 : !x86.reg64<r11>), ^bb3(%359 : !x86.reg64<rdi>, %358 : !x86.reg64<rsi>, %357 : !x86.reg64<rdx>, %351 : !x86.reg64<rbp>, %352 : !x86.reg64<rsp>, %353 : !x86.reg64<r11>)
// CHECK-IR-LIBXSMM-NEXT:    ^bb3(%361: !x86.reg64<rdi>, %362: !x86.reg64<rsi>, %363: !x86.reg64<rdx>, %364: !x86.reg64<rbp>, %365: !x86.reg64<rsp>, %366: !x86.reg64<r11>):
// CHECK-IR-LIBXSMM-NEXT:      %367 = x86.ds.mov %364 : (!x86.reg64<rbp>) -> !x86.reg64<rsp>
// CHECK-IR-LIBXSMM-NEXT:      %368, %369 = x86.d.pop %367 : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<rbp>)
// CHECK-IR-LIBXSMM-NEXT:      x86_func.ret
// CHECK-IR-LIBXSMM-NEXT:    }
// CHECK-IR-LIBXSMM-NEXT:  }

