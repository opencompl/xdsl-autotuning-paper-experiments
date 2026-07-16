// RUN: libxsmm-gemm dense %t matmul_bac 8 8 8 8 8 8 1 1 0 0 skx nopf DP && xdsl-opt %t -f mlir -p x86-regalloc-verify-liveness,x86-prologue-epilogue-insertion -t x86-asm | filecheck %s

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul_bac
// CHECK-NEXT:  matmul_bac:
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      push rbp
// CHECK-NEXT:      mov rbp, rsp
// CHECK-NEXT:      sub rsp, 192
// CHECK-NEXT:      mov r10, -64
// CHECK-NEXT:      and rsp, r10
// CHECK-NEXT:      mov r11, 0
// CHECK-NEXT:  l33:
// CHECK-NEXT:      add r11, 8
// CHECK-NEXT:      mov r10, 0
// CHECK-NEXT:  l34:
// CHECK-NEXT:      add r10, 8
// CHECK-NEXT:      vmovupd zmm24, [rdx]
// CHECK-NEXT:      vmovupd zmm25, [rdx+64]
// CHECK-NEXT:      vmovupd zmm26, [rdx+128]
// CHECK-NEXT:      vmovupd zmm27, [rdx+192]
// CHECK-NEXT:      vmovupd zmm28, [rdx+256]
// CHECK-NEXT:      vmovupd zmm29, [rdx+320]
// CHECK-NEXT:      vmovupd zmm30, [rdx+384]
// CHECK-NEXT:      vmovupd zmm31, [rdx+448]
// CHECK-NEXT:      vpxord zmm16, zmm16, zmm16
// CHECK-NEXT:      vpxord zmm17, zmm17, zmm17
// CHECK-NEXT:      vpxord zmm18, zmm18, zmm18
// CHECK-NEXT:      vpxord zmm19, zmm19, zmm19
// CHECK-NEXT:      vpxord zmm20, zmm20, zmm20
// CHECK-NEXT:      vpxord zmm21, zmm21, zmm21
// CHECK-NEXT:      vpxord zmm22, zmm22, zmm22
// CHECK-NEXT:      vpxord zmm23, zmm23, zmm23
// CHECK-NEXT:      vmovupd zmm0, [rdi]
// CHECK-NEXT:      vmovupd zmm1, [rdi+64]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm0, [rsi]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+64]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+128]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+192]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+256]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+320]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+384]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+448]{1to8}
// CHECK-NEXT:      vmovupd zmm0, [rdi+128]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+8]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+72]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+136]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+200]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+264]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+328]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+392]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+456]{1to8}
// CHECK-NEXT:      vmovupd zmm1, [rdi+192]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+16]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+80]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+144]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+208]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+272]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+336]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+400]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+464]{1to8}
// CHECK-NEXT:      vmovupd zmm0, [rdi+256]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+24]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+88]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+152]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+216]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+280]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+344]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+408]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+472]{1to8}
// CHECK-NEXT:      vmovupd zmm1, [rdi+320]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+32]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+96]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+160]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+224]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+288]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+352]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+416]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+480]{1to8}
// CHECK-NEXT:      vmovupd zmm0, [rdi+384]
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+40]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+104]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+168]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+232]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+296]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+360]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+424]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+488]{1to8}
// CHECK-NEXT:      vmovupd zmm1, [rdi+448]
// CHECK-NEXT:      vfmadd231pd zmm24, zmm0, [rsi+48]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm25, zmm0, [rsi+112]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm26, zmm0, [rsi+176]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm27, zmm0, [rsi+240]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm28, zmm0, [rsi+304]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm29, zmm0, [rsi+368]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm30, zmm0, [rsi+432]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm31, zmm0, [rsi+496]{1to8}
// CHECK-NEXT:      add rdi, 512
// CHECK-NEXT:      vfmadd231pd zmm16, zmm1, [rsi+56]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm17, zmm1, [rsi+120]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm18, zmm1, [rsi+184]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm19, zmm1, [rsi+248]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm20, zmm1, [rsi+312]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm21, zmm1, [rsi+376]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm22, zmm1, [rsi+440]{1to8}
// CHECK-NEXT:      vfmadd231pd zmm23, zmm1, [rsi+504]{1to8}
// CHECK-NEXT:      add rsi, 64
// CHECK-NEXT:      vaddpd zmm24, zmm16, zmm24
// CHECK-NEXT:      vaddpd zmm25, zmm17, zmm25
// CHECK-NEXT:      vaddpd zmm26, zmm18, zmm26
// CHECK-NEXT:      vaddpd zmm27, zmm19, zmm27
// CHECK-NEXT:      vaddpd zmm28, zmm20, zmm28
// CHECK-NEXT:      vaddpd zmm29, zmm21, zmm29
// CHECK-NEXT:      vaddpd zmm30, zmm22, zmm30
// CHECK-NEXT:      vaddpd zmm31, zmm23, zmm31
// CHECK-NEXT:      vmovupd [rdx], zmm24
// CHECK-NEXT:      vmovupd [rdx+64], zmm25
// CHECK-NEXT:      vmovupd [rdx+128], zmm26
// CHECK-NEXT:      vmovupd [rdx+192], zmm27
// CHECK-NEXT:      vmovupd [rdx+256], zmm28
// CHECK-NEXT:      vmovupd [rdx+320], zmm29
// CHECK-NEXT:      vmovupd [rdx+384], zmm30
// CHECK-NEXT:      vmovupd [rdx+448], zmm31
// CHECK-NEXT:      add rdx, 64
// CHECK-NEXT:      sub rdi, 448
// CHECK-NEXT:      cmp r10, 8
// CHECK-NEXT:      jl l34
// CHECK-NEXT:      add rdx, 448
// CHECK-NEXT:      add rsi, 512
// CHECK-NEXT:      sub rdi, 64
// CHECK-NEXT:      cmp r11, 8
// CHECK-NEXT:      jl l33
// CHECK-NEXT:      mov rsp, rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      pop rbp
// CHECK-NEXT:      ret
