// RUN: compxsmm-gemm dense %t matmul_bac 16 5 64 16 64 16 1 1 1 1 skx nopf SP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE
// RUN: compxsmm-gemm dense %t matmul_bac 16 5 64 16 64 16 1 1 1 1 skx nopf SP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE

// This exercises the unmasked single-precision fsdbcst path with four accumulator
// sets and a K loop, including the vaddps accumulator reduction variant.

// CHECK-REGALLOC-STRUCTURE:       .intel_syntax noprefix
// CHECK-REGALLOC-STRUCTURE-NEXT:  .text
// CHECK-REGALLOC-STRUCTURE-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-STRUCTURE-NEXT:  matmul_bac:
// CHECK-REGALLOC-STRUCTURE-NEXT:      push rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      push [[K:\S+]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      push rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rbp, rsp
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsp, 192
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[STACK_ALIGN:\S+]], -64
// CHECK-REGALLOC-STRUCTURE-NEXT:      and rsp, [[STACK_ALIGN]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[N:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_2_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[N]], 5
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[M:\S+]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_1_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[ACC0:\S+]], [rdx]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[ACC1:\S+]], [rdx+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[ACC2:\S+]], [rdx+128]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[ACC3:\S+]], [rdx+192]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[ACC4:\S+]], [rdx+256]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[K]], 0
// CHECK-REGALLOC-STRUCTURE-NEXT:  scf_body_0_for:
// CHECK-REGALLOC-STRUCTURE-NEXT:      add [[K]], 4
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC10:\S+]], [[ACC10]], [[ACC10]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC11:\S+]], [[ACC11]], [[ACC11]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC12:\S+]], [[ACC12]], [[ACC12]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC13:\S+]], [[ACC13]], [[ACC13]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC14:\S+]], [[ACC14]], [[ACC14]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC20:\S+]], [[ACC20]], [[ACC20]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC21:\S+]], [[ACC21]], [[ACC21]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC22:\S+]], [[ACC22]], [[ACC22]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC23:\S+]], [[ACC23]], [[ACC23]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC24:\S+]], [[ACC24]], [[ACC24]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC30:\S+]], [[ACC30]], [[ACC30]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC31:\S+]], [[ACC31]], [[ACC31]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC32:\S+]], [[ACC32]], [[ACC32]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC33:\S+]], [[ACC33]], [[ACC33]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vpxord [[ACC34:\S+]], [[ACC34]], [[ACC34]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[A0:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[A1:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC0]], [[A0]], [rsi]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC1]], [[A0]], [rsi+256]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC2]], [[A0]], [rsi+512]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC3]], [[A0]], [rsi+768]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC4]], [[A0]], [rsi+1024]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[A0]], [rdi+128]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC10]], [[A1]], [rsi+4]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC11]], [[A1]], [rsi+260]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC12]], [[A1]], [rsi+516]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC13]], [[A1]], [rsi+772]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC14]], [[A1]], [rsi+1028]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [[A1]], [rdi+192]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC20]], [[A0]], [rsi+8]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC21]], [[A0]], [rsi+264]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC22]], [[A0]], [rsi+520]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC23]], [[A0]], [rsi+776]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC24]], [[A0]], [rsi+1032]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 256
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC30]], [[A1]], [rsi+12]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC31]], [[A1]], [rsi+268]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC32]], [[A1]], [rsi+524]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC33]], [[A1]], [rsi+780]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231ps [[ACC34]], [[A1]], [rsi+1036]{1to16}
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC0]], [[ACC10]], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC1]], [[ACC11]], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC2]], [[ACC12]], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC3]], [[ACC13]], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC4]], [[ACC14]], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC0]], [[ACC20]], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC1]], [[ACC21]], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC2]], [[ACC22]], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC3]], [[ACC23]], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC4]], [[ACC24]], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC0]], [[ACC30]], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC1]], [[ACC31]], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC2]], [[ACC32]], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC3]], [[ACC33]], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddps [[ACC4]], [[ACC34]], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[K]], 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [rdx], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [rdx+64], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [rdx+128], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [rdx+192], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovaps [rdx+256], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 4032
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 256
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[M]], 16
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_1_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 1280
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 256
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[N]], 5
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_2_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov rsp, rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop [[K]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      pop rbp
// CHECK-REGALLOC-STRUCTURE-NEXT:      ret
