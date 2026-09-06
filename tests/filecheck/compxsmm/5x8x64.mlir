// RUN: compxsmm-gemm dense %t matmul_bac 8 5 64 8 64 8 1 1 1 1 skx nopf DP && xdsl-opt %t -f mlir -p COMPXSMM_MANUAL_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefixes CHECK-REGALLOC-STRUCTURE,CHECK-MANUAL-REGALLOC
// RUN: compxsmm-gemm dense %t matmul_bac 8 5 64 8 64 8 1 1 1 1 skx nopf DP --disable-regalloc && xdsl-opt %t -f mlir -p COMPXSMM_AUTO_REGALLOC_PIPELINE -t x86-asm | filecheck %s --check-prefix CHECK-REGALLOC-STRUCTURE

// This exercises the unmasked double-precision fsdbcst path with four accumulator
// sets and a K loop. The shared check verifies that automatic allocation preserves
// the manual pipeline's instruction structure and accumulator reuse relationships.

// CHECK-REGALLOC-STRUCTURE:       .intel_syntax noprefix
// CHECK-REGALLOC-STRUCTURE-NEXT:  .text
// CHECK-REGALLOC-STRUCTURE-NEXT:  .globl matmul_bac
// CHECK-REGALLOC-STRUCTURE-NEXT:  matmul_bac:
// CHECK-MANUAL-REGALLOC-NEXT:      push r12
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC0:\S+]], [rdx]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC1:\S+]], [rdx+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC2:\S+]], [rdx+128]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC3:\S+]], [rdx+192]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[ACC4:\S+]], [rdx+256]
// CHECK-REGALLOC-STRUCTURE-NEXT:      mov [[K:\S+]], 0
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
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0:\S+]], [rdi]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1:\S+]], [rdi+64]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC0]], [[A0]], [rsi]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC1]], [[A0]], [rsi+512]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC2]], [[A0]], [rsi+1024]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC3]], [[A0]], [rsi+1536]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC4]], [[A0]], [rsi+2048]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A0]], [rdi+128]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC10]], [[A1]], [rsi+8]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC11]], [[A1]], [rsi+520]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC12]], [[A1]], [rsi+1032]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC13]], [[A1]], [rsi+1544]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC14]], [[A1]], [rsi+2056]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [[A1]], [rdi+192]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC20]], [[A0]], [rsi+16]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC21]], [[A0]], [rsi+528]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC22]], [[A0]], [rsi+1040]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC23]], [[A0]], [rsi+1552]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC24]], [[A0]], [rsi+2064]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdi, 256
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC30]], [[A1]], [rsi+24]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC31]], [[A1]], [rsi+536]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC32]], [[A1]], [rsi+1048]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC33]], [[A1]], [rsi+1560]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      vfmadd231pd [[ACC34]], [[A1]], [rsi+2072]{1to8}
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 32
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC0]], [[ACC10]], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC1]], [[ACC11]], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC2]], [[ACC12]], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC3]], [[ACC13]], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC4]], [[ACC14]], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC0]], [[ACC20]], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC1]], [[ACC21]], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC2]], [[ACC22]], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC3]], [[ACC23]], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC4]], [[ACC24]], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC0]], [[ACC30]], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC1]], [[ACC31]], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC2]], [[ACC32]], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC3]], [[ACC33]], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vaddpd [[ACC4]], [[ACC34]], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      cmp [[K]], 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      jl scf_body_0_for
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx], [[ACC0]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+64], [[ACC1]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+128], [[ACC2]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+192], [[ACC3]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      vmovapd [rdx+256], [[ACC4]]
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 4032
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rsi, 512
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      sub rdi, 64
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rsi, 2560
// CHECK-REGALLOC-STRUCTURE-NEXT:      add rdx, 256
// CHECK-MANUAL-REGALLOC-NEXT:      pop r12
// CHECK-REGALLOC-STRUCTURE-NEXT:      ret
