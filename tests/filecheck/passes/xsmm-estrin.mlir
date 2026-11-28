// RUN: xdsl-opt %s -p 'xsmm-expand-polynomial{schedule=estrin5}' | filecheck %s --check-prefix=TREE
// RUN: xdsl-opt %s -p 'xsmm-expand-polynomial{schedule=estrin5},xsmm-expand-horner' | filecheck %s --check-prefix=ALL
// RUN: xdsl-opt %s -p 'xsmm-expand-polynomial{schedule=estrin5},xsmm-expand-horner,xsmm-expand-polynomial-constants,x86-regalloc-verify-liveness,x86-allocate-registers-with-masks,x86-regalloc-verify-liveness' -t x86-asm | filecheck %s --check-prefix=ASM

!v = !x86.avx512reg
x86_func.func @quintic(%in: !x86.reg64<rdi>, %out: !x86.reg64<rsi>) {
  %c0, %c1, %c2, %c3, %c4, %c5 = xsmm.polynomial_constants
      [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32]
      label_prefix ".Lquintic" : () -> (!v, !v, !v, !v, !v, !v)
  %r = x86.dm.vmovups [%in] : (!x86.reg64<rdi>) -> !v
  %p = xsmm.polynomial %r
      coefficients [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32]
      bindings(%c0, %c1, %c2, %c3, %c4, %c5)
      {datatype = f32, fastmath = #arith.fastmath<reassoc, contract>}
      : (!v, !v, !v, !v, !v, !v, !v) -> !v
  x86.ms.vmovups [%out], %p : (!x86.reg64<rsi>, !v) -> ()
  x86_func.ret
}

// TREE:      %r = x86.dm.vmovups
// TREE-NEXT: %[[R2:.*]] = x86_ext.dss.vmulps %r, %r
// TREE-NEXT: %[[Q0:.*]] = xsmm.horner %r coefficients(%c0, %c1)
// TREE-NEXT: %[[Q1:.*]] = xsmm.horner %r coefficients(%c2, %c3)
// TREE-NEXT: %[[S:.*]] = xsmm.horner %[[R2]] accumulator(%[[Q1]]) coefficients(%[[Q0]])
// TREE-NEXT: %[[Q2:.*]] = xsmm.horner %r coefficients(%c4, %c5)
// TREE-NEXT: %[[R4:.*]] = x86_ext.dss.vmulps %[[R2]], %[[R2]]
// TREE-NEXT: %[[P:.*]] = xsmm.horner %[[R4]] accumulator(%[[Q2]]) coefficients(%[[S]])
// TREE-NEXT: x86.ms.vmovups [%out], %[[P]]
// TREE-NEXT: x86_func.ret

// ALL:      %r = x86.dm.vmovups
// ALL-NEXT: %[[R2:.*]] = x86_ext.dss.vmulps %r, %r
// ALL-NEXT: %[[SEED0:.*]] = x86_ext.ds.vmovdqu64 %r
// ALL-NEXT: %[[Q0:.*]] = x86_ext.rss.vfmadd213ps %[[SEED0]], %c1, %c0
// ALL-NEXT: %[[SEED1:.*]] = x86_ext.ds.vmovdqu64 %r
// ALL-NEXT: %[[Q1:.*]] = x86_ext.rss.vfmadd213ps %[[SEED1]], %c3, %c2
// ALL-NEXT: %[[S:.*]] = x86_ext.rss.vfmadd213ps %[[Q1]], %[[R2]], %[[Q0]]
// ALL-NEXT: %[[SEED2:.*]] = x86_ext.ds.vmovdqu64 %r
// ALL-NEXT: %[[Q2:.*]] = x86_ext.rss.vfmadd213ps %[[SEED2]], %c5, %c4
// ALL-NEXT: %[[R4:.*]] = x86_ext.dss.vmulps %[[R2]], %[[R2]]
// ALL-NEXT: %[[P:.*]] = x86_ext.rss.vfmadd213ps %[[Q2]], %[[R4]], %[[S]]
// ALL-NEXT: x86.ms.vmovups [%out], %[[P]]
// ALL-NEXT: x86_func.ret

// ASM:      vmulps
// ASM-NEXT: vmovdqu64
// ASM-NEXT: vfmadd213ps
// ASM-NEXT: vmovdqu64
// ASM-NEXT: vfmadd213ps
// ASM-NEXT: vfmadd213ps
// ASM-NEXT: vmovdqu64
// ASM-NEXT: vfmadd213ps
// ASM-NEXT: vmulps
// ASM-NEXT: vfmadd213ps
// ASM-NEXT: vmovups [rsi],
// ASM-NEXT: ret
