// RUN: xdsl-opt %s -p 'xsmm-expand-horner{fully=false}' | filecheck %s --check-prefix=ONE
// RUN: xdsl-opt %s -p xsmm-expand-horner | filecheck %s --check-prefix=ALL
// RUN: xdsl-opt %s -p xsmm-expand-horner -t x86-asm | filecheck %s --check-prefix=ASM

!z0 = !x86.avx512reg<zmm0>
!z1 = !x86.avx512reg<zmm1>
!z3 = !x86.avx512reg<zmm3>
!z4 = !x86.avx512reg<zmm4>
!z5 = !x86.avx512reg<zmm5>
!z6 = !x86.avx512reg<zmm6>

x86_func.func @carried(%x: !z0, %acc: !z1, %c0: !z6, %c1: !z5, %out: !x86.reg64<rdi>) {
  %before = x86.di.mov 7 : () -> !x86.reg64<r10>
  %p = xsmm.horner %x accumulator(%acc) coefficients(%c0, %c1) {datatype = f32}
      : (!z0, !z1, !z6, !z5) -> !z1
  x86.ms.vmovups [%out], %p : (!x86.reg64<rdi>, !z1) -> ()
  x86_func.ret
}

// ONE:      %before = x86.di.mov 7
// ONE-NEXT: %[[ACC:.*]] = x86_ext.rss.vfmadd213ps %acc, %x, %c1
// ONE-NEXT: %[[P:.*]] = xsmm.horner %x accumulator(%[[ACC]]) coefficients(%c0) {datatype = f32}
// ONE-NEXT: x86.ms.vmovups [%out], %[[P]]
// ONE-NEXT: x86_func.ret

// ALL:      %before = x86.di.mov 7
// ALL-NEXT: %[[ACC:.*]] = x86_ext.rss.vfmadd213ps %acc, %x, %c1
// ALL-NEXT: %[[P:.*]] = x86_ext.rss.vfmadd213ps %[[ACC]], %x, %c0
// ALL-NEXT: x86.ms.vmovups [%out], %[[P]]
// ALL-NEXT: x86_func.ret

// ASM:      mov r10, 7
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm5
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm6
// ASM-NEXT: vmovups [rdi], zmm1
// ASM-NEXT: ret

x86_func.func @cubic(%x: !z0, %c0: !z6, %c1: !z5, %c2: !z4, %c3: !z3, %out: !x86.reg64<rdi>) {
  %before = x86.di.mov 9 : () -> !x86.reg64<r10>
  %p = xsmm.horner %x coefficients(%c0, %c1, %c2, %c3) {datatype = f32}
      : (!z0, !z6, !z5, !z4, !z3) -> !z1
  x86.ms.vmovups [%out], %p : (!x86.reg64<rdi>, !z1) -> ()
  x86_func.ret
}

// ONE:      %before = x86.di.mov 9
// ONE-NEXT: %[[SEED:.*]] = x86_ext.ds.vmovdqu64 %x
// ONE-NEXT: %[[ACC2:.*]] = x86_ext.rss.vfmadd213ps %[[SEED]], %c3, %c2
// ONE-NEXT: %[[P2:.*]] = xsmm.horner %x accumulator(%[[ACC2]]) coefficients(%c0, %c1) {datatype = f32}
// ONE-NEXT: x86.ms.vmovups [%out], %[[P2]]
// ONE-NEXT: x86_func.ret

// ALL:      %before = x86.di.mov 9
// ALL-NEXT: %[[SEED:.*]] = x86_ext.ds.vmovdqu64 %x
// ALL-NEXT: %[[ACC2:.*]] = x86_ext.rss.vfmadd213ps %[[SEED]], %c3, %c2
// ALL-NEXT: %[[ACC1:.*]] = x86_ext.rss.vfmadd213ps %[[ACC2]], %x, %c1
// ALL-NEXT: %[[P2:.*]] = x86_ext.rss.vfmadd213ps %[[ACC1]], %x, %c0
// ALL-NEXT: x86.ms.vmovups [%out], %[[P2]]
// ALL-NEXT: x86_func.ret

// ASM:      mov r10, 9
// ASM-NEXT: vmovdqu64 zmm1, zmm0
// ASM-NEXT: vfmadd213ps zmm1, zmm3, zmm4
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm5
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm6
// ASM-NEXT: vmovups [rdi], zmm1
// ASM-NEXT: ret
