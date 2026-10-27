// RUN: xdsl-opt %s -p 'xsmm-expand-horner{fully=false}' | filecheck %s --check-prefix=ONE
// RUN: xdsl-opt %s -p xsmm-expand-horner | filecheck %s --check-prefix=ALL
// RUN: xdsl-opt %s -p xsmm-expand-horner -t x86-asm | filecheck %s --check-prefix=ASM

!z0 = !x86.avx512reg<zmm0>
!z1 = !x86.avx512reg<zmm1>
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
