// RUN: xdsl-opt %s | xdsl-opt | filecheck %s --check-prefix=ROUNDTRIP
// RUN: xdsl-opt %s -p xsmm-expand-polynomial | filecheck %s --check-prefix=HORNER
// RUN: xdsl-opt %s -p xsmm-expand-polynomial,xsmm-expand-horner,xsmm-expand-polynomial-constants -t x86-asm | filecheck %s --check-prefix=ASM

!v = !x86.avx512reg<zmm0>
!p = !x86.avx512reg<zmm1>
!c0 = !x86.avx512reg<zmm2>
!c1 = !x86.avx512reg<zmm3>
!c2 = !x86.avx512reg<zmm4>
!c3 = !x86.avx512reg<zmm5>
!c4 = !x86.avx512reg<zmm6>
!c5 = !x86.avx512reg<zmm7>

x86_func.func @quintic(%r: !v, %out: !x86.reg64<rdi>) {
  %c0, %c1, %c2, %c3, %c4, %c5 = xsmm.polynomial_constants
      [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32]
      label_prefix ".Lquintic" : () -> (!c0, !c1, !c2, !c3, !c4, !c5)
  %before = x86.di.mov 7 : () -> !x86.reg64<r10>
  %p = xsmm.polynomial %r
      coefficients [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32, 6.0 : f32]
      bindings(%c0, %c1, %c2, %c3, %c4, %c5)
      {datatype = f32, fastmath = #arith.fastmath<reassoc, contract>}
      : (!v, !c0, !c1, !c2, !c3, !c4, !c5) -> !p
  x86.ms.vmovups [%out], %p : (!x86.reg64<rdi>, !p) -> ()
  x86_func.ret
}

// ROUNDTRIP: xsmm.polynomial %r coefficients [1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32, 4.000000e+00 : f32, 5.000000e+00 : f32, 6.000000e+00 : f32] bindings(%c0, %c1, %c2, %c3, %c4, %c5)
// HORNER:      %before = x86.di.mov 7
// HORNER-NEXT: %[[SEED:.*]] = x86_ext.ds.vmovdqu64 %c5
// HORNER-NEXT: %[[P:.*]] = xsmm.horner %r accumulator(%[[SEED]]) coefficients(%c0, %c1, %c2, %c3, %c4)
// HORNER-NEXT: x86.ms.vmovups [%out], %[[P]]
// ASM:      mov r10, 7
// ASM-NEXT: vmovdqu64 zmm1, zmm7
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm6
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm5
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm4
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm3
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm2
// ASM-NEXT: vmovups [rdi], zmm1

x86_func.func @cubic(%r: !v, %out: !x86.reg64<rdi>) {
  %c0, %c1, %c2, %c3 = xsmm.polynomial_constants
      [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
      label_prefix ".Lcubic" : () -> (!c0, !c1, !c2, !c3)
  %before = x86.di.mov 9 : () -> !x86.reg64<r10>
  %p = xsmm.polynomial %r
      coefficients [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32]
      bindings(%c0, %c1, %c2, %c3)
      {datatype = f32, fastmath = #arith.fastmath<reassoc, contract>}
      : (!v, !c0, !c1, !c2, !c3) -> !p
  x86.ms.vmovups [%out], %p : (!x86.reg64<rdi>, !p) -> ()
  x86_func.ret
}

// HORNER:      %before = x86.di.mov 9
// HORNER-NEXT: %[[P:.*]] = xsmm.horner %r coefficients(%c0, %c1, %c2, %c3)
// HORNER-NEXT: x86.ms.vmovups [%out], %[[P]]
// ASM:      mov r10, 9
// ASM-NEXT: vmovdqu64 zmm1, zmm0
// ASM-NEXT: vfmadd213ps zmm1, zmm5, zmm4
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm3
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm2
// ASM-NEXT: vmovups [rdi], zmm1
