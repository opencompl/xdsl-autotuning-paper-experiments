// RUN: xdsl-opt %s -p xsmm-expand-polynomial-constants | filecheck %s
// RUN: xdsl-opt %s -p xsmm-expand-horner,xsmm-expand-polynomial-constants -t x86-asm | filecheck %s --check-prefix=ASM

!z0 = !x86.avx512reg<zmm0>
!z1 = !x86.avx512reg<zmm1>
!z3 = !x86.avx512reg<zmm3>
!z5 = !x86.avx512reg<zmm5>
!z6 = !x86.avx512reg<zmm6>

x86_func.func @constants(%x: !z0, %out: !x86.reg64<rdi>) {
  %before = x86.di.mov 7 : () -> !x86.reg64<r10>
  %c0, %c1, %c2 = xsmm.polynomial_constants [1.5 : f32, -0.5 : f32, 0.0 : f32]
      label_prefix ".Lconstants_c" : () -> (!z3, !z6, !z5)
  %between = x86.di.mov 11 : () -> !x86.reg64<r11>
  %p = xsmm.horner %x coefficients(%c0, %c1, %c2) {datatype = f32}
      : (!z0, !z3, !z6, !z5) -> !z1
  x86.ms.vmovups [%out], %p : (!x86.reg64<rdi>, !z1) -> ()
  x86_func.ret
}

// CHECK:      %before = x86.di.mov 7
// CHECK-NEXT: x86_ext.inline_constant ".Lconstants_c0", 1069547520 : i64
// CHECK-NEXT: %[[C0:.*]] = x86_ext.dm.vmovups_rip ".Lconstants_c0" : !x86.avx512reg<zmm3>
// CHECK-NEXT: x86_ext.inline_constant ".Lconstants_c1", 3204448256 : i64
// CHECK-NEXT: %[[C1:.*]] = x86_ext.dm.vmovups_rip ".Lconstants_c1" : !x86.avx512reg<zmm6>
// CHECK-NEXT: x86_ext.inline_constant ".Lconstants_c2", 0 : i64
// CHECK-NEXT: %[[C2:.*]] = x86_ext.dm.vmovups_rip ".Lconstants_c2" : !x86.avx512reg<zmm5>
// CHECK-NEXT: %between = x86.di.mov 11
// CHECK-NEXT: %p = xsmm.horner %x coefficients(%[[C0]], %[[C1]], %[[C2]]) {datatype = f32}

// ASM:      mov r10, 7
// ASM-NEXT: jmp .Lconstants_c0_end
// ASM-NEXT: .Lconstants_c0:
// ASM-NEXT: .long 0x3fc00000,
// ASM-NEXT: .Lconstants_c0_end:
// ASM-NEXT: vmovups zmm3, [rip + .Lconstants_c0]
// ASM-NEXT: jmp .Lconstants_c1_end
// ASM-NEXT: .Lconstants_c1:
// ASM-NEXT: .long 0xbf000000,
// ASM-NEXT: .Lconstants_c1_end:
// ASM-NEXT: vmovups zmm6, [rip + .Lconstants_c1]
// ASM-NEXT: jmp .Lconstants_c2_end
// ASM-NEXT: .Lconstants_c2:
// ASM-NEXT: .long 0x00000000,
// ASM-NEXT: .Lconstants_c2_end:
// ASM-NEXT: vmovups zmm5, [rip + .Lconstants_c2]
// ASM-NEXT: mov r11, 11
// ASM-NEXT: vmovdqu64 zmm1, zmm0
// ASM-NEXT: vfmadd213ps zmm1, zmm5, zmm6
// ASM-NEXT: vfmadd213ps zmm1, zmm0, zmm3
// ASM-NEXT: vmovups [rdi], zmm1
// ASM-NEXT: ret
