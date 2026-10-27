// RUN: xdsl-opt %s | xdsl-opt -t x86-asm -o %t.s
// RUN: filecheck %s < %t.s
// RUN: clang -target x86_64-unknown-linux-gnu -c %t.s -o %t.o

// Check operand order, immediates, SAE, predicate destinations, and inline data
// through text roundtrip and the real assembler, without requiring AVX-512.
// CHECK: jmp .Lconstant_end
// CHECK-NEXT: .Lconstant:
// CHECK-NEXT: .long {{(0x3f000000, ){15}0x3f000000}}
// CHECK-NEXT: .Lconstant_end:
// CHECK-NEXT: vmovups zmm0, [rip + .Lconstant]
// CHECK-NEXT: vmovdqu64 zmm1, zmm0
// CHECK-NEXT: vrndscaleps zmm2, zmm1, {sae}, 1
// CHECK-NEXT: vrndscaleps zmm3, zmm1, 0
// CHECK-NEXT: vsubps zmm4, zmm1, zmm2
// CHECK-NEXT: vmulps zmm5, zmm4, zmm3
// CHECK-NEXT: vscalefps zmm6, zmm5, zmm2
// CHECK-NEXT: vminps zmm7, zmm0, zmm6
// CHECK-NEXT: vmaxps zmm8, zmm1, zmm7
// CHECK-NEXT: vfmadd213ps zmm8, zmm4, zmm0
// CHECK-NEXT: vfnmadd231ps zmm8, zmm3, zmm2
// CHECK-NEXT: vcvtps2dq zmm9, zmm8
// CHECK-NEXT: vpaddd zmm10, zmm9, zmm0
// CHECK-NEXT: vpslld zmm11, zmm10, 23
// CHECK-NEXT: vcmpps k2, zmm11, zmm1, 1
// CHECK-NEXT: vblendmps zmm12 {k2}, zmm11, zmm0
// CHECK-NEXT: ret

x86_func.func public @ops() {
  x86_ext.inline_constant ".Lconstant", 1056964608 : i64
  %a = x86_ext.dm.vmovups_rip ".Lconstant" : !x86.avx512reg<zmm0>
  %b = x86_ext.ds.vmovdqu64 %a : (!x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm1>
  %c = x86_ext.dsi.vrndscaleps %b, 1 {sae} : (!x86.avx512reg<zmm1>) -> !x86.avx512reg<zmm2>
  %d = x86_ext.dsi.vrndscaleps %b, 0 : (!x86.avx512reg<zmm1>) -> !x86.avx512reg<zmm3>
  %e = x86_ext.dss.vsubps %b, %c : (!x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm4>
  %f = x86_ext.dss.vmulps %e, %d : (!x86.avx512reg<zmm4>, !x86.avx512reg<zmm3>) -> !x86.avx512reg<zmm5>
  %g = x86_ext.dss.vscalefps %f, %c : (!x86.avx512reg<zmm5>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm6>
  %h = x86_ext.dss.vminps %a, %g : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm6>) -> !x86.avx512reg<zmm7>
  %i = x86_ext.dss.vmaxps %b, %h : (!x86.avx512reg<zmm1>, !x86.avx512reg<zmm7>) -> !x86.avx512reg<zmm8>
  %j = x86_ext.rss.vfmadd213ps %i, %e, %a : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm8>
  %k = x86_ext.rss.vfnmadd231ps %j, %d, %c : (!x86.avx512reg<zmm8>, !x86.avx512reg<zmm3>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm8>
  %l = x86_ext.ds.vcvtps2dq %k : (!x86.avx512reg<zmm8>) -> !x86.avx512reg<zmm9>
  %m = x86_ext.dss.vpaddd %l, %a : (!x86.avx512reg<zmm9>, !x86.avx512reg<zmm0>) -> !x86.avx512reg<zmm10>
  %n = x86_ext.dsi.vpslld %m, 23 : (!x86.avx512reg<zmm10>) -> !x86.avx512reg<zmm11>
  %mask = x86_ext.dssi.vcmpps %n, %b, 1 : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm1>) -> !x86.avx512maskreg<k2>
  %o = x86_ext.dssk.vblendmps %n, %a, %mask : (!x86.avx512reg<zmm11>, !x86.avx512reg<zmm0>, !x86.avx512maskreg<k2>) -> !x86.avx512reg<zmm12>
  x86_func.ret
}
