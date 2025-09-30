// RUN: xdsl-opt -p vectorize-libxsmm,scf-for-loop-unroll,convert-vector-to-ptr,convert-memref-to-ptr{lower_func=true},convert-ptr-type-offsets,canonicalize,convert-func-to-x86-func,convert-vector-to-x86{arch=avx512},convert-ptr-to-x86{arch=avx512},convert-arith-to-x86,reconcile-unrealized-casts,canonicalize,x86-infer-broadcast,dce,x86-allocate-registers,canonicalize -t x86-asm %s | filecheck %s

func.func @matmul(
  %A: memref<3x42xf64>,
  %B: memref<42x16xf64>,
  %C: memref<3x16xf64>
) {
  linalg.matmul ins(%A, %B: memref<3x42xf64>, memref<42x16xf64>) outs(%C: memref<3x16xf64>)
  return
}

// CHECK:       .intel_syntax noprefix
// CHECK-NEXT:  .text
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      mov rbx, rdi
// CHECK-NEXT:      mov rcx, rsi
// CHECK-NEXT:      mov rax, rdx
// CHECK-NEXT:      vmovupd ymm11, [rax]
// CHECK-NEXT:      vmovupd ymm10, [rax+128]
// CHECK-NEXT:      vmovupd ymm9, [rax+256]
// CHECK-NEXT:      vmovupd ymm8, [rax+32]
// CHECK-NEXT:      vmovupd ymm7, [rax+160]
// CHECK-NEXT:      vmovupd ymm6, [rax+288]
// CHECK-NEXT:      vmovupd ymm5, [rax+64]
// CHECK-NEXT:      vmovupd ymm4, [rax+192]
// CHECK-NEXT:      vmovupd ymm3, [rax+320]
// CHECK-NEXT:      vmovupd ymm2, [rax+96]
// CHECK-NEXT:      vmovupd ymm1, [rax+224]
// CHECK-NEXT:      vmovupd ymm0, [rax+352]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+336]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+672]
// CHECK-NEXT:      vmovupd ymm14, [rcx]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+32]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+64]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+96]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+8]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+344]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+680]
// CHECK-NEXT:      vmovupd ymm13, [rcx+128]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+160]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+192]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+224]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+16]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+352]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+688]
// CHECK-NEXT:      vmovupd ymm15, [rcx+256]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+288]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+320]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+352]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+24]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+360]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+696]
// CHECK-NEXT:      vmovupd ymm12, [rcx+384]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+416]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+448]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+480]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+32]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+368]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+704]
// CHECK-NEXT:      vmovupd ymm14, [rcx+512]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+544]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+576]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+608]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+40]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+376]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+712]
// CHECK-NEXT:      vmovupd ymm13, [rcx+640]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+672]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+704]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+736]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+48]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+384]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+720]
// CHECK-NEXT:      vmovupd ymm15, [rcx+768]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+800]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+832]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+864]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+56]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+392]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+728]
// CHECK-NEXT:      vmovupd ymm12, [rcx+896]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+928]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+960]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+992]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+64]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+400]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+736]
// CHECK-NEXT:      vmovupd ymm14, [rcx+1024]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+1056]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+1088]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+1120]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+72]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+408]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+744]
// CHECK-NEXT:      vmovupd ymm13, [rcx+1152]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+1184]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+1216]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+1248]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+80]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+416]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+752]
// CHECK-NEXT:      vmovupd ymm15, [rcx+1280]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+1312]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+1344]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+1376]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+88]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+424]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+760]
// CHECK-NEXT:      vmovupd ymm12, [rcx+1408]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+1440]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+1472]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+1504]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+96]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+432]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+768]
// CHECK-NEXT:      vmovupd ymm14, [rcx+1536]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+1568]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+1600]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+1632]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+104]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+440]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+776]
// CHECK-NEXT:      vmovupd ymm13, [rcx+1664]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+1696]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+1728]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+1760]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+112]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+448]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+784]
// CHECK-NEXT:      vmovupd ymm15, [rcx+1792]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+1824]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+1856]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+1888]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+120]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+456]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+792]
// CHECK-NEXT:      vmovupd ymm12, [rcx+1920]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+1952]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+1984]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+2016]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+128]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+464]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+800]
// CHECK-NEXT:      vmovupd ymm14, [rcx+2048]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+2080]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+2112]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+2144]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+136]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+472]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+808]
// CHECK-NEXT:      vmovupd ymm13, [rcx+2176]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+2208]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+2240]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+2272]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+144]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+480]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+816]
// CHECK-NEXT:      vmovupd ymm15, [rcx+2304]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+2336]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+2368]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+2400]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+152]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+488]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+824]
// CHECK-NEXT:      vmovupd ymm12, [rcx+2432]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+2464]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+2496]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+2528]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+160]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+496]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+832]
// CHECK-NEXT:      vmovupd ymm14, [rcx+2560]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+2592]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+2624]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+2656]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+168]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+504]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+840]
// CHECK-NEXT:      vmovupd ymm13, [rcx+2688]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+2720]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+2752]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+2784]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+176]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+512]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+848]
// CHECK-NEXT:      vmovupd ymm15, [rcx+2816]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+2848]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+2880]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+2912]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+184]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+520]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+856]
// CHECK-NEXT:      vmovupd ymm12, [rcx+2944]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+2976]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+3008]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+3040]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+192]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+528]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+864]
// CHECK-NEXT:      vmovupd ymm14, [rcx+3072]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+3104]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+3136]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+3168]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+200]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+536]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+872]
// CHECK-NEXT:      vmovupd ymm13, [rcx+3200]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+3232]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+3264]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+3296]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+208]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+544]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+880]
// CHECK-NEXT:      vmovupd ymm15, [rcx+3328]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+3360]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+3392]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+3424]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+216]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+552]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+888]
// CHECK-NEXT:      vmovupd ymm12, [rcx+3456]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+3488]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+3520]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+3552]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+224]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+560]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+896]
// CHECK-NEXT:      vmovupd ymm14, [rcx+3584]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+3616]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+3648]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+3680]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+232]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+568]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+904]
// CHECK-NEXT:      vmovupd ymm13, [rcx+3712]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+3744]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+3776]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+3808]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+240]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+576]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+912]
// CHECK-NEXT:      vmovupd ymm15, [rcx+3840]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+3872]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+3904]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+3936]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+248]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+584]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+920]
// CHECK-NEXT:      vmovupd ymm12, [rcx+3968]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+4000]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+4032]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+4064]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+256]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+592]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+928]
// CHECK-NEXT:      vmovupd ymm14, [rcx+4096]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+4128]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+4160]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+4192]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+264]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+600]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+936]
// CHECK-NEXT:      vmovupd ymm13, [rcx+4224]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+4256]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+4288]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+4320]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+272]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+608]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+944]
// CHECK-NEXT:      vmovupd ymm15, [rcx+4352]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+4384]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+4416]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+4448]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+280]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+616]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+952]
// CHECK-NEXT:      vmovupd ymm12, [rcx+4480]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+4512]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+4544]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+4576]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+288]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+624]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+960]
// CHECK-NEXT:      vmovupd ymm14, [rcx+4608]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+4640]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+4672]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+4704]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+296]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+632]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+968]
// CHECK-NEXT:      vmovupd ymm13, [rcx+4736]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+4768]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+4800]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+4832]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+304]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+640]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+976]
// CHECK-NEXT:      vmovupd ymm15, [rcx+4864]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm10, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm9, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+4896]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm7, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm6, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+4928]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm4, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm3, ymm14, ymm15
// CHECK-NEXT:      vmovupd ymm15, [rcx+4960]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm12, ymm15
// CHECK-NEXT:      vfmadd231pd ymm1, ymm13, ymm15
// CHECK-NEXT:      vfmadd231pd ymm0, ymm14, ymm15
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+312]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+648]
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+984]
// CHECK-NEXT:      vmovupd ymm12, [rcx+4992]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm10, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm9, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+5024]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm7, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm6, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+5056]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm4, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm3, ymm13, ymm12
// CHECK-NEXT:      vmovupd ymm12, [rcx+5088]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm14, ymm12
// CHECK-NEXT:      vfmadd231pd ymm1, ymm15, ymm12
// CHECK-NEXT:      vfmadd231pd ymm0, ymm13, ymm12
// CHECK-NEXT:      vbroadcastsd ymm13, [rbx+320]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+656]
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+992]
// CHECK-NEXT:      vmovupd ymm14, [rcx+5120]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm10, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm9, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+5152]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm7, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm6, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+5184]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm4, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm3, ymm15, ymm14
// CHECK-NEXT:      vmovupd ymm14, [rcx+5216]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm13, ymm14
// CHECK-NEXT:      vfmadd231pd ymm1, ymm12, ymm14
// CHECK-NEXT:      vfmadd231pd ymm0, ymm15, ymm14
// CHECK-NEXT:      vbroadcastsd ymm15, [rbx+328]
// CHECK-NEXT:      vbroadcastsd ymm14, [rbx+664]
// CHECK-NEXT:      vbroadcastsd ymm12, [rbx+1000]
// CHECK-NEXT:      vmovupd ymm13, [rcx+5248]
// CHECK-NEXT:      vfmadd231pd ymm11, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm10, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm9, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+5280]
// CHECK-NEXT:      vfmadd231pd ymm8, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm7, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm6, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+5312]
// CHECK-NEXT:      vfmadd231pd ymm5, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm4, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm3, ymm12, ymm13
// CHECK-NEXT:      vmovupd ymm13, [rcx+5344]
// CHECK-NEXT:      vfmadd231pd ymm2, ymm15, ymm13
// CHECK-NEXT:      vfmadd231pd ymm1, ymm14, ymm13
// CHECK-NEXT:      vfmadd231pd ymm0, ymm12, ymm13
// CHECK-NEXT:      vmovapd [rax], ymm11
// CHECK-NEXT:      vmovapd [rax+128], ymm10
// CHECK-NEXT:      vmovapd [rax+256], ymm9
// CHECK-NEXT:      vmovapd [rax+32], ymm8
// CHECK-NEXT:      vmovapd [rax+160], ymm7
// CHECK-NEXT:      vmovapd [rax+288], ymm6
// CHECK-NEXT:      vmovapd [rax+64], ymm5
// CHECK-NEXT:      vmovapd [rax+192], ymm4
// CHECK-NEXT:      vmovapd [rax+320], ymm3
// CHECK-NEXT:      vmovapd [rax+96], ymm2
// CHECK-NEXT:      vmovapd [rax+224], ymm1
// CHECK-NEXT:      vmovapd [rax+352], ymm0
// CHECK-NEXT:      ret
