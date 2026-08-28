# RUN: python3 kernels/matmul_rowmaj/generate_avx512_kdot_asm.py --M 2 --N 1 --K 64 --output %t.n1.S
# RUN: filecheck %s --input-file %t.n1.S --check-prefix N1
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.n1.S -o %t.n1.o
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_kdot_asm.py --M 2 --N 4 --K 64 --output %t.n4.S
# RUN: filecheck %s --input-file %t.n4.S --check-prefix N4
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.n4.S -o %t.n4.o

# N1-LABEL: matmul:
# N1-NOT: push
# N1-NOT: rsp
# N1: vmovupd zmm0, [rsi]
# N1: vmovupd zmm7, [rsi+448]
# N1: vfmadd231pd zmm16, zmm0, [rdi]
# N1: vfmadd231pd zmm17, zmm7, [rdi+960]
# N1: vextractf64x4 ymm0, zmm16, 1
# N1: vextractf64x4 ymm1, zmm17, 1
# N1: vaddsd xmm16, xmm0, xmm8
# N1: vaddsd xmm17, xmm1, xmm9
# N1: vmovsd [rdx], xmm16
# N1: vmovsd [rdx+8], xmm17
# N1-NEXT: ret

# N4-LABEL: matmul:
# N4-NOT: push
# N4-NOT: rsp
# N4: vmovdqu64 zmm0, [rip + .Lkdot_indices]
# N4: kmovb k1, eax
# N4: vgatherqpd zmm1 {k1}, [rsi + zmm0*8]
# N4: vfmadd231pd zmm8, zmm1, [rdi]
# N4: vgatherqpd zmm1 {k1}, [rsi + zmm0*8+8]
# N4: vfmadd231pd zmm9, zmm1, [rdi]
# N4: vextractf64x4 ymm0, zmm8, 1
# N4: vaddsd xmm8, xmm0, xmm4
# N4: vmovsd [rdx], xmm8
# N4: ret
# N4: .Lkdot_indices:
# N4-NEXT: .quad 0, 4, 8, 12, 16, 20, 24, 28
