# RUN: python3 kernels/matmul_rowmaj/generate_avx512_outer_asm.py --M 2 --N 3 --K 64 --output %t.S
# RUN: filecheck %s --input-file %t.S
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.S -o %t.o

# CHECK-LABEL: matmul:
# CHECK-NOT: push
# CHECK-NOT: rsp
# CHECK: mov eax, 7
# CHECK-NEXT: kmovb k1, eax
# CHECK: vmovupd zmm16 {k1}{z}, [rdx]
# CHECK: vmovupd zmm24 {k1}{z}, [rdx+24]
# CHECK: vmovupd zmm0 {k1}{z}, [rsi]
# CHECK: vfmadd231pd zmm16, zmm0, [rdi]{1to8}
# CHECK: vfmadd231pd zmm24, zmm0, [rdi+512]{1to8}
# CHECK: vmovupd zmm0 {k1}{z}, [rsi+24]
# CHECK: vfmadd231pd zmm17, zmm0, [rdi+8]{1to8}
# CHECK: vaddpd zmm16, zmm16, zmm17
# CHECK: vmovupd [rdx] {k1}, zmm16
# CHECK: vmovupd [rdx+24] {k1}, zmm24
# CHECK-NEXT: ret
