# RUN: python3 kernels/matmul_rowmaj/generate_avx512_partitioned_asm.py --M 25 --N 2 --K 64 --strategy equalized --output %t.m25-equalized.S
# RUN: filecheck %s --input-file %t.m25-equalized.S --check-prefix M25-EQUALIZED
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.m25-equalized.S -o %t.m25-equalized.o
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_partitioned_asm.py --M 25 --N 2 --K 64 --strategy cost --output %t.m25-cost.S
# RUN: filecheck %s --input-file %t.m25-cost.S --check-prefix M25-COST
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.m25-cost.S -o %t.m25-cost.o
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_partitioned_asm.py --M 26 --N 2 --K 64 --strategy equalized --output %t.m26-equalized.S
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.m26-equalized.S -o %t.m26-equalized.o
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_partitioned_asm.py --M 26 --N 2 --K 64 --strategy cost --output %t.m26-cost.S
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.m26-cost.S -o %t.m26-cost.o
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_partitioned_asm.py --M 32 --N 1 --K 64 --strategy equalized --output %t.m32-equalized.S
# RUN: filecheck %s --input-file %t.m32-equalized.S --check-prefix M32-EQUALIZED
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.m32-equalized.S -o %t.m32-equalized.o
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_partitioned_asm.py --M 32 --N 1 --K 64 --strategy cost --output %t.m32-cost.S
# RUN: filecheck %s --input-file %t.m32-cost.S --check-prefix M32-COST
# RUN: clang -c -target x86_64-unknown-linux-gnu -mavx512f %t.m32-cost.S -o %t.m32-cost.o

# The equalized M=25 schedule consists of three gather tiles: 9 + 8 + 8.
# M25-EQUALIZED-LABEL: matmul:
# M25-EQUALIZED-COUNT-3: vmovdqu64 zmm0, [rip + .Lindices_n2]
# M25-EQUALIZED-NOT: {1to8}
# M25-EQUALIZED: ret

# The cost-aware M=25 schedule uses two full 12-row gather tiles and a
# one-row outer-product tail.  The offsets show that each tile addresses its
# global A and C rows rather than restarting at row zero.
# M25-COST-LABEL: matmul:
# M25-COST: vmovdqu64 zmm0, [rip + .Lindices_n2]
# M25-COST: vmovdqu64 zmm0, [rip + .Lindices_n2]
# M25-COST: vfmadd231pd zmm8, zmm1, [rdi+6144]
# M25-COST: vmovupd zmm16 {k1}{z}, [rdx+384]
# M25-COST: vfmadd231pd zmm16, zmm0, [rdi+12288]{1to8}
# M25-COST: vmovupd [rdx+384] {k1}, zmm16
# M25-COST-NEXT: ret

# M32-EQUALIZED-LABEL: matmul:
# M32-EQUALIZED-COUNT-2: vmovupd zmm0, [rsi]
# M32-EQUALIZED: vfmadd231pd zmm16, zmm0, [rdi+8192]
# M32-EQUALIZED: vmovsd [rdx+248], xmm31
# M32-EQUALIZED-NEXT: ret

# M32-COST-LABEL: matmul:
# M32-COST-COUNT-4: vmovupd zmm0, [rsi]
# M32-COST: vfmadd231pd zmm16, zmm0, [rdi+12288]
# M32-COST: vmovsd [rdx+248], xmm23
# M32-COST-NEXT: ret
