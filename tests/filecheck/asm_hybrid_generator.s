# RUN: python3 kernels/matmul_rowmaj/generate_avx512_hybrid_asm.py --M 4 --N 1 --K 64 --fallback /dev/null --output %t.n1.S
# RUN: filecheck %s --input-file %t.n1.S --check-prefix N1
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_hybrid_asm.py --M 4 --N 2 --K 64 --fallback /dev/null --output %t.outer.S
# RUN: filecheck %s --input-file %t.outer.S --check-prefix OUTER
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_hybrid_asm.py --M 8 --N 2 --K 64 --fallback /dev/null --output %t.gather.S
# RUN: filecheck %s --input-file %t.gather.S --check-prefix GATHER
# RUN: python3 kernels/matmul_rowmaj/generate_avx512_hybrid_asm.py --M 8 --N 5 --K 64 --fallback %s --output %t.fallback.S
# RUN: cmp %s %t.fallback.S

# N1-LABEL: matmul:
# N1: vmovupd zmm0, [rsi]
# N1-NOT: vgather

# OUTER-LABEL: matmul:
# OUTER: vfmadd231pd zmm16, zmm0, [rdi]{1to8}
# OUTER-NOT: vgather

# GATHER-LABEL: matmul:
# GATHER: vgatherqpd zmm1 {k1}, [rsi + zmm0*8]
