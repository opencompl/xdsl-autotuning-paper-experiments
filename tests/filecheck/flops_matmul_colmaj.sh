# RUN: ./kernels/matmul_colmaj/flops.sh 1 2 3 | filecheck %s

# CHECK: 12
