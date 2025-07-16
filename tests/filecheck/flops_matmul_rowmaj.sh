# RUN: ./kernels/matmul_rowmaj/flops.sh 1 2 3 | filecheck %s

# CHECK: 12
