// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=float -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=float -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=double -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=double -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s

#include "../../headers/mnk.h"

void matmul_colmaj(DTYPE *A, DTYPE *B, DTYPE *C) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
}

// CHECK: Test Passed: The results are equal!
