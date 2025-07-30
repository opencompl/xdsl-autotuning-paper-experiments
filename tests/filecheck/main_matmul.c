// RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=float -o %t \
// RUN: kernels/matmul_rowmaj/test.c %s && %t | filecheck %s
// RUN: clang-20 -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=float -o %t \
// RUN: kernels/matmul_rowmaj/test.c %s && %t | filecheck %s
// RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=double -o %t \
// RUN: kernels/matmul_rowmaj/test.c %s && %t | filecheck %s
// RUN: clang-20 -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=double -o %t \
// RUN: kernels/matmul_rowmaj/test.c %s && %t | filecheck %s

#include "../../headers/mnk.h"

void matmul(DTYPE *C, DTYPE *A, DTYPE *B) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

// CHECK: Test Passed: The results are equal!
