// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -o %t \
// RUN: arm_4x4_matmul_asm_colmajor/main.c %s && %t | filecheck %s
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -o %t \
// RUN: arm_4x4_matmul_asm_colmajor/main.c %s && %t | filecheck %s

#include "../../headers/mnk.h"

void matmul_colmaj(float *C, float *A, float *B) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
}

// CHECK: Test Passed: The results are equal!
