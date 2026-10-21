// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=float -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=float -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=double -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=double -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && %t | filecheck %s
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=float -DMUTATE_INPUT=A -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && not %t | filecheck %s --check-prefix=MUTATED
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=double -DMUTATE_INPUT=A -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && not %t | filecheck %s --check-prefix=MUTATED
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=float -DMUTATE_INPUT=B -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && not %t | filecheck %s --check-prefix=MUTATED
// RUN: clang -DCROWS=5 -DCCOLS=6 -DINNER=7 -DDTYPE=double -DMUTATE_INPUT=B -o %t \
// RUN: kernels/matmul_colmaj/test.c %s && not %t | filecheck %s --check-prefix=MUTATED

#include "../../headers/mnk.h"
#include <assert.h>

static int initialized;
void matmul_init(void) {
  assert(!initialized);
  initialized = 1;
}

void matmul(DTYPE *A, DTYPE *B, DTYPE *C) {
  assert(initialized);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
#ifdef MUTATE_INPUT
  // The output is correct; only preservation of the input can catch this.
  MUTATE_INPUT[0] += 1;
#endif
}

// CHECK: Test Passed: The results are equal!
// MUTATED: Test Failed: The kernel modified an input matrix.
