// RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=float -o %t \
// RUN: kernels/matmul_rowmaj/test.c %s || %t | filecheck %s
// RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=double -o %t \
// RUN: kernels/matmul_rowmaj/test.c %s || %t | filecheck %s

#include "../../headers/mnk.h"

void matmul(DTYPE *A, DTYPE *B, DTYPE *C) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
