// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4=4 -DINNER=4 -o %t \
// RUN: arm_4x4_matmul_asm_rowmajor/main.c %s || %t | filecheck %s

#include "../../headers/mnk.h"

void matmul(float *C, float *A, float *B) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
