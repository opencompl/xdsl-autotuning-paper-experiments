// RUN: clang -o %t arm_4x4_matmul_asm_rowmajor/main.c %s || %t | filecheck %s

void matmul(float *C, float *A, float *B) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
