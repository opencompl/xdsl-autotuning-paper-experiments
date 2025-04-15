// RUN: clang -o %t arm_4x4_matmul_asm_colmajor/main.c %s || %t | filecheck %s

void matmul_colmaj(float *C, float *A, float *B) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
