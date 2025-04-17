// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -o %t \
// RUN: arm_4x4_matmul_asm_colmajor/main.c %s || %t | filecheck %s

void matmul_colmaj(float *C, float *A, float *B) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
