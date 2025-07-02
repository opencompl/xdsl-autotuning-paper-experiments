// RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -o %t \
// RUN: kernels/matmul_colmaj/test.c %s || %t | filecheck %s

void matmul_colmaj(float *C, float *A, float *B) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
