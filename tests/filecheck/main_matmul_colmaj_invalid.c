// RUN: clang -DCROWS=4 -DCCOLS=4 -DINNER=4 -DDTYPE=float -o %t \
// RUN: kernels/matmul_colmaj/test.c %s || %t | filecheck %s

void matmul_colmaj(DTYPE *A, DTYPE *B, DTYPE *C) {
  // Do nothing
}

// CHECK: Test Failed: The results do not match.
