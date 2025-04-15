// RUN: clang -o %t arm_4x4_matmul_asm_colmajor/main.c %s && %t | filecheck %s

#define M 4
#define N 4
#define K 4

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
