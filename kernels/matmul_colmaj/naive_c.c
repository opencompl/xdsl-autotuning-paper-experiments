#include "../../headers/mnk.h"

void matmul_colmaj(DTYPE *A, DTYPE *B, DTYPE *C) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
}
