#include "../../headers/mnk.h"

void matmul(DTYPE *C, DTYPE *A, DTYPE *B) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}
