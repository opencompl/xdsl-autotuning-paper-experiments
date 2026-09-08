#include "../../headers/mnk.h"

// C += A * B, with A: M * K, B: K * N and C: M * N all column-major, so the
// leading dimensions are M, K and M and the M index is the contiguous one.
void matmul(DTYPE *A, DTYPE *B, DTYPE *C) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
}
