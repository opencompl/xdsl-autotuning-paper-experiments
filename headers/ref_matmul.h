// C += AB, where C: M * N, A: M * K, B: K * N, all row-major
void ref_matmul(DTYPE *A, DTYPE *B, DTYPE *C, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

// C += AB, where C: M * N, A: M * K, B: K * N, all column-major
void ref_matmul_colmaj(DTYPE *A, DTYPE *B, DTYPE *C, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
}

void transpose(DTYPE *out, DTYPE *in, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      out[c * rows + r] = in[r * cols + c];
    }
  }
}
