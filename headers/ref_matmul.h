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

#ifndef CTYPE
#define CTYPE DTYPE
#endif

// The type C is accumulated in: C's own, unless C is narrower than the sums,
// as a BF16 C is, whose kernels sum in F32 and round once.
#ifndef ACCTYPE
#define ACCTYPE CTYPE
#endif

// C += AB, where C: M * N, A: M * K, B: K * N, all column-major.  Products are
// formed in ACCTYPE, the way a BF16 kernel widens before it accumulates, and
// each element of C is rounded back to CTYPE once, after its last product.
void ref_matmul_colmaj(DTYPE *A, DTYPE *B, CTYPE *C, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      ACCTYPE acc = (ACCTYPE)C[n * M + m];
      for (int k = 0; k < K; k++) {
        acc += (ACCTYPE)A[k * M + m] * (ACCTYPE)B[n * K + k];
      }
      C[n * M + m] = (CTYPE)acc;
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
