#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void matmul(MKL_DTYPE *A, const MKL_DTYPE *B, MKL_DTYPE *C) {
  if (sizeof(MKL_DTYPE) == 4) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                MKL_M, MKL_N, MKL_K,
                1.0f /* alpha */,
                A, MKL_K,
                B, MKL_N,
                0.0f /* beta */,
                C, MKL_N);
  } else {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                MKL_M, MKL_N, MKL_K,
                1.0f /* alpha */,
                A, MKL_K,
                B, MKL_N,
                0.0f /* beta */,
                C, MKL_N);
  }
}
