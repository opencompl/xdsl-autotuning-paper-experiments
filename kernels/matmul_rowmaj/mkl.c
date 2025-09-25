#include <mkl.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>

void matmul(void *C, const void *A, const void *B) {
  if (sizeof(MKL_DTYPE) == 32) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                MKL_M, MKL_N, MKL_K,
                1.0 /* alpha */,
                A, MKL_M,
                B, MKL_K,
                0.0 /* beta */,
                C, MKL_M);
  } else {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                MKL_M, MKL_N, MKL_K,
                1.0 /* alpha */,
                A, MKL_M,
                B, MKL_K,
                0.0 /* beta */,
                C, MKL_M);
  }
}
