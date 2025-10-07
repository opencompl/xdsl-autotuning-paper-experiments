#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#if defined(MKL_DTYPE_IS_FLOAT)
  #define GEMM  cblas_sgemm
  #define ALPHA 1.0f
  #define BETA  1.0f
  #define MKL_DTYPE float
#elif defined(MKL_DTYPE_IS_DOUBLE)
  #define GEMM  cblas_dgemm
  #define ALPHA 1.0
  #define BETA  1.0
  #define MKL_DTYPE double
#else
  #error "You must define MKL_DTYPE_IS_FLOAT or MKL_DTYPE_IS_DOUBLE"
#endif

void matmul(MKL_DTYPE *A, MKL_DTYPE *B, MKL_DTYPE *C) {
  GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       MKL_M, MKL_N, MKL_K,
       ALPHA,
       A, MKL_K,
       B, MKL_N,
       BETA,
       C, MKL_N);
}
