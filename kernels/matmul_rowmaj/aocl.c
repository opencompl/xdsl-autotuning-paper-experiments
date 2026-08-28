#include <cblas.h>

#if defined(AOCL_DTYPE_IS_FLOAT)
  #define GEMM cblas_sgemm
  #define ALPHA 1.0f
  #define BETA 1.0f
  #define AOCL_DTYPE float
#elif defined(AOCL_DTYPE_IS_DOUBLE)
  #define GEMM cblas_dgemm
  #define ALPHA 1.0
  #define BETA 1.0
  #define AOCL_DTYPE double
#else
  #error "You must define AOCL_DTYPE_IS_FLOAT or AOCL_DTYPE_IS_DOUBLE"
#endif

void matmul(AOCL_DTYPE *A, AOCL_DTYPE *B, AOCL_DTYPE *C) {
  GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       AOCL_M, AOCL_N, AOCL_K,
       ALPHA,
       A, AOCL_K,
       B, AOCL_N,
       BETA,
       C, AOCL_N);
}
