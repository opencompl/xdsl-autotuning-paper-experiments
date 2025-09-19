#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../headers/gendata.h"
#include "../../headers/isclose.h"
#include "../../headers/mnk.h"
#include "../../headers/print_matrix.h"
#include "../../headers/ref_matmul.h"

extern void matmul_colmaj(DTYPE A[M * K], DTYPE B[K * N], DTYPE result[M * N]);

int main() {
  set_random_seed(42);

  DTYPE *A, *B, *C, *A_colmaj, *B_colmaj, *C_colmaj;

  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(DTYPE));
  posix_memalign((void **)&A_colmaj, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B_colmaj, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C_colmaj, 64, M * N * sizeof(DTYPE));

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  fill_random_data(C, M * N);

  printf("A\n");
  print_matrix(A, M, K);
  printf("B\n");
  print_matrix(B, K, N);
  printf("C\n");
  print_matrix(C, M, N);

  transpose(A_colmaj, A, M, K);
  transpose(B_colmaj, B, K, N);
  transpose(C_colmaj, C, M, N);

  printf("A_colmaj\n");
  print_matrix_colmaj(A_colmaj, M, K);
  printf("B_colmaj\n");
  print_matrix_colmaj(B_colmaj, K, N);
  printf("C_colmaj\n");
  print_matrix_colmaj(C_colmaj, M, N);

  ref_matmul(A, B, C, M, N, K);
  matmul_colmaj(A_colmaj, B_colmaj, C_colmaj);

  printf("C out\n");
  print_matrix(C, M, N);

  DTYPE res[M * N];

  printf("C_asm out\n");
  transpose(res, C_colmaj, N, M);

  print_matrix(res, M, N);

  if (isclose(C, res, M * N)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }
  free(A);
  free(B);
  free(C);
  free(A_colmaj);
  free(B_colmaj);
  free(C_colmaj);
}
