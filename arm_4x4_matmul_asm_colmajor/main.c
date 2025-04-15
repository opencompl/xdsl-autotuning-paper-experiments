#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../headers/gendata.h"
#include "../headers/isclose.h"
#include "../headers/mnk.h"
#include "../headers/print_matrix.h"
#include "../headers/ref_matmul.h"

extern void matmul_colmaj(float result[M * N], float A[M * K], float B[K * N]);

int main() {
  set_random_seed(42);

  float A[M * K], B[K * N], C[M * N], A_colmaj[M * K], B_colmaj[K * N],
      C_colmaj[M * N];

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
  print_matrix(A_colmaj, M, K);
  printf("B_colmaj\n");
  print_matrix(B_colmaj, K, N);
  printf("C_colmaj\n");
  print_matrix(C_colmaj, M, N);

  ref_matmul(C, A, B, M, N, K);
  matmul_colmaj(C_colmaj, A_colmaj, B_colmaj);

  printf("C out\n");
  print_matrix(C, M, N);

  float res[M * N];

  printf("C_asm out\n");
  transpose(res, C_colmaj, M, N);

  print_matrix(res, M, N);

  if (isclose(C, res, M * N)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }
}
