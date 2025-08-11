#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../headers/gendata.h"
#include "../../headers/isclose.h"
#include "../../headers/mnk.h"
#include "../../headers/print_matrix.h"
#include "../../headers/ref_matmul.h"

extern void matmul(DTYPE result[M * N], DTYPE A[M * K], DTYPE B[K * N]);

int main() {
  set_random_seed(42);

  DTYPE *A = aligned_alloc(64, M * K * sizeof(DTYPE));
  DTYPE *B = aligned_alloc(64, K * N * sizeof(DTYPE));
  DTYPE *C = aligned_alloc(64, M * N * sizeof(DTYPE));
  DTYPE *A_asm = aligned_alloc(64, M * K * sizeof(DTYPE));
  DTYPE *B_asm = aligned_alloc(64, K * N * sizeof(DTYPE));
  DTYPE *C_asm = aligned_alloc(64, M * N * sizeof(DTYPE));

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  fill_random_data(C, M * N);
  printf("A\n");
  print_matrix(A, M, K);
  printf("B\n");
  print_matrix(B, K, N);
  printf("C\n");
  print_matrix(C, M, N);

  memcpy(A_asm, A, M * K * sizeof(DTYPE));
  memcpy(B_asm, B, K * N * sizeof(DTYPE));
  memcpy(C_asm, C, M * N * sizeof(DTYPE));

  ref_matmul(A, B, C, M, N, K);
  matmul(A_asm, B_asm, C_asm);

  printf("C out\n");
  print_matrix(C, M, N);
  printf("C_asm out\n");
  print_matrix(C_asm, M, N);

  if (isclose(C, C_asm, M * N)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }

  free(A);
  free(B);
  free(C);
  free(A_asm);
  free(B_asm);
  free(C_asm);
}
