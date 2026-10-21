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

// Called once before any kernel invocation; per-call work belongs in matmul.
extern void matmul_init(void);
extern void matmul(DTYPE *A, DTYPE *B, DTYPE *C);

int main() {
  matmul_init();
  set_random_seed(42);

  DTYPE *A, *B, *C, *A_asm, *B_asm, *C_asm;

  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(DTYPE));
  posix_memalign((void **)&A_asm, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B_asm, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C_asm, 64, M * N * sizeof(DTYPE));

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  fill_random_data(C, M * N);
  printf("A\n");
  print_matrix_colmaj(A, M, K);
  printf("B\n");
  print_matrix_colmaj(B, K, N);
  printf("C\n");
  print_matrix_colmaj(C, M, N);

  memcpy(A_asm, A, M * K * sizeof(DTYPE));
  memcpy(B_asm, B, K * N * sizeof(DTYPE));
  memcpy(C_asm, C, M * N * sizeof(DTYPE));

  ref_matmul_colmaj(A, B, C, M, N, K);
  matmul(A_asm, B_asm, C_asm);

  printf("C out\n");
  print_matrix_colmaj(C, M, N);
  printf("C_asm out\n");
  print_matrix_colmaj(C_asm, M, N);

  int passed = isclose(C, C_asm, M * N);
  if (memcmp(A, A_asm, M * K * sizeof(DTYPE)) != 0 ||
      memcmp(B, B_asm, K * N * sizeof(DTYPE)) != 0) {
    printf("Test Failed: The kernel modified an input matrix.\n");
    passed = 0;
  }

  free(A);
  free(B);
  free(C);
  free(A_asm);
  free(B_asm);
  free(C_asm);

  if (passed) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }
}
