#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../headers/gendata.h"
#include "../headers/isclose.h"
#include "../headers/mnk.h"
#include "../headers/print_matrix.h"
#include "../headers/ref_matmul.h"

extern void matmul(float result[M * N], float A[M * K], float B[K * N]);

int main() {
  set_random_seed(42);

  float A[M * K], B[K * N], C[M * N], A_asm[M * K], B_asm[K * N], C_asm[M * N];

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  fill_random_data(C, M * N);
  printf("A\n");
  print_matrix(A, M, K);
  printf("B\n");
  print_matrix(B, K, N);
  printf("C\n");
  print_matrix(C, M, N);

  memcpy(A_asm, A, M * K * sizeof(float));
  memcpy(B_asm, B, K * N * sizeof(float));
  memcpy(C_asm, C, M * N * sizeof(float));

  ref_matmul(C, A, B, M, N, K);
  matmul(C_asm, A_asm, B_asm);

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
}
