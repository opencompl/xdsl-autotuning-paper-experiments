#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../headers/gendata.h"
#include "../headers/isclose.h"
#include "../headers/print_matrix.h"
#include "../headers/ref_matmul.h"

#define SIZE 4

extern void matmul_colmaj(float result[SIZE * SIZE], float A[SIZE * SIZE],
                          float B[SIZE * SIZE]);

int main() {
  set_random_seed(42);

  float A[SIZE * SIZE], B[SIZE * SIZE], C[SIZE * SIZE], A_colmaj[SIZE * SIZE],
      B_colmaj[SIZE * SIZE], C_colmaj[SIZE * SIZE];

  fill_random_data(A, SIZE * SIZE);
  fill_random_data(B, SIZE * SIZE);
  fill_random_data(C, SIZE * SIZE);

  printf("A\n");
  print_matrix(A, SIZE, SIZE);
  printf("B\n");
  print_matrix(B, SIZE, SIZE);
  printf("C\n");
  print_matrix(C, SIZE, SIZE);

  transpose(A_colmaj, A, SIZE, SIZE);
  transpose(B_colmaj, B, SIZE, SIZE);
  transpose(C_colmaj, C, SIZE, SIZE);

  printf("A_colmaj\n");
  print_matrix(A_colmaj, SIZE, SIZE);
  printf("B_colmaj\n");
  print_matrix(B_colmaj, SIZE, SIZE);
  printf("C_colmaj\n");
  print_matrix(C_colmaj, SIZE, SIZE);

  ref_matmul(C, A, B, SIZE, SIZE, SIZE);
  matmul_colmaj(C_colmaj, A_colmaj, B_colmaj);

  printf("C out\n");
  print_matrix(C, SIZE, SIZE);

  float res[SIZE * SIZE];

  printf("C_asm out\n");
  transpose(res, C_colmaj, SIZE, SIZE);

  print_matrix(res, SIZE, SIZE);

  if (isclose(C, res, SIZE * SIZE)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }
}
