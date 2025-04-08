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

extern void matrix_mul_4x4_asm(float result[SIZE * SIZE], float A[SIZE * SIZE],
                               float B[SIZE * SIZE]);

void copy_matrix(float src[SIZE * SIZE], float dest[SIZE * SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      dest[i * SIZE + j] = src[i * SIZE + j]; // Copy element
    }
  }
}

int main() {
  set_random_seed(42);

  float A[SIZE * SIZE], B[SIZE * SIZE], C[SIZE * SIZE], A_asm[SIZE * SIZE],
      B_asm[SIZE * SIZE], C_asm[SIZE * SIZE];

  fill_random_data(A, SIZE * SIZE);
  fill_random_data(B, SIZE * SIZE);
  fill_random_data(C, SIZE * SIZE);
  printf("A\n");
  print_matrix(A, SIZE, SIZE);
  printf("B\n");
  print_matrix(B, SIZE, SIZE);
  printf("C\n");
  print_matrix(C, SIZE, SIZE);

  copy_matrix(A, A_asm);
  copy_matrix(B, B_asm);
  copy_matrix(C, C_asm);

  ref_matmul(C, A, B, SIZE, SIZE, SIZE);
  matrix_mul_4x4_asm(C_asm, A_asm, B_asm);

  printf("C out\n");
  print_matrix(C, SIZE, SIZE);
  printf("C_asm out\n");
  print_matrix(C_asm, SIZE, SIZE);

  if (isclose(C, C_asm, SIZE * SIZE)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }
}
