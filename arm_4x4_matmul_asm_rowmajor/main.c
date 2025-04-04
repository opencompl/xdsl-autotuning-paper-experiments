#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../headers/gendata.h"
#include "../headers/isclose.h"
#include "../headers/print_matrix.h"

#define SIZE 4

extern void matrix_mul_4x4_asm(float result[SIZE * SIZE], float A[SIZE * SIZE],
                               float B[SIZE * SIZE]);

void matrix_mul_4x4_ref(float result[SIZE * SIZE], float A[SIZE * SIZE],
                        float B[SIZE * SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      result[i * SIZE + j] = 0;
      for (int k = 0; k < SIZE; k++) {
        result[i * SIZE + j] += A[i * SIZE + k] * B[k * SIZE + j];
      }
    }
  }
}

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
  print_matrix(A, SIZE, SIZE);
  fill_random_data(B, SIZE * SIZE);
  print_matrix(B, SIZE, SIZE);

  copy_matrix(A, A_asm);
  copy_matrix(B, B_asm);

  matrix_mul_4x4_ref(C, A, B);
  matrix_mul_4x4_asm(C_asm, A_asm, B_asm);

  if (isclose(C, C_asm, SIZE * SIZE)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    printf("Matrix C (ref):\n");
    print_matrix(C, SIZE, SIZE);
    printf("\nMatrix C (asm):\n");
    print_matrix(C_asm, SIZE, SIZE);
    return 1;
  }
}
