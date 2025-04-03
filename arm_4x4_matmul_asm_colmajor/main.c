#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../headers/isclose.h"

#define SIZE 4

extern void matrix_mul_4x4_asm(float result[SIZE * SIZE], float A[SIZE * SIZE],
                               float B[SIZE * SIZE]);

void matrix_mul_4x4_ref(float result[SIZE][SIZE], float A[SIZE][SIZE],
                        float B[SIZE][SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      result[i][j] = 0;
      for (int k = 0; k < SIZE; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void generate_random_matrix(float matrix[SIZE][SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      matrix[i][j] = (float)(rand() % 10);
    }
  }
}

void row_maj_to_col_maj(float arr_rm[SIZE][SIZE], float arr_cm[SIZE * SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      arr_cm[(j * SIZE) + i] = arr_rm[i][j];
    }
  }
}

int main() {
  srand(time(NULL));

  float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE], A_colmaj[SIZE * SIZE],
      B_colmaj[SIZE * SIZE], C_colmaj[SIZE * SIZE];

  generate_random_matrix(A);
  generate_random_matrix(B);

  row_maj_to_col_maj(A, A_colmaj);
  row_maj_to_col_maj(B, B_colmaj);

  matrix_mul_4x4_ref(C, A, B);
  matrix_mul_4x4_asm(C_colmaj, A_colmaj, B_colmaj);

  if (is_close(C, C_colmaj, SIZE * SIZE)) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    printf("Matrix C (ref):\n");
    print_matrix(C);
    printf("\nMatrix C (asm):\n");
    print_matrix_colmaj(C_colmaj);
    return 1;
  }
}
