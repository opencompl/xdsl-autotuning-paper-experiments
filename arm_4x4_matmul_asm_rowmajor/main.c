#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void generate_random_matrix(float matrix[SIZE * SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      matrix[i * SIZE + j] = (float)(rand() % 10);
    }
  }
}

bool matrices_are_equal(float rm1[SIZE * SIZE], float rm2[SIZE * SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      float val1 = rm1[i * SIZE + j];
      float val2 = rm2[i * SIZE + j];

      if (fabs(val1 - val2) > 1e-6) {
        printf("Mismatch at (%d, %d): matrix1 = %f, matrix2 = %f\n", i, j, val1,
               val2);
        return false;
      }
    }
  }
  return true;
}

void copy_matrix(float src[SIZE * SIZE], float dest[SIZE * SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      dest[i * SIZE + j] = src[i * SIZE + j]; // Copy element
    }
  }
}

int main() {
  srand(time(NULL));

  float A[SIZE * SIZE], B[SIZE * SIZE], C[SIZE * SIZE], A_asm[SIZE * SIZE],
      B_asm[SIZE * SIZE], C_asm[SIZE * SIZE];

  generate_random_matrix(A);
  generate_random_matrix(B);

  copy_matrix(A, A_asm);
  copy_matrix(B, B_asm);

  matrix_mul_4x4_ref(C, A, B);
  matrix_mul_4x4_asm(C_asm, A_asm, B_asm);

  if (matrices_are_equal(C, C_asm)) {
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
