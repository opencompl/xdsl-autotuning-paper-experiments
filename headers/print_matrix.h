

#include <stdbool.h>
#include <stdio.h>

void _print_matrix(float *matrix, int num_rows, int num_cols, bool colmajor) {
  printf("[");
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      int idx = colmajor ? (j * num_rows + i) : (i * num_cols + j);
      printf("%.2f", matrix[idx]);
      if (j < num_cols - 1) {
        printf(", ");
      }
    }
    if (i < num_rows - 1) {
      printf(";\n ");
    }
  }
  printf("]\n");
}

void print_matrix(float *matrix, int num_rows, int num_cols) {
  _print_matrix(matrix, num_rows, num_cols, false);
}

void print_matrix_colmaj(float *matrix, int num_rows, int num_cols) {
  _print_matrix(matrix, num_rows, num_cols, true);
}
