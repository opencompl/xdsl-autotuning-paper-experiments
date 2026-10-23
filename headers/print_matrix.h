#include <stdbool.h>
#include <stdio.h>

#ifndef CTYPE
#define CTYPE DTYPE
#endif

// One printer per element type: inputs are DTYPE, results CTYPE.  The cast is
// what lets a BF16 element through printf, which does not promote it.
#define DEFINE_PRINT_MATRIX(NAME, T)                                           \
  static inline void NAME(const T *matrix, int num_rows, int num_cols,         \
                          bool colmajor) {                                     \
    printf("[");                                                               \
    for (int i = 0; i < num_rows; i++) {                                       \
      printf("[");                                                             \
      for (int j = 0; j < num_cols; j++) {                                     \
        int idx = colmajor ? (j * num_rows + i) : (i * num_cols + j);          \
        printf("%.2f", (double)matrix[idx]);                                   \
        if (j < num_cols - 1) {                                                \
          printf(", ");                                                        \
        }                                                                      \
      }                                                                        \
      printf("]");                                                             \
      if (i < num_rows - 1) {                                                  \
        printf(",\n ");                                                        \
      }                                                                        \
    }                                                                          \
    printf("]\n");                                                             \
  }

DEFINE_PRINT_MATRIX(_print_matrix, DTYPE)
DEFINE_PRINT_MATRIX(_print_matrix_acc, CTYPE)

void print_matrix(const DTYPE *matrix, int num_rows, int num_cols) {
  _print_matrix(matrix, num_rows, num_cols, false);
}

void print_matrix_colmaj(const DTYPE *matrix, int num_rows, int num_cols) {
  _print_matrix(matrix, num_rows, num_cols, true);
}

void print_matrix_colmaj_acc(const CTYPE *matrix, int num_rows, int num_cols) {
  _print_matrix_acc(matrix, num_rows, num_cols, true);
}
