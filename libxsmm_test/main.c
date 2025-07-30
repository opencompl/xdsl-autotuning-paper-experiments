#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void libxsmm_matmul(const double* A, const double* B, double* C);

void naive_matmul_column_major(const double* A, const double* B, double* C) {
  for (size_t i = 0; i < I; ++i) {
    for (size_t j = 0; j < J; ++j) {
      C[i * J + j] = 0.0;
      C[i * J + j] = 0.0;
    }
  }
  for (size_t j = 0; j < J; ++j)
    for (size_t i = 0; i < I; ++i)
      for (size_t k = 0; k < K; ++k)
        C[i + j * I] += A[i + k * I] * B[k + j * K];
}

int main() {
  
  double* A0 = malloc(I * K * sizeof(double));
  double* B0 = malloc(K * J * sizeof(double));
  double* C0 = malloc(I * J * sizeof(double));
  double* A1 = malloc(I * K * sizeof(double));
  double* B1 = malloc(K * J * sizeof(double));
  double* C1 = malloc(I * J * sizeof(double));

  srand((unsigned int)time(NULL));
  
  for (size_t i = 0; i < I * K; ++i) {
    double a = (double)(rand() % 10);
    A0[i] = a;
    A1[i] = a;
  }

  for (size_t i = 0; i < K * J; ++i) {
    double b = (double)(rand() % 10);
    B0[i] = b;
    B1[i] = b;
  }
  
  naive_matmul_column_major(A0, B0, C0);
  libxsmm_matmul(A1, B1, C1);

  for (size_t i = 0; i < I; ++i) {
    for (size_t j = 0; j < J; ++j) {
      double c0 = C0[i * J + j];
      double c1 = C1[i * J + j];
      if (fabs(c0 - c1) > 1e-10) {
        printf("Error: %f vs %f\n", c0, c1);
      }
    }
  }
  
  free(A0);
  free(B0);
  free(C0);
  free(A1);
  free(B1);
  free(C1);

  return 0;
}
