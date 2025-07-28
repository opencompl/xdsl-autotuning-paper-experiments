// RUN: clang-20 -o %t %s && %t | filecheck %s

#include "../../headers/print_matrix.h"
#include "../../headers/ref_matmul.h"
#include <stdbool.h>

int main() {
  float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float B[6] = {7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f};
  float C[6] = {3.0f, 4.0f, 5.0f, 6.0f};

  float A_T[6], B_T[6], C_T[6];

  transpose(A_T, A, 2, 3);
  transpose(B_T, B, 3, 2);
  transpose(C_T, C, 2, 2);

  printf("C_T:\n");
  // CHECK: C_T:

  print_matrix(C_T, 2, 2);

  // CHECK-NEXT: [3.00, 5.00],
  // CHECK-NEXT: [4.00, 6.00]

  ref_matmul(C, A, B, 2, 2, 3);

  printf("C out:\n");
  // CHECK: C out:

  print_matrix(C, 2, 2);

  // CHECK-NEXT: [31.00, 18.00],
  // CHECK-NEXT: [84.00, 50.00]

  ref_matmul_colmaj(C_T, A_T, B_T, 2, 2, 3);

  printf("C_T out:\n");
  // CHECK: C_T out:

  print_matrix(C_T, 2, 2);

  // CHECK-NEXT: [31.00, 84.00],
  // CHECK-NEXT: [18.00, 50.00]

  return 0;
}
