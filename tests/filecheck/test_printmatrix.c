// RUN: clang-20 -o %t %s && %t | filecheck %s

#include "../../headers/print_matrix.h"
#include <stdbool.h>

int main() {
  float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  printf("Row major\n");

  print_matrix(data, 1, 6);
  // CHECK: [1.00, 2.00, 3.00, 4.00, 5.00, 6.00]

  print_matrix(data, 2, 3);
  // CHECK: [1.00, 2.00, 3.00;
  // CHECK:  4.00, 5.00, 6.00]

  print_matrix(data, 3, 2);
  // CHECK: [1.00, 2.00;
  // CHECK:  3.00, 4.00;
  // CHECK:  5.00, 6.00]

  print_matrix(data, 6, 1);
  // CHECK: [1.00;
  // CHECK:  2.00;
  // CHECK:  3.00;
  // CHECK:  4.00;
  // CHECK:  5.00;
  // CHECK:  6.00]

  printf("Column major\n");

  print_matrix_colmaj(data, 1, 6);
  // CHECK: [1.00, 2.00, 3.00, 4.00, 5.00, 6.00]

  print_matrix_colmaj(data, 2, 3);
  // CHECK: [1.00, 3.00, 5.00;
  // CHECK:  2.00, 4.00, 6.00]

  print_matrix_colmaj(data, 3, 2);
  // CHECK: [1.00, 4.00;
  // CHECK:  2.00, 5.00;
  // CHECK:  3.00, 6.00]

  print_matrix_colmaj(data, 6, 1);
  // CHECK: [1.00;
  // CHECK:  2.00;
  // CHECK:  3.00;
  // CHECK:  4.00;
  // CHECK:  5.00;
  // CHECK:  6.00]

  return 0;
}
