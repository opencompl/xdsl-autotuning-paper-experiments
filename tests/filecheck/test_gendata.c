// RUN: clang-20 -DDTYPE=float -o %t %s && %t | filecheck %s

#include "../../headers/gendata.h"
#include "../../headers/print_matrix.h"
#include <stdbool.h>

int main() {
  float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  set_random_seed(42);

  fill_random_data(A, 6);

  print_matrix(A, 1, 6);
  // CHECK: [0.00, 1.00, 5.00, 5.00, 3.00, 4.00]

  fill_random_data(A, 6);

  print_matrix(A, 1, 6);
  // CHECK-NEXT: [4.00, 4.00, 4.00, 3.00, 6.00, 3.00]

  set_random_seed(42);

  fill_random_data(A, 6);

  print_matrix(A, 1, 6);
  // CHECK-NEXT: [0.00, 1.00, 5.00, 5.00, 3.00, 4.00]

  return 0;
}
