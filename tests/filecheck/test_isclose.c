// RUN: clang -DDTYPE=float -o %t %s && %t | filecheck %s

#include <stdio.h>

#include "../../headers/isclose.h"
#include "../../headers/print_matrix.h"

void print_mismatch_index(char *lhs_name, float *lhs, char *rhs_name,
                          float *rhs, int size, float rtol, float atol) {
  int index = first_mismatch_index(lhs, rhs, size, rtol, atol);
  if (index == size) {
    printf("%s matches %s, rtol: %.10f, atol: %.10f\n", lhs_name, rhs_name,
           rtol, atol);
  } else {
    printf("%s[%d] (%.10f) does not match %s[%d] (%.10f), rtol: %.10f, atol: "
           "%.10f\n",
           lhs_name, index, lhs[index], rhs_name, index, rhs[index], rtol,
           atol);
  }
}

int main() {
#define LEN 6

  printf("Length of A and B: %d\n", LEN);
  // CHECK: Length of A and B: 6

  float A[LEN] = {0.0f, 1.0f, 10.0f, 100.0f, 1000.0f, NAN};
  float B[LEN] = {0.0f, 1.0f, 10.0f, 100.0f, 1011.0f, NAN};

  print_matrix(A, 1, LEN);
  // CHECK: [0.00, 1.00, 10.00, 100.00, 1000.00, nan]
  print_matrix(B, 1, LEN);
  // CHECK: [0.00, 1.00, 10.00, 100.00, 1011.00, nan]

  char *MATCHES[2] = {"does not match", "matches"};

  printf("A %s A\n", MATCHES[isclose(A, A, LEN)]);
  // CHECK: A matches A

  printf("B %s B\n", MATCHES[isclose(B, B, LEN)]);
  // CHECK: B matches B

  printf("A %s B\n", MATCHES[isclose(A, B, LEN)]);
  // CHECK: A does not match B

  print_mismatch_index("A", A, "A", A, LEN, 1e-5f, 1e-8f);
  // CHECK: A matches A
  // CHECK-SAME: , rtol: 0.0000100000, atol: 0.0000000100

  print_mismatch_index("B", B, "B", B, LEN, 1e-5f, 1e-8f);
  // CHECK: B matches B
  // CHECK-SAME: , rtol: 0.0000100000, atol: 0.0000000100

  print_mismatch_index("A", A, "B", B, LEN, 1e-5f, 1e-8f);
  // CHECK: A[4] (1000.0000000000) does not match B[4] (1011.0000000000)
  // CHECK-SAME: , rtol: 0.0000100000, atol: 0.0000000100

  print_mismatch_index("A", A, "B", B, LEN, 0.02, 1e-8f);
  // CHECK: A matches B
  // CHECK-SAME: , rtol: 0.0199999996, atol: 0.0000000100

  print_mismatch_index("A", A, "B", B, LEN, 1e-5f, 11.0f);
  // CHECK: A matches B
  // CHECK-SAME: , rtol: 0.0000100000, atol: 11.0000000000

  return 0;
}
