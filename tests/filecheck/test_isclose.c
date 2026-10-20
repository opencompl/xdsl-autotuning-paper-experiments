// RUN: clang -DDTYPE=float -o %t %s && %t
// RUN: clang -DDTYPE=double -o %t %s && %t

#include <assert.h>
#include "../../headers/isclose.h"

int main(void) {
  DTYPE a[] = {0, -2, 3};
  DTYPE b[] = {0, -2, 4};
  assert(isclose(a, a, 3));
  assert(!isclose(a, b, 3));
  // Equality is within tolerance, and the relative scale belongs to rhs.
  assert(first_mismatch_index(a, b, 3, 0.25, 0) == 3);
  assert(first_mismatch_index(b, a, 3, 0.25, 0) == 2);
  assert(first_mismatch_index(a, b, 3, 0, 1) == 3);
  assert(first_mismatch_index(a, b, 3, 0, 0.5) == 2);
  DTYPE x = 2.5, y = 4;
  assert(first_mismatch_index(&x, &y, 1, 0.25, 0.5) == 1);
  assert(first_mismatch_index(&x, &y, 1, 0.25, 0.25) == 0);

  DTYPE bad[] = {NAN, INFINITY, -INFINITY};
  for (int i = 0; i < 3; i++) {
    assert(!isclose(bad + i, bad + i, 1));
    assert(!isclose(a, bad + i, 1));
    assert(!isclose(bad + i, a, 1));
  }
  if (sizeof(DTYPE) == sizeof(double)) {
    // These distinct doubles become identical when narrowed to float.
    x = 16777216;
    y = 16777217;
    assert(first_mismatch_index(&x, &y, 1, 0, 0) == 0);
    assert(!isclose(&x, &y, 1));
    x = 1;
    y = 1 + 5e-13;
    assert(isclose(&x, &y, 1));
    y = 1 + 2e-11;
    assert(!isclose(&x, &y, 1));
  }
  x = 0;
  y = sizeof(DTYPE) == sizeof(float) ? 5e-9 : 5e-13;
  assert(isclose(&x, &y, 1));
  y = sizeof(DTYPE) == sizeof(float) ? 2e-8 : 2e-12;
  assert(!isclose(&x, &y, 1));
}
