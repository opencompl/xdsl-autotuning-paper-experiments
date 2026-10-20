#ifndef ISCLOSE_H
#define ISCLOSE_H

#include <math.h>

// Returns index of first value not within tolerance, size otherwise.
static inline int first_mismatch_index(const DTYPE *lhs, const DTYPE *rhs,
                                       int size, double rtol, double atol) {
  int i = 0;
  for (; i < size; i++) {
    double l = (double)lhs[i];
    double r = (double)rhs[i];

    if (!isfinite(l) || !isfinite(r) || fabs(l - r) > rtol * fabs(r) + atol) {
      break;
    }
  }
  return i;
}

// Finite benchmark results only. Allow rounding differences from FMA/reordering,
// without narrowing double-precision results to float.
static inline int isclose(const DTYPE *lhs, const DTYPE *rhs, int size) {
  double rtol = sizeof(DTYPE) == sizeof(float) ? 1e-5 : 1e-12;
  double atol = sizeof(DTYPE) == sizeof(float) ? 1e-8 : 1e-12;
  int mismatch_index = first_mismatch_index(lhs, rhs, size, rtol, atol);
  return mismatch_index == size;
}

#endif // ISCLOSE_H
