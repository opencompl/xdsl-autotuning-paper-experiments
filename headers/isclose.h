#ifndef ISCLOSE_H
#define ISCLOSE_H

#include <math.h>

#ifndef CTYPE
#define CTYPE DTYPE
#endif

// Returns index of first value not within tolerance, size otherwise.
static inline int first_mismatch_index(const CTYPE *lhs, const CTYPE *rhs,
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

// Compares results, so in the accumulator type. Finite benchmark results only.
// Allow rounding differences from FMA/reordering, without narrowing
// double-precision results to float.
static inline int isclose(const CTYPE *lhs, const CTYPE *rhs, int size) {
  double rtol = sizeof(CTYPE) == sizeof(float) ? 1e-5 : 1e-12;
  double atol = sizeof(CTYPE) == sizeof(float) ? 1e-8 : 1e-12;
  int mismatch_index = first_mismatch_index(lhs, rhs, size, rtol, atol);
  return mismatch_index == size;
}

#endif // ISCLOSE_H
