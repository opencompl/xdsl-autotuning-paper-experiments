#ifndef ISCLOSE_H
#define ISCLOSE_H

#include <math.h>

// Returns index of first value not within tolerance, size otherwise.
static inline int first_mismatch_index(float *lhs, float *rhs, int size,
                                       float rtol, float atol) {
  int i = 0;
  for (; i < size; i++) {
    float l = lhs[i];
    float r = rhs[i];

    float diff = fabsf(l - r);

    if (diff > rtol * fabsf(r) + atol) {
      break;
    }
  }
  return i;
}

// Returns true if each pair of vales satisfies |l - r| <= |r| / 1e-5 + 1e-8
// https://pytorch.org/docs/stable/generated/torch.isclose.html
static inline int isclose(float *lhs, float *rhs, int size) {
  int mismatch_index = first_mismatch_index(lhs, rhs, size, 1e-5, 1e-8);
  return mismatch_index == size;
}

#endif // ISCLOSE_H
