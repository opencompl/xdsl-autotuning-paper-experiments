#ifndef GENDATA_H
#define GENDATA_H

#include <stdint.h>

// C, the accumulator, has the element type of A and B unless a mixed-precision
// GEMM (BF16 inputs, F32 accumulation) names its own.
#ifndef CTYPE
#define CTYPE DTYPE
#endif

// Ranq1 from Numerical Recipes
// https://numerical.recipes/book.html
// Page 351

// The algorithm doesn't matter too much as long as it gives the same results on
// different platforms.

#define INITIAL_STATE 4101842887655102017LL

static unsigned long long _state = INITIAL_STATE;

static unsigned long long get_random_next() {
  _state ^= _state >> 21;
  _state ^= _state << 35;
  _state ^= _state >> 4;
  return _state * 2685821657736338717LL;
}

static void set_random_seed(unsigned int seed) {
  _state = INITIAL_STATE ^ seed;
  _state = get_random_next();
}

// Integers 0..9 are exact in every element type, BF16 included, and so are the
// products and sums the shapes we sweep make of them.  Both fill functions draw
// from the same stream, so a C filled as CTYPE gets the values it would as DTYPE.
#define FILL_RANDOM_DATA(T)                                                    \
  for (int i = 0; i < size; i++) {                                             \
    data[i] = (T)(float)(get_random_next() % 10);                              \
  }

static inline void fill_random_data(DTYPE *data, int size) {
  FILL_RANDOM_DATA(DTYPE)
}

static inline void fill_random_acc(CTYPE *data, int size) {
  FILL_RANDOM_DATA(CTYPE)
}

#endif // GENDATA_H
