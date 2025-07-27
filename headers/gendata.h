#ifndef GENDATA_H
#define GENDATA_H

#include <stdint.h>

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

void fill_random_data(float *data, int size) {
  for (int i = 0; i < size; i++) {
    // Use modulo 10 to get values between 0 and 9
    data[i] = (float)(get_random_next() % 10);
  }
}

void fill_zeros(float *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 0.0f;
  }
}

void fill_identity(float *data, int width, int height) {
  fill_zeros(data, width * height);
  int size;
  if (width < height) {
    size = width;
  } else {
    size = height;
  }
  for (int i = 0; i < size; i++) {
    data[width * i + i] = 1.0f;
  }
}

#endif // GENDATA_H
