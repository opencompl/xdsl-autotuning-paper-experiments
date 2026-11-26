#ifndef REF_DFT_H
#define REF_DFT_H

#include <math.h>

// out[k] = sum_j in[j] * exp(-2*pi*i * j*k / n), with in and out interleaved
// complex (re, im) arrays of n points.  The direct O(n^2) sum, in long double
// and with the exponent reduced mod n first, so the reference does not share
// an FFT's error profile.
void ref_dft(const DTYPE *in, long double *out, int n) {
  const long double pi = 3.141592653589793238462643383279502884L;
  for (int k = 0; k < n; k++) {
    long double re = 0.0L, im = 0.0L;
    for (int j = 0; j < n; j++) {
      long double angle = -2.0L * pi * (long double)((long)j * k % n) / n;
      long double c = cosl(angle), s = sinl(angle);
      long double x = in[2 * j], y = in[2 * j + 1];
      re += x * c - y * s;
      im += x * s + y * c;
    }
    out[2 * k] = re;
    out[2 * k + 1] = im;
  }
}

#endif // REF_DFT_H
