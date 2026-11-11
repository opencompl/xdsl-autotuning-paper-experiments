#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../headers/gendata.h"
#include "../../headers/ref_dft.h"

static const int N = DFT_N;

// Called once before any kernel invocation; per-call work belongs in dft.
extern void dft_init(void);
extern void dft(const DTYPE *in, DTYPE *out);

int main() {
  dft_init();
  set_random_seed(42);

  DTYPE *in, *in_copy, *out;
  long double *ref;
  posix_memalign((void **)&in, 64, 2 * N * sizeof(DTYPE));
  posix_memalign((void **)&in_copy, 64, 2 * N * sizeof(DTYPE));
  posix_memalign((void **)&out, 64, 2 * N * sizeof(DTYPE));
  ref = malloc(2 * N * sizeof(long double));

  fill_random_data(in, 2 * N);
  memcpy(in_copy, in, 2 * N * sizeof(DTYPE));
  // A kernel that leaves an output element unwritten fails the finite check.
  for (int i = 0; i < 2 * N; i++) {
    out[i] = NAN;
  }

  ref_dft(in, ref, N);
  dft(in, out);

  // Normwise relative error, the measure FFT accuracy is quoted in: an FFT's
  // rounding error scales with the whole signal, so outputs that are near zero
  // carry an absolute error that an element-wise relative check would reject.
  int passed = 1;
  long double err = 0.0L, norm = 0.0L;
  for (int i = 0; i < 2 * N; i++) {
    if (!isfinite((double)out[i])) {
      printf("out[%d] is not finite: %f\n", i, (double)out[i]);
      passed = 0;
    }
    long double d = (long double)out[i] - ref[i];
    err += d * d;
    norm += ref[i] * ref[i];
  }
  double rel_err = (double)sqrtl(err / norm);
  // Comfortably above O(eps * log N) for every size up to 2^20.
  double tol = sizeof(DTYPE) == sizeof(float) ? 1e-5 : 1e-13;
  printf("N = %d, relative L2 error = %.3e (tolerance %.0e)\n", N, rel_err,
         tol);
  if (!(rel_err <= tol)) {
    passed = 0;
  }

  if (memcmp(in, in_copy, 2 * N * sizeof(DTYPE)) != 0) {
    printf("Test Failed: The kernel modified its input.\n");
    passed = 0;
  }

  free(in);
  free(in_copy);
  free(out);
  free(ref);

  if (passed) {
    printf("\nTest Passed: The results are equal!\n");
    return 0;
  } else {
    printf("\nTest Failed: The results do not match.\n");
    return 1;
  }
}
