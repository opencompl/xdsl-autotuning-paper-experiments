#include <math.h>
#include <stdio.h>
#include <string.h>

#include "exp.h"

int main(void) {
  libxsmm_meltwfunction_unary kernel = exp_init();
  if (!kernel) {
    puts("Test Failed: No exp kernel.");
    return 1;
  }

  // Cache-line guards keep the logical arrays aligned while detecting writes
  // outside them. The initial experiment has no strided or partial vectors.
  _Alignas(64) float input[LENGTH + 32], output[LENGTH + 32];
  float saved[LENGTH + 32];
  libxsmm_meltw_unary_param param = {0};
  param.in.primary = input + 16;
  param.out.primary = output + 16;
  double max_relative = 0;

  for (int trial = 0; trial < 2; ++trial) {
    for (int i = 0; i < LENGTH + 32; ++i) {
      input[i] = output[i] = -12345.0f;
    }
    fill_exp_input(input + 16, 42 + trial);
    memcpy(saved, input, sizeof(input));
    for (int i = 0; i < LENGTH; ++i) output[i + 16] = NAN;
    kernel(&param);

    if (memcmp(input, saved, sizeof(input)) != 0) {
      puts("Test Failed: The kernel modified its input.");
      return 1;
    }
    for (int i = 0; i < 16; ++i) {
      if (output[i] != -12345.0f || output[LENGTH + 16 + i] != -12345.0f) {
        puts("Test Failed: The kernel overwrote an output guard.");
        return 1;
      }
    }
    for (int i = 0; i < LENGTH; ++i) {
      double reference = exp((double)input[i + 16]);
      double relative = fabs((double)output[i + 16] - reference) / reference;
      // Degree-specific bounds on our finite [-10, 10) experiment, with no
      // absolute floor that would conceal errors in small positive outputs.
      if (!isfinite(output[i + 16]) || relative > EXP_RTOL) {
        printf("Test Failed: exp mismatch at %d (x=%g, got=%g, reference=%g).\n",
               i, input[i + 16], output[i + 16], reference);
        return 1;
      }
      if (relative > max_relative) max_relative = relative;
    }
  }

  printf("Maximum relative error: %.9g\n", max_relative);
  printf("Test Passed: degree-%d exp is within %g relative error; "
         "input and guards unchanged.\n", EXP_DEGREE, EXP_RTOL);
  return 0;
}
