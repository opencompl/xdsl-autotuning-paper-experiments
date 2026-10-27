#include <math.h>
#include <stdio.h>

#include "exp.h"
#include "../../headers/perf.h"

#define ITERATIONS (268435456 / LENGTH)

int main(void) {
  libxsmm_meltwfunction_unary kernel = exp_init();
  if (!kernel) return 1;

  _Alignas(64) float input[LENGTH], output[LENGTH];
  fill_exp_input(input, 42);
  libxsmm_meltw_unary_param param = {0};
  param.in.primary = input;
  param.out.primary = output;
  time_init();
  for (int i = 0; i < 1000; ++i) kernel(&param);

  // Out-of-place exp never reads its output, so repeated calls do not have
  // GEMM's accumulating-C dependency. Keep input, output, and params fixed.
  time_start();
  for (int i = 0; i < ITERATIONS; ++i) kernel(&param);
  TIMETY elapsed = time_end(FREQ);

  float checksum = 0;
  for (int i = 0; i < LENGTH; ++i) checksum += output[i];
  if (!isfinite(checksum)) {
    fputs("Nonfinite exp output.\n", stderr);
    return 1;
  }
  printf("%Lf\n", elapsed / ITERATIONS);
  return 0;
}
