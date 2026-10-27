// Differential oracle for the literal generator, separate from the benchmark's
// double-exp accuracy test. Include underflow lanes so the quintic's blend
// cannot silently use a wrong predicate or reversed operands.
#include <libxsmm.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void exp_kernel(const libxsmm_meltw_unary_param *param);

int main(void) {
  if (setenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX",
             EXP_DEGREE == 5 ? "1" : "0", 1)) return 1;
  libxsmm_init();
  libxsmm_set_target_arch("skx");
  libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
      LENGTH, 1, LENGTH, LENGTH, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
      LIBXSMM_DATATYPE_F32);
  libxsmm_meltwfunction_unary native = libxsmm_dispatch_meltw_unary(
      LIBXSMM_MELTW_TYPE_UNARY_EXP, shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  if (!native) return 2;
  float input[LENGTH + 32], saved[LENGTH + 32];
  float expected[LENGTH + 32], actual[LENGTH + 32];
  const float edges[] = {-INFINITY, -104, -100, -87.34f, -87.33f,
                        -0.0f, 0.0f, 0.34657359f, 87, 88.72f, 89, INFINITY, NAN};
  uint32_t seed = 314159;
  for (int trial = 0; trial < 128; ++trial) {
    for (int i = 0; i < LENGTH + 32; ++i) {
      seed = seed * 1664525u + 1013904223u;
      input[i] = -100.0f + (float)(seed >> 8) * (200.0f / 16777216.0f);
      expected[i] = actual[i] = -12345.0f;
    }
    // Rotate edge values through different SIMD lanes and register blocks.
    for (unsigned i = 0; i < sizeof(edges) / sizeof(*edges); ++i)
      input[16 + (trial * 13 + i) % LENGTH] = edges[i];
    memcpy(saved, input, sizeof(input));
    libxsmm_meltw_unary_param param = {0};
    param.in.primary = input + 16;
    param.out.primary = expected + 16;
    native(&param);
    param.out.primary = actual + 16;
    exp_kernel(&param);
    for (int i = 0; i < LENGTH + 32; ++i) {
      if (memcmp(&expected[i], &actual[i], sizeof(float)) &&
          !(isnan(expected[i]) && isnan(actual[i]))) {
        fprintf(stderr, "trial %d lane %d: x=%a native=%a port=%a\n",
                trial, i - 16, input[i], expected[i], actual[i]);
        return 3;
      }
    }
    if (memcmp(input, saved, sizeof(input))) return 4;
  }
  puts("Native and generated exp agree; input and guards unchanged.");
  return 0;
}
