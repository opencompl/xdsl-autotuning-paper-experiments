#ifndef EXP_BENCH_H
#define EXP_BENCH_H

#include <libxsmm_typedefs.h>
#include <stdint.h>

_Static_assert(LENGTH > 0 && LENGTH % 16 == 0,
               "The initial FP32 exp experiment uses full AVX-512 vectors");
_Static_assert(EXP_DEGREE == 3 || EXP_DEGREE == 5,
               "The FP32 exp experiment supports polynomial degrees 3 and 5");

#define EXP_RTOL (EXP_DEGREE == 3 ? 1e-3 : 1e-6)

// All implementations return the same unary ABI, called indirectly by the
// harness. Initialization (including JIT dispatch) is never timed.
libxsmm_meltwfunction_unary exp_init(void);

static inline void fill_exp_input(float *input, uint32_t seed) {
  for (int i = 0; i < LENGTH; ++i) {
    // Deterministic fractional inputs in [-10, 10), not GEMM's integers 0..9.
    seed = seed * 1664525u + 1013904223u;
    input[i] = -10.0f + (float)(seed >> 8) * (20.0f / 16777216.0f);
  }
}

#endif
