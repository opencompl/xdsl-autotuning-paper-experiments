// REQUIRES: libxsmm-headers
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=3 -DLENGTH=16 kernels/exp/test.c %s -lm -o %t && %t | filecheck %s --check-prefix=CUBIC
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=3 -DLENGTH=1024 kernels/exp/test.c %s -lm -o %t && %t | filecheck %s --check-prefix=CUBIC
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=5 -DLENGTH=16 kernels/exp/test.c %s -lm -o %t && %t | filecheck %s --check-prefix=QUINTIC
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=5 -DLENGTH=1024 kernels/exp/test.c %s -lm -o %t && %t | filecheck %s --check-prefix=QUINTIC
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=3 -DLENGTH=16 -DFAULT=1 kernels/exp/test.c %s -lm -o %t && not %t | filecheck %s --check-prefix=WRONG
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=5 -DLENGTH=16 -DFAULT=2 kernels/exp/test.c %s -lm -o %t && not %t | filecheck %s --check-prefix=WRONG
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=3 -DLENGTH=16 -DFAULT=3 kernels/exp/test.c %s -lm -o %t && not %t | filecheck %s --check-prefix=INPUT
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=5 -DLENGTH=16 -DFAULT=4 kernels/exp/test.c %s -lm -o %t && not %t | filecheck %s --check-prefix=GUARD
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=3 -DLENGTH=16 -DFAULT=5 kernels/exp/test.c %s -lm -o %t && %t | filecheck %s --check-prefix=CUBIC
// RUN: clang %libxsmm_cflags -DEXP_DEGREE=5 -DLENGTH=16 -DFAULT=5 kernels/exp/test.c %s -lm -o %t && not %t | filecheck %s --check-prefix=WRONG

#include <math.h>
#include "../../kernels/exp/exp.h"

static void reference(const libxsmm_meltw_unary_param *param) {
  float *input = param->in.primary;
  float *output = param->out.primary;
  for (int i = 0; i < LENGTH; ++i) output[i] = expf(input[i]);
#if FAULT == 1 || FAULT == 5
  // Even small outputs must be checked relatively, without an absolute floor.
  int smallest = 0;
  for (int i = 1; i < LENGTH; ++i)
    if (input[i] < input[smallest]) smallest = i;
  output[smallest] *= FAULT == 1 ? 1.01f : 1.00001f;
#elif FAULT == 2
  output[0] = NAN;
#elif FAULT == 3
  input[0] += 1;
#elif FAULT == 4
  output[LENGTH] = 0;
#endif
}

libxsmm_meltwfunction_unary exp_init(void) { return reference; }

// CUBIC: Test Passed: degree-3 exp is within 0.001 relative error; input and guards unchanged.
// QUINTIC: Test Passed: degree-5 exp is within 1e-06 relative error; input and guards unchanged.
// WRONG: Test Failed: exp mismatch
// INPUT: Test Failed: The kernel modified its input.
// GUARD: Test Failed: The kernel overwrote an output guard.
