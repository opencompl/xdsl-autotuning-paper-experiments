#include <libxsmm.h>
#include <stdio.h>
#include <stdlib.h>

#include "exp.h"

libxsmm_meltwfunction_unary exp_init(void) {
  // The approximation is part of this experiment, not an ambient preference.
  if (setenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX",
             EXP_DEGREE == 5 ? "1" : "0", 1) != 0) {
    perror("selecting the LIBXSMM exp approximation");
    return NULL;
  }
  libxsmm_init();
  libxsmm_set_target_arch(EXP_LIBXSMM_ARCH);
  libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
      LENGTH, 1, LENGTH, LENGTH, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
      LIBXSMM_DATATYPE_F32);
  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(
      LIBXSMM_MELTW_TYPE_UNARY_EXP, shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  if (!kernel) {
    fputs("LIBXSMM exp dispatch failed.\n", stderr);
    return NULL;
  }

  libxsmm_kernel_info info = {0};
  if (libxsmm_get_kernel_info((const void *)kernel, &info) != 0 ||
      info.is_reference_kernel) {
    fputs("Expected a generated LIBXSMM exp kernel, not a reference fallback.\n",
          stderr);
    return NULL;
  }
  return kernel;
}
