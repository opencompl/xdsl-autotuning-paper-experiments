#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(FFTW_DTYPE_IS_FLOAT)
  #define FFTW(name) fftwf_##name
  #define FFTW_DTYPE float
#elif defined(FFTW_DTYPE_IS_DOUBLE)
  #define FFTW(name) fftw_##name
  #define FFTW_DTYPE double
#else
  #error "You must define FFTW_DTYPE_IS_FLOAT or FFTW_DTYPE_IS_DOUBLE"
#endif

static FFTW(plan) plan;

// Planning is FFTW's autotuner, so it happens once here, outside the timed
// region, with the most thorough search FFTW offers.  The scratch arrays are
// out-of-place and SIMD-aligned like the harness's 64-byte-aligned buffers:
// `execute_dft` may only be handed arrays with the alignment it planned for.
void dft_init(void) {
  FFTW(complex) *in = FFTW(malloc)(sizeof(FFTW(complex)) * DFT_N);
  FFTW(complex) *out = FFTW(malloc)(sizeof(FFTW(complex)) * DFT_N);
  plan = FFTW(plan_dft_1d)(DFT_N, in, out, FFTW_FORWARD,
                           FFTW_EXHAUSTIVE | FFTW_PRESERVE_INPUT);
  if (!plan) {
    fprintf(stderr, "FFTW could not plan a DFT of size %d\n", DFT_N);
    exit(1);
  }
  FFTW(free)(in);
  FFTW(free)(out);
}

void dft(const FFTW_DTYPE *in, FFTW_DTYPE *out) {
  // FFTW_PRESERVE_INPUT makes the cast away from const sound.
  FFTW(execute_dft)(plan, (FFTW(complex) *)in, (FFTW(complex) *)out);
}
