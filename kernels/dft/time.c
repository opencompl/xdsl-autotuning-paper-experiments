#include <stdio.h>
#include <stdlib.h>

#include "../../headers/gendata.h"
#include "../../headers/perf.h"

static const int N = DFT_N;

// Called once before warmup and timing; per-call work belongs in dft.
extern void dft_init(void);
extern void dft(const DTYPE *in, DTYPE *out);

// About the same total work at every size.  Unlike the matmul, which
// accumulates into C, a DFT only writes its output, so consecutive calls into
// one buffer do not wait on each other and need no ring of outputs.
#define TARGET_SIZE 16777216
#define NUM_ITERATIONS (TARGET_SIZE / N > 0 ? TARGET_SIZE / N : 1)

int main() {
  dft_init();

  DTYPE *in, *out;
  posix_memalign((void **)&in, 64, 2 * N * sizeof(DTYPE));
  posix_memalign((void **)&out, 64, 2 * N * sizeof(DTYPE));

  time_init();

  set_random_seed(42);
  fill_random_data(in, 2 * N);

  // Warm up the cache with one iteration
  dft(in, out);

  time_start();

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    dft(in, out);
  }

  TIMETY elapsed = time_end(FREQ);

  TIMETY average_cycles = elapsed / (TIMETY)NUM_ITERATIONS;
  printf("%Lf\n", average_cycles);

  free(in);
  free(out);

  return 0;
}
