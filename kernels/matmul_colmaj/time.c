#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#include "../../headers/gendata.h"
#include "../../headers/mnk.h"
#include "../../headers/perf.h"

#define TARGET_SIZE 268435456
#define NUM_ITERATIONS (TARGET_SIZE / (M * N * K))
#define CLOCKS_PER_USEC ((TIMETY)CLOCKS_PER_SEC / 1000000.0)

// C is a ring of buffers rather than one, and the loop below walks a slot per
// call.  The kernel reads C, accumulates into it and writes it back, so timing
// one buffer serialises consecutive calls on a store-to-load dependency
// through masked AVX-512 stores.  At M = N = K = 4 that stall was two thirds
// of the measurement, and it fell on the variants unevenly -- 52 cycles for
// LIBXSMM against 71 for CompXSMM, from the same instructions -- so curves
// crossed where the kernels do not differ.  By the time a slot comes round
// again its store has long retired.
//
// Half of L1, so that breaking the dependency does not buy an L2 miss in its
// place: a ring spilling out of L1 measures the cache instead of the kernel.
#define C_RING_BYTES 16384
// Enough slots to cover the store's latency several times over at the sizes
// where it is visible at all; past that a longer ring only costs footprint.
#define C_RING_MAX_SLOTS 32

extern void matmul(DTYPE A[M * K], DTYPE B[K * N], DTYPE C[M * N]);

int main() {

  DTYPE *A, *B, *C;
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));

  // Elements between two slots: M*N rounded up to a whole number of cache
  // lines, so every slot starts 64-byte aligned the way the single C did.
  const int stride =
      (int)(((M * N * sizeof(DTYPE) + 63) / 64) * (64 / sizeof(DTYPE)));
  // One slot once C alone fills the budget, which is every shape big enough
  // that the dependency is amortised anyway.
  int slots = C_RING_BYTES / (int)(stride * sizeof(DTYPE));
  if (slots < 1) {
    slots = 1;
  }
  if (slots > C_RING_MAX_SLOTS) {
    slots = C_RING_MAX_SLOTS;
  }

  // One slab, so the ring's footprint is exactly what the budget says.
  posix_memalign((void **)&C, 64, (size_t)slots * stride * sizeof(DTYPE));

  time_init();
  
  set_random_seed(42);

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  fill_random_data(C, slots * stride);

  // Warm up the cache with one iteration -- every slot, so none of them is
  // still cold when it is first timed.
  for (int slot = 0; slot < slots; slot++) {
    matmul(A, B, C + (size_t)slot * stride);
  }

  DTYPE *c = C;
  DTYPE *const last = C + (size_t)(slots - 1) * stride;

  time_start();
  
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    matmul(A, B, c);
    c = (c == last) ? C : c + stride;
  }
  
  TIMETY elapsed = time_end(FREQ);

  TIMETY average_cycles = elapsed / (TIMETY)NUM_ITERATIONS;
  printf("%Lf\n", average_cycles);

  free(A);
  free(B);
  free(C);

  return 0;
}
