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

// The kernel accumulates into C, so timing one buffer serialises consecutive
// calls on a store-to-load dependency that dominates the small shapes.  C is a
// ring of buffers instead, walked a slot per call, sized to half of L1 so that
// breaking the dependency does not buy an L2 miss in its place.
#define C_RING_BYTES 16384
#define C_RING_MAX_SLOTS 32

extern void matmul(DTYPE A[M * K], DTYPE B[K * N], DTYPE C[M * N]);

int main() {

  DTYPE *A, *B, *C;
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));

  // Elements between two slots: M*N rounded up to whole cache lines, so every
  // slot starts 64-byte aligned the way the single C did.
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
  posix_memalign((void **)&C, 64, (size_t)slots * stride * sizeof(DTYPE));

  time_init();
  
  set_random_seed(42);

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  fill_random_data(C, slots * stride);

  // Warm up the cache with one iteration, over every slot
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
