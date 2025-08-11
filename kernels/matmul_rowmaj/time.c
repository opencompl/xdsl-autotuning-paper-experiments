#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../headers/gendata.h"
#include "../../headers/mnk.h"

#define NUM_ITERATIONS 128
#define CLOCKS_PER_USEC ((double)CLOCKS_PER_SEC / 1000000.0)

extern void matmul(DTYPE A[M * K], DTYPE B[K * N], DTYPE C[M * N]);

int main() {
  set_random_seed(42);

  DTYPE *A = aligned_alloc(64, M * K * sizeof(DTYPE));
  DTYPE *B = aligned_alloc(64, K * N * sizeof(DTYPE));
  DTYPE *C = aligned_alloc(64, (NUM_ITERATIONS + 1) * M * N * sizeof(DTYPE));

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  for (int i = 0; i < NUM_ITERATIONS + 1; i++) {
    fill_random_data(&C[i * M * N], M * N);
  }

  // Warm up the cache with one iteration
  matmul(A, B, C);

  // Timing loop - NUM_ITERATIONS iterations
  struct timespec ts_start;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    matmul(A, B, &C[(i + 1) * M * N]);
  }

  struct timespec ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double elapsed = ts_end.tv_nsec - ts_start.tv_nsec;

  double average_cycles = (double)elapsed / (double)NUM_ITERATIONS;
  printf("%f\n", average_cycles);

  free(A);
  free(B);
  free(C);
  return 0;
}
