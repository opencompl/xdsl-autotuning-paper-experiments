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

extern void matmul(float C[M * N], float A[M * K], float B[K * N]);

int main() {
  set_random_seed(42);

  float A[M * K], B[K * N];
  float C[(NUM_ITERATIONS + 1) * M * N];

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  for (int i = 0; i < NUM_ITERATIONS + 1; i++) {
    fill_random_data(&C[i * M * N], M * N);
  }

  // Warm up the cache with one iteration
  matmul(C, A, B);

  // Timing loop - NUM_ITERATIONS iterations
  struct timespec ts_start;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    matmul(&C[(i + 1) * M * N], A, B);
  }

  struct timespec ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  double elapsed = ts_end.tv_nsec - ts_start.tv_nsec;

  double average_cycles = (double)elapsed / (double)NUM_ITERATIONS;
  printf("%f\n", average_cycles);
  return 0;
}
