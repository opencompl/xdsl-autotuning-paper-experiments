#include <errno.h>
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

extern void matmul(DTYPE A[M * K], DTYPE B[K * N], DTYPE C[M * N]);

int main() {

  unsigned long long num_iterations = NUM_ITERATIONS;
  const char *num_iterations_override = getenv("NUM_ITERATIONS");
  if (num_iterations_override != NULL) {
    char *end;
    errno = 0;
    num_iterations = strtoull(num_iterations_override, &end, 10);
    if (*num_iterations_override == '\0' || *num_iterations_override == '-' ||
        *end != '\0' || errno == ERANGE || num_iterations == 0) {
      fprintf(stderr, "NUM_ITERATIONS must be a positive integer\n");
      return 1;
    }
  }

  DTYPE *A, *B, *C;
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(DTYPE));

  time_init();

  set_random_seed(42);

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  for (unsigned long long i = 0; i < num_iterations + 1; i++) {
    fill_random_data(C, M * N);
  }

  // Warm up the cache with one iteration
  matmul(A, B, C);

  time_start();

  for (unsigned long long i = 0; i < num_iterations; i++) {
    matmul(A, B, C);
  }

  TIMETY elapsed = time_end(FREQ);

  TIMETY average_cycles = elapsed / (TIMETY)num_iterations;
  printf("%Lf\n", average_cycles);

  free(A);
  free(B);
  free(C);

  return 0;
}
