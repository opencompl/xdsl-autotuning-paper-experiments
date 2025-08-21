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

#define NUM_ITERATIONS 128
#define CLOCKS_PER_USEC ((double)CLOCKS_PER_SEC / 1000000.0)

extern void matmul(DTYPE A[M * K], DTYPE B[K * N], DTYPE C[M * N]);

#if __has_include(<papi.h>)
#include <papi.h>
#endif

#define CHECK(call) do { \
  int __ret = (call); \
  if (__ret != PAPI_OK) { \
    fprintf(stderr, #call " failed: %s\n", PAPI_strerror(__ret)); \
    exit(1); \
  } \
} while(0)

int hardware_counters_available(void) {
  int res = 0;
#if __has_include(<papi.h>)
  // Check the availability of the event
  int av = PAPI_query_event(PAPI_TOT_CYC);
  if (av == PAPI_OK) {
    // Check the level of paranoid
    FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (f) {
      int val;
      if (fscanf(f, "%d", &val) == 1) {
        res = (val <= 2);
      }
      fclose(f);
    }
  }
#endif
  return res;
}

int main() {

  DTYPE *A, *B, *C;
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(DTYPE));
  
  long long values[1];
  struct timespec ts_start;
  struct timespec ts_end;
  double elapsed;
  
#if __has_include(<papi.h>)
  int ret = PAPI_library_init(PAPI_VER_CURRENT);
  if (ret != PAPI_VER_CURRENT) { fprintf(stderr, "papi init failed\n"); return 1; }
  int EventSet = PAPI_NULL;
  CHECK(PAPI_create_eventset(&EventSet));
#endif
  
  int hw_counters = hardware_counters_available();

  if (hw_counters) {
#if __has_include(<papi.h>)
    CHECK(PAPI_add_event(EventSet, PAPI_TOT_CYC));
#endif
  }
    
  set_random_seed(42);

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  for (int i = 0; i < NUM_ITERATIONS + 1; i++) {
    fill_random_data(C, M * N);
  }

  // Warm up the cache with one iteration
  matmul(A, B, C);
  
  // Start the counters
  if (hw_counters) {
#if __has_include(<papi.h>)
    CHECK(PAPI_start(EventSet));
    PAPI_start(EventSet);
#endif
  }
  else {
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
  }
  
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    matmul(A, B, C);
  }

  // Stop the counters
  if (hw_counters) {
#if __has_include(<papi.h>)
    CHECK(PAPI_stop(EventSet, values));
#endif
    elapsed = (double) values[0] * FREQ;
  }
  else {
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  elapsed = ts_end.tv_nsec - ts_start.tv_nsec;
  }
  
  double average_cycles = (double)elapsed / (double)NUM_ITERATIONS;
  printf("%f\n", average_cycles);

  free(A);
  free(B);
  free(C);
  
  return 0;
}
