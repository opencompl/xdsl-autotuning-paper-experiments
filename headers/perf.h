#ifndef PERF
#define PERF

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>

#ifdef USE_PAPI
#include <papi.h>
#endif

#define TIMETY long long

#define CHECK(call)                                                            \
  do {                                                                         \
    int __ret = (call);                                                        \
    if (__ret != PAPI_OK) {                                                    \
      fprintf(stderr, #call " failed: %s\n", PAPI_strerror(__ret));            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#ifdef USE_PAPI
int hardware_counters_available(void) {
  int res = 0;
  // Check the availability of the event
  int av = PAPI_query_event(PAPI_TOT_CYC);
  if (av == PAPI_OK) {
    // Check the level of paranoid
    FILE *f = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (f) {
      int val;
      if (fscanf(f, "%d", &val) == 1) {
        if (val > 2) {
          fprintf(stderr, "System too paranoid.\n");
          res = 0;
        } else {
          res = 1;
        }
        res = (val <= 2);
      }
      fclose(f);
    } else {
      fprintf(stderr,"/proc/sys/kernel/perf_event_paranoid not available.\n");
    }
  } else {
    fprintf(stderr,"%s\n", PAPI_strerror(av));
  }
  return res;
}
#endif

#ifdef USE_PAPI
int EventSet = PAPI_NULL;
TIMETY values[1];
#else
struct timespec ts_start;
struct timespec ts_end;
#endif

void time_init() {
#ifdef USE_PAPI
  int ret = PAPI_library_init(PAPI_VER_CURRENT);
  if (ret != PAPI_VER_CURRENT) { fprintf(stderr, "papi init failed\n"); exit(1); }
  CHECK(PAPI_create_eventset(&EventSet));

  int hw_counters = hardware_counters_available();
  if (!hw_counters) {
    exit(1);
  }
  CHECK(PAPI_add_event(EventSet, PAPI_TOT_CYC));
#endif
}

void time_start() {
#ifdef USE_PAPI
  CHECK(PAPI_start(EventSet));
#else
  clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif
}

TIMETY time_end(TIMETY freq) {
  TIMETY elapsed;
#ifdef USE_PAPI
  CHECK(PAPI_stop(EventSet, values));
  if (elapsed <= 0.0) {
    fprintf(stderr, "Measured cycles must not be negative.\n");
  }
  elapsed = values[0];
#else
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  if (ts_end.tv_nsec <= ts_start.tv_nsec) {
    fprintf(stderr, "ts_end.tv_nsec must not be smaller than ts_start.tv_nsec.\n");
  }
  elapsed = (ts_end.tv_nsec - ts_start.tv_nsec) * freq;
#endif
  if (elapsed <= 0.0) {
    fprintf(stderr, "Elapsed time must not be negative.\n");
    exit(1);
  }
  return elapsed;
}

#endif
