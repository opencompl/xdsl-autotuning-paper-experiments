#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/utsname.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#include "../../headers/amx.h"
#include "../../headers/gendata.h"
#include "../../headers/mnk.h"

#define NUM_ITERATIONS 128

extern void matmul_init(void);
extern void matmul(DTYPE A[M * K], DTYPE B[K * N], CTYPE C[M * N]);

int main() {
  request_amx();
  matmul_init();

  DTYPE *A, *B;
  CTYPE *C;
  posix_memalign((void **)&A, 64, M * K * sizeof(DTYPE));
  posix_memalign((void **)&B, 64, K * N * sizeof(DTYPE));
  posix_memalign((void **)&C, 64, M * N * sizeof(CTYPE));
  
  set_random_seed(42);

  fill_random_data(A, M * K);
  fill_random_data(B, K * N);
  for (int i = 0; i < NUM_ITERATIONS + 1; i++) {
    fill_random_acc(C, M * N);
  }

  // Warm up the cache with one iteration
  matmul(A, B, C);
  
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    matmul(A, B, C);
  }

  free(A);
  free(B);
  free(C);
  
  return 0;
}
