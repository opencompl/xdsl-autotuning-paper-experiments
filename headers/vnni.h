#ifndef VNNI_H
#define VNNI_H

#include <string.h>

// Copies column-major A (M x K) into the layout the kernel reads it in.  With
// VNNI_A that is LIBXSMM's VNNI-2 layout: the two K rows of each pair are
// interleaved, so element (m, k) sits at (k / 2) * 2 * M + 2 * m + k % 2 and
// one 32-bit lane holds the BF16 pair `vdpbf16ps` multiplies.  Otherwise A is
// copied as it is.
static inline void kernel_layout_a(const DTYPE *A, DTYPE *out, int M, int K) {
#ifdef VNNI_A
  for (int k = 0; k < K; k++) {
    for (int m = 0; m < M; m++) {
      out[(k / 2) * 2 * M + 2 * m + k % 2] = A[k * M + m];
    }
  }
#else
  memcpy(out, A, (size_t)M * K * sizeof(DTYPE));
#endif
}

#endif // VNNI_H
