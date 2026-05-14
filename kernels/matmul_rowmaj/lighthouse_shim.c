// C shim that adapts the Lighthouse-emitted memref calling convention to the
// plain `void matmul(A, B, C)` signature `test.c` / `time.c` expect.
//
// Lighthouse's `--func lh_matmul_inner` lowering emits two LLVM functions:
//   * `lh_matmul_inner` — the raw MLIR memref ABI, with each `memref<MxNxT>`
//     exploded into (allocated_ptr, aligned_ptr, offset, sizes..., strides...).
//   * `_mlir_ciface_lh_matmul_inner` — a wrapper that takes pointers to a
//     `{ allocated, aligned, offset, sizes[2], strides[2] }` descriptor per
//     operand (plus a leading sret descriptor for the returned tensor).
//
// This file exports `matmul(A, B, C)` so the existing harnesses link cleanly,
// builds three row-major descriptors, and forwards through the cinterface.
// The Lighthouse pipeline allocates a fresh result buffer (bufferization
// materialises `tensor<MxN> %res` before `drop-equivalent-buffer-results`
// can fold it back into `%C`), so we copy `result.aligned + offset` back into
// the user's `C` buffer and free the temporary.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../../headers/mnk.h"

typedef struct {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} MemRef2D;

extern void _mlir_ciface_lh_matmul_inner(
    MemRef2D *result,
    MemRef2D *A,
    MemRef2D *B,
    MemRef2D *C);

void matmul(DTYPE *A, DTYPE *B, DTYPE *C) {
    MemRef2D dA  = { A, A, 0, { M, K }, { K, 1 } };
    MemRef2D dB  = { B, B, 0, { K, N }, { N, 1 } };
    MemRef2D dC  = { C, C, 0, { M, N }, { N, 1 } };
    MemRef2D res = { 0 };

    _mlir_ciface_lh_matmul_inner(&res, &dA, &dB, &dC);

    DTYPE *res_data = ((DTYPE *)res.aligned) + res.offset;
    if (res_data != C) {
        memcpy(C, res_data, sizeof(DTYPE) * (size_t)(M * N));
    }
    // Free the buffer materialised inside the kernel unless bufferization
    // chose to reuse `C` in place (in which case `allocated == C` and we
    // must not touch it: it was allocated by the harness via posix_memalign).
    if (res.allocated && res.allocated != C) {
        free(res.allocated);
    }
}
