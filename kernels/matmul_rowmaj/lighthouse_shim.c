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
//
// It also vendors `memrefCopy`, an MLIR runtime helper that bufferization
// emits for `memref.copy` ops on non-tiny shapes. Upstream lives in
// `mlir/lib/ExecutionEngine/CRunnerUtils.cpp`; replicating it here lets the
// Lighthouse path link without dragging in `libmlir_c_runner_utils`.

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

// `UnrankedMemRefType` is the boxed-rank descriptor MLIR's lowering passes
// around the C runtime: a rank plus a pointer to a strided descriptor whose
// layout is `{ void *allocated, void *aligned, int64_t offset,
//             int64_t sizes[rank], int64_t strides[rank] }`.
typedef struct {
    int64_t rank;
    void *descriptor;
} UnrankedMemRefType;

static void memref_copy_strided(
    char *src, char *dst,
    const int64_t *sizes,
    const int64_t *src_strides,
    const int64_t *dst_strides,
    int64_t elem_size,
    int64_t rank,
    int64_t dim
) {
    if (dim == rank - 1) {
        for (int64_t i = 0; i < sizes[dim]; ++i) {
            memcpy(dst + i * dst_strides[dim] * elem_size,
                   src + i * src_strides[dim] * elem_size,
                   (size_t)elem_size);
        }
    } else {
        for (int64_t i = 0; i < sizes[dim]; ++i) {
            memref_copy_strided(
                src + i * src_strides[dim] * elem_size,
                dst + i * dst_strides[dim] * elem_size,
                sizes, src_strides, dst_strides,
                elem_size, rank, dim + 1);
        }
    }
}

void memrefCopy(int64_t elem_size,
                UnrankedMemRefType *src_arg,
                UnrankedMemRefType *dst_arg) {
    int64_t rank = src_arg->rank;

    char *src_desc = (char *)src_arg->descriptor;
    char *dst_desc = (char *)dst_arg->descriptor;
    void *src_aligned = ((void **)src_desc)[1];
    void *dst_aligned = ((void **)dst_desc)[1];

    int64_t *src_ints = (int64_t *)(src_desc + 2 * sizeof(void *));
    int64_t *dst_ints = (int64_t *)(dst_desc + 2 * sizeof(void *));
    int64_t src_offset = src_ints[0];
    int64_t dst_offset = dst_ints[0];
    const int64_t *src_sizes = &src_ints[1];
    const int64_t *src_strides = &src_ints[1 + rank];
    const int64_t *dst_strides = &dst_ints[1 + rank];

    char *src_ptr = (char *)src_aligned + src_offset * elem_size;
    char *dst_ptr = (char *)dst_aligned + dst_offset * elem_size;

    if (rank == 0) {
        memcpy(dst_ptr, src_ptr, (size_t)elem_size);
        return;
    }

    // Fast path: fully row-major contiguous on both sides.
    int contiguous = 1;
    int64_t expected = 1;
    int64_t total = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        if (src_strides[i] != expected || dst_strides[i] != expected) {
            contiguous = 0;
        }
        total *= src_sizes[i];
        expected *= src_sizes[i];
    }
    if (contiguous) {
        memcpy(dst_ptr, src_ptr, (size_t)(total * elem_size));
        return;
    }

    memref_copy_strided(src_ptr, dst_ptr, src_sizes,
                        src_strides, dst_strides,
                        elem_size, rank, 0);
}

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
