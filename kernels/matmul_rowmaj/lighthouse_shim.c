// C shim that adapts the Lighthouse-emitted memref calling convention to the
// plain `void matmul(A, B, C)` signature `test.c` / `time.c` expect.
//
// Torch ingress + `convert_function_results` lowers `main` to take the output
// buffer as its first memref argument, followed by the two inputs:
//   `_mlir_ciface_main(result, A, B)`
//
// The harness computes `C += A*B`, while the compiled kernel overwrites its
// output buffer with `A@B`. We save the incoming `C`, run the kernel, then add
// the saved values back.
//
// `memrefCopy` is vendored from MLIR's CRunnerUtils for any bufferization
// copies the pipeline may emit.

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

extern void _mlir_ciface_main(
    MemRef2D *result,
    MemRef2D *A,
    MemRef2D *B);

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
    DTYPE saved[M * N];
    memcpy(saved, C, sizeof(saved));

    MemRef2D dC = { C, C, 0, { M, N }, { N, 1 } };
    MemRef2D dA = { A, A, 0, { M, K }, { K, 1 } };
    MemRef2D dB = { B, B, 0, { K, N }, { N, 1 } };

    _mlir_ciface_main(&dC, &dA, &dB);

    for (int i = 0; i < M * N; ++i) {
        C[i] += saved[i];
    }
}
