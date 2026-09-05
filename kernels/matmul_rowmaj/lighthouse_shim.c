// Adapts the memref calling convention of the Lighthouse-compiled kernel to the
// plain `void matmul(A, B, C)` signature that `time.c` and `test.c` expect.
//
// `scripts/lighthouse_codegen.py` turns the payload's returned tensor into a
// leading destination argument, so the compiled entry point is
//
//     lighthouse_matmul(destination, A, B, C)   with C the accumulator
//
// and `_mlir_ciface_*` (emitted by `llvm.emit_c_interface`) takes one memref
// descriptor pointer per argument. Passing `C` as both destination and
// accumulator gives the `C += A * B` the harness expects, without the kernel
// allocating or the shim copying anything: the payload reads a `C` element
// before writing that same element back, so the two roles never conflict.

#include <stdint.h>

#include "../../headers/mnk.h"

typedef struct {
    DTYPE *allocated;
    DTYPE *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} MemRef2D;

extern void _mlir_ciface_lighthouse_matmul(
    MemRef2D *destination, MemRef2D *A, MemRef2D *B, MemRef2D *C);

void matmul(DTYPE *A, DTYPE *B, DTYPE *C) {
    MemRef2D dA = {A, A, 0, {M, K}, {K, 1}};
    MemRef2D dB = {B, B, 0, {K, N}, {N, 1}};
    MemRef2D dC = {C, C, 0, {M, N}, {N, 1}};

    _mlir_ciface_lighthouse_matmul(&dC, &dA, &dB, &dC);
}
