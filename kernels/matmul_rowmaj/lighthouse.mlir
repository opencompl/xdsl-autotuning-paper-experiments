// Lighthouse-friendly variant of `kernels/matmul_rowmaj/mlir.mlir`.
//
// The shared template's `func.func @matmul` discards the matmul result with a
// bare `return`. That works for the existing flow because the early
// one-shot-bufferize step turns `outs(%C)` into an in-place write before any
// DCE runs. The Lighthouse pipeline interleaves transform-dialect scheduling
// with bufferization, so the tensor result must be live; otherwise the early
// canonicalization passes eliminate the matmul entirely and we end up with an
// empty function body. Returning `%res` keeps it alive without changing the
// observable ABI: bufferization's `drop-equivalent-buffer-results` folds the
// returned tensor back into the destination memref.
//
// The exported symbol is intentionally `@lh_matmul_inner`, not `@matmul`: the
// `llvm.emit_c_interface` lowering uses the MLIR memref calling convention
// (each `memref<MxNxT>` becomes 7 split arguments, plus a sret descriptor
// pointer for the result). That ABI is incompatible with the
// `void matmul(A, B, C)` signature the C `test.c` / `time.c` harnesses call,
// so we keep that name available for the thin shim in
// `kernels/matmul_rowmaj/lighthouse_shim.c`, which builds the memref
// descriptors and forwards through `_mlir_ciface_lh_matmul_inner`.
func.func public @lh_matmul_inner(
    %arg0: tensor<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg1: tensor<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg2: tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias}
) -> tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}> {
    %res = linalg.matmul ins(%arg0, %arg1 : tensor<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}>, tensor<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}>) outs(%arg2 : tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>) -> tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>

    return %res : tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>
}
