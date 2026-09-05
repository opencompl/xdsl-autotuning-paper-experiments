// Payload for the `lighthouse` variant: the same `linalg.matmul` as `mlir.mlir`,
// with two adjustments for a pipeline that schedules at the tensor level and
// bufferizes at the very end.
//
//  * The result is returned instead of dropped. `mlir.mlir` relies on the
//    bufferization that immediately follows it to turn the write into `%arg2`
//    into a side effect; here the matmul would be dead code and get folded away
//    before it ever reaches a schedule.
//  * `%arg2` is not marked `llvm.noalias`. Lighthouse's pipeline hands the
//    result back in its own buffer, so `scripts/lighthouse_codegen.py` converts
//    the result into a destination argument and the shim passes `C` both as the
//    destination and as the accumulator, exactly like `C += A * B` in the other
//    variants.
func.func public @matmul(
    %arg0: tensor<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg1: tensor<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg2: tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>
) -> tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}> {
    %res = linalg.matmul ins(%arg0, %arg1 : tensor<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}>, tensor<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}>) outs(%arg2 : tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>) -> tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>

    return %res : tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>
}
