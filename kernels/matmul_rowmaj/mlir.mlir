func.func @matmul(
    %arg0: tensor<{{wildcards.m}}x{{wildcards.n}}xf32> {llvm.noalias},
    %arg1: tensor<{{wildcards.m}}x{{wildcards.k}}xf32> {llvm.noalias},
    %arg2: tensor<{{wildcards.k}}x{{wildcards.n}}xf32> {llvm.noalias}
) {
    %res = linalg.matmul ins(%arg1, %arg2 : tensor<{{wildcards.m}}x{{wildcards.k}}xf32>, tensor<{{wildcards.k}}x{{wildcards.n}}xf32>) outs(%arg0 : tensor<{{wildcards.m}}x{{wildcards.n}}xf32>) -> tensor<{{wildcards.m}}x{{wildcards.n}}xf32>

    return
}
