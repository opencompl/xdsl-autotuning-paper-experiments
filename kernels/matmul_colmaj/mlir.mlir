func.func @matmul_colmaj(
    %arg0: tensor<{{wildcards.n}}x{{wildcards.m}}xf32> {llvm.noalias},
    %arg1: tensor<{{wildcards.k}}x{{wildcards.m}}xf32> {llvm.noalias},
    %arg2: tensor<{{wildcards.n}}x{{wildcards.k}}xf32> {llvm.noalias}
) {
    %res = linalg.matmul ins(%arg2, %arg1 : tensor<{{wildcards.n}}x{{wildcards.k}}xf32>, tensor<{{wildcards.k}}x{{wildcards.m}}xf32>) outs(%arg0 : tensor<{{wildcards.n}}x{{wildcards.m}}xf32>) -> tensor<{{wildcards.n}}x{{wildcards.m}}xf32>

    return
}
