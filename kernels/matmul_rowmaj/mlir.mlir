func.func @matmul(
    %arg0: tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg1: tensor<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg2: tensor<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias}
) {
    %res = linalg.matmul ins(%arg1, %arg2 : tensor<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}>, tensor<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}>) outs(%arg0 : tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>) -> tensor<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>

    return
}
