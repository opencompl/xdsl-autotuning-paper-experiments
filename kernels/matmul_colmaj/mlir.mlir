func.func @matmul_colmaj(
    %arg0: tensor<{{wildcards.k}}x{{wildcards.m}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg1: tensor<{{wildcards.n}}x{{wildcards.k}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg2: tensor<{{wildcards.n}}x{{wildcards.m}}x{{wildcards.dtype}}> {llvm.noalias}
) {
    %res = linalg.matmul ins(%arg1, %arg0 : tensor<{{wildcards.n}}x{{wildcards.k}}x{{wildcards.dtype}}>, tensor<{{wildcards.k}}x{{wildcards.m}}x{{wildcards.dtype}}>) outs(%arg2 : tensor<{{wildcards.n}}x{{wildcards.m}}x{{wildcards.dtype}}>) -> tensor<{{wildcards.n}}x{{wildcards.m}}x{{wildcards.dtype}}>

    return
}
