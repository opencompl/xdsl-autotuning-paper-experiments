func.func @matmul_colmaj(
    %arg0: tensor<{{N}}x{{M}}xf32> {llvm.noalias},
    %arg1: tensor<{{K}}x{{M}}xf32> {llvm.noalias},
    %arg2: tensor<{{N}}x{{K}}xf32> {llvm.noalias}
) {
    %res = linalg.matmul ins(%arg2, %arg1 : tensor<{{N}}x{{K}}xf32>, tensor<{{K}}x{{M}}xf32>) outs(%arg0 : tensor<{{N}}x{{M}}xf32>) -> tensor<{{N}}x{{M}}xf32>

    return
}
