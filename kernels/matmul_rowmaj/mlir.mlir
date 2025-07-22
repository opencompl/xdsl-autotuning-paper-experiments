func.func @matmul(
    %arg0: tensor<{{M}}x{{N}}xf32> {llvm.noalias},
    %arg1: tensor<{{M}}x{{K}}xf32> {llvm.noalias},
    %arg2: tensor<{{K}}x{{N}}xf32> {llvm.noalias}
) {
    %res = linalg.matmul ins(%arg1, %arg2 : tensor<{{M}}x{{K}}xf32>, tensor<{{K}}x{{N}}xf32>) outs(%arg0 : tensor<{{M}}x{{N}}xf32>) -> tensor<{{M}}x{{N}}xf32>

    return
}
