func.func @myfun(
    %arg0: memref<{{M}}x{{N}}xf32> {llvm.noalias},
    %arg1: memref<{{M}}x{{K}}xf32> {llvm.noalias},
    %arg2: memref<{{K}}x{{N}}xf32> {llvm.noalias}
) {
    linalg.matmul ins(%arg1, %arg2 : memref<{{M}}x{{K}}xf32>, memref<{{K}}x{{N}}xf32>) outs(%arg0 : memref<{{M}}x{{N}}xf32>)
    return
}
