  func.func @myfun(%arg0: memref<4x4xf32> {llvm.noalias}, %arg1: memref<4x4xf32> {llvm.noalias}, %arg2: memref<4x4xf32> {llvm.noalias}) {
    linalg.matmul ins(%arg0, %arg1 : memref<4x4xf32>, memref<4x4xf32>) outs(%arg2 : memref<4x4xf32>)
    return
  }
