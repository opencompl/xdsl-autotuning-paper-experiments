
func.func @matmul(
  %A: memref<3x42xf64>,
  %B: memref<42x16xf64>,
  %C: memref<3x16xf64>
) {
  linalg.matmul ins(%A, %B: memref<3x42xf64>, memref<42x16xf64>) outs(%C: memref<3x16xf64>)
  return
}
