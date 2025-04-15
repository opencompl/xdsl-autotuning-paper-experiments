func.func @matrix_mul_4x4_asm(%C: memref<4x4xf32>, %A: memref<4x4xf32>, %B: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  // Load A Rows
  %A0 = vector.load %A[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  %A1 = vector.load %A[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
  %A2 = vector.load %A[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
  %A3 = vector.load %A[%c3, %c0] : memref<4x4xf32>, vector<4xf32>

  // Load B Rows
  %B0 = vector.load %B[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  %B1 = vector.load %B[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
  %B2 = vector.load %B[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
  %B3 = vector.load %B[%c3, %c0] : memref<4x4xf32>, vector<4xf32>

  // Load C Rows
  %C0 = vector.load %C[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  %C1 = vector.load %C[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
  %C2 = vector.load %C[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
  %C3 = vector.load %C[%c3, %c0] : memref<4x4xf32>, vector<4xf32>

  // Compute the row 0 of result
  %a00 = vector.extract %A0[0]: f32 from vector<4xf32>
  %A00 = vector.splat %a00 : vector<4xf32>
  %C00 = vector.fma %B0, %A00, %C0: vector<4xf32>
  %a01 = vector.extract %A0[1]: f32 from vector<4xf32>
  %A01 = vector.splat %a01 : vector<4xf32>
  %C01 = vector.fma %B1, %A01, %C00: vector<4xf32>
  %a02 = vector.extract %A0[2]: f32 from vector<4xf32>
  %A02 = vector.splat %a02 : vector<4xf32>
  %C02 = vector.fma %B2, %A02, %C01: vector<4xf32>
  %a03 = vector.extract %A0[3]: f32 from vector<4xf32>
  %A03 = vector.splat %a03 : vector<4xf32>
  %C03 = vector.fma %B3, %A03, %C02: vector<4xf32>

  // Compute the row 1 of result
  %a10 = vector.extract %A1[0]: f32 from vector<4xf32>
  %A10 = vector.splat %a10 : vector<4xf32>
  %C10 = vector.fma %B0, %A10, %C1: vector<4xf32>
  %a11 = vector.extract %A1[1]: f32 from vector<4xf32>
  %A11 = vector.splat %a11 : vector<4xf32>
  %C11 = vector.fma %B1, %A11, %C10: vector<4xf32>
  %a12 = vector.extract %A1[2]: f32 from vector<4xf32>
  %A12 = vector.splat %a12 : vector<4xf32>
  %C12 = vector.fma %B2, %A12, %C11: vector<4xf32>
  %a13 = vector.extract %A1[3]: f32 from vector<4xf32>
  %A13 = vector.splat %a13 : vector<4xf32>
  %C13 = vector.fma %B3, %A13, %C12: vector<4xf32>

  // Compute the row 2 of result
  %a20 = vector.extract %A2[0]: f32 from vector<4xf32>
  %A20 = vector.splat %a20 : vector<4xf32>
  %C20 = vector.fma %B0, %A20, %C2: vector<4xf32>
  %a21 = vector.extract %A2[1]: f32 from vector<4xf32>
  %A21 = vector.splat %a21 : vector<4xf32>
  %C21 = vector.fma %B1, %A21, %C20: vector<4xf32>
  %a22 = vector.extract %A2[2]: f32 from vector<4xf32>
  %A22 = vector.splat %a22 : vector<4xf32>
  %C22 = vector.fma %B2, %A22, %C21: vector<4xf32>
  %a23 = vector.extract %A2[3]: f32 from vector<4xf32>
  %A23 = vector.splat %a23 : vector<4xf32>
  %C23 = vector.fma %B3, %A23, %C22: vector<4xf32>

  // Compute the row 3 of result
  %a30 = vector.extract %A3[0]: f32 from vector<4xf32>
  %A30 = vector.splat %a30 : vector<4xf32>
  %C30 = vector.fma %B0, %A30, %C3: vector<4xf32>
  %a31 = vector.extract %A3[1]: f32 from vector<4xf32>
  %A31 = vector.splat %a31 : vector<4xf32>
  %C31 = vector.fma %B1, %A31, %C30: vector<4xf32>
  %a32 = vector.extract %A3[2]: f32 from vector<4xf32>
  %A32 = vector.splat %a32 : vector<4xf32>
  %C32 = vector.fma %B2, %A32, %C31: vector<4xf32>
  %a33 = vector.extract %A3[3]: f32 from vector<4xf32>
  %A33 = vector.splat %a33 : vector<4xf32>
  %C33 = vector.fma %B3, %A33, %C32: vector<4xf32>

  // Store result matrix
  vector.store %C03, %C[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
  vector.store %C13, %C[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
  vector.store %C23, %C[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
  vector.store %C33, %C[%c3, %c0] : memref<4x4xf32>, vector<4xf32>

  return
}
