module {
  func.func @matrix_mul_4x4_asm(%result: memref<4x4xf32>, %A: memref<4x4xf32>, %B: memref<4x4xf32>) {
    // Cast memrefs to vectors for easier handling
    %A_flat = memref.cast %A : memref<4x4xf32> to memref<?xf32>
    %B_flat = memref.cast %B : memref<4x4xf32> to memref<?xf32>
    %C_flat = memref.cast %result : memref<4x4xf32> to memref<?xf32>

    // Load all 4 rows of matrix A into vectors (v0-v3 in the assembly)
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %a_row0 = vector.load %A_flat[%c0] : memref<?xf32>, vector<4xf32>
    %c4 = arith.constant 4 : index
    %a_row1 = vector.load %A_flat[%c4] : memref<?xf32>, vector<4xf32>
    %c8 = arith.constant 8 : index
    %a_row2 = vector.load %A_flat[%c8] : memref<?xf32>, vector<4xf32>
    %c12 = arith.constant 12 : index
    %a_row3 = vector.load %A_flat[%c12] : memref<?xf32>, vector<4xf32>

    // Load all 4 columns of matrix B into vectors (v4-v7 in the assembly)
    // Since B is row-major but we need columns, we'll extract them manually
    %b_row0 = vector.load %B_flat[%c0] : memref<?xf32>, vector<4xf32>
    %b_row1 = vector.load %B_flat[%c4] : memref<?xf32>, vector<4xf32>
    %b_row2 = vector.load %B_flat[%c8] : memref<?xf32>, vector<4xf32>
    %b_row3 = vector.load %B_flat[%c12] : memref<?xf32>, vector<4xf32>

    // Extract columns from B's rows
    %b_col0 = vector.shuffle %b_row0, %b_row1 [0, 4] : vector<4xf32>, vector<4xf32>
    %b_col0_partial = vector.shuffle %b_row2, %b_row3 [0, 4] : vector<4xf32>, vector<4xf32>
    %b_col0_full = vector.shuffle %b_col0, %b_col0_partial [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>

    %b_col1 = vector.shuffle %b_row0, %b_row1 [1, 5] : vector<4xf32>, vector<4xf32>
    %b_col1_partial = vector.shuffle %b_row2, %b_row3 [1, 5] : vector<4xf32>, vector<4xf32>
    %b_col1_full = vector.shuffle %b_col1, %b_col1_partial [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>

    %b_col2 = vector.shuffle %b_row0, %b_row1 [2, 6] : vector<4xf32>, vector<4xf32>
    %b_col2_partial = vector.shuffle %b_row2, %b_row3 [2, 6] : vector<4xf32>, vector<4xf32>
    %b_col2_full = vector.shuffle %b_col2, %b_col2_partial [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>

    %b_col3 = vector.shuffle %b_row0, %b_row1 [3, 7] : vector<4xf32>, vector<4xf32>
    %b_col3_partial = vector.shuffle %b_row2, %b_row3 [3, 7] : vector<4xf32>, vector<4xf32>
    %b_col3_full = vector.shuffle %b_col3, %b_col3_partial [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>

    // Compute row 0 of result (v8 in the assembly)
    %a0_0 = vector.extract %a_row0[0] : vector<4xf32>
    %a0_1 = vector.extract %a_row0[1] : vector<4xf32>
    %a0_2 = vector.extract %a_row0[2] : vector<4xf32>
    %a0_3 = vector.extract %a_row0[3] : vector<4xf32>

    %mul0_0 = vector.splat %a0_0 : vector<4xf32>
    %mul0_1 = vector.splat %a0_1 : vector<4xf32>
    %mul0_2 = vector.splat %a0_2 : vector<4xf32>
    %mul0_3 = vector.splat %a0_3 : vector<4xf32>

    %prod0_0 = arith.mulf %mul0_0, %b_col0_full : vector<4xf32>
    %acc0_1 = arith.mulf %mul0_1, %b_col1_full : vector<4xf32>
    %acc0_2 = arith.mulf %mul0_2, %b_col2_full : vector<4xf32>
    %acc0_3 = arith.mulf %mul0_3, %b_col3_full : vector<4xf32>

    %res0_1 = arith.addf %prod0_0, %acc0_1 : vector<4xf32>
    %res0_2 = arith.addf %res0_1, %acc0_2 : vector<4xf32>
    %res0 = arith.addf %res0_2, %acc0_3 : vector<4xf32>

    // Compute row 1 of result (v9 in the assembly)
    %a1_0 = vector.extract %a_row1[0] : vector<4xf32>
    %a1_1 = vector.extract %a_row1[1] : vector<4xf32>
    %a1_2 = vector.extract %a_row1[2] : vector<4xf32>
    %a1_3 = vector.extract %a_row1[3] : vector<4xf32>

    %mul1_0 = vector.splat %a1_0 : vector<4xf32>
    %mul1_1 = vector.splat %a1_1 : vector<4xf32>
    %mul1_2 = vector.splat %a1_2 : vector<4xf32>
    %mul1_3 = vector.splat %a1_3 : vector<4xf32>

    %prod1_0 = arith.mulf %mul1_0, %b_col0_full : vector<4xf32>
    %acc1_1 = arith.mulf %mul1_1, %b_col1_full : vector<4xf32>
    %acc1_2 = arith.mulf %mul1_2, %b_col2_full : vector<4xf32>
    %acc1_3 = arith.mulf %mul1_3, %b_col3_full : vector<4xf32>

    %res1_1 = arith.addf %prod1_0, %acc1_1 : vector<4xf32>
    %res1_2 = arith.addf %res1_1, %acc1_2 : vector<4xf32>
    %res1 = arith.addf %res1_2, %acc1_3 : vector<4xf32>

    // Compute row 2 of result (v10 in the assembly)
    %a2_0 = vector.extract %a_row2[0] : vector<4xf32>
    %a2_1 = vector.extract %a_row2[1] : vector<4xf32>
    %a2_2 = vector.extract %a_row2[2] : vector<4xf32>
    %a2_3 = vector.extract %a_row2[3] : vector<4xf32>

    %mul2_0 = vector.splat %a2_0 : vector<4xf32>
    %mul2_1 = vector.splat %a2_1 : vector<4xf32>
    %mul2_2 = vector.splat %a2_2 : vector<4xf32>
    %mul2_3 = vector.splat %a2_3 : vector<4xf32>

    %prod2_0 = arith.mulf %mul2_0, %b_col0_full : vector<4xf32>
    %acc2_1 = arith.mulf %mul2_1, %b_col1_full : vector<4xf32>
    %acc2_2 = arith.mulf %mul2_2, %b_col2_full : vector<4xf32>
    %acc2_3 = arith.mulf %mul2_3, %b_col3_full : vector<4xf32>

    %res2_1 = arith.addf %prod2_0, %acc2_1 : vector<4xf32>
    %res2_2 = arith.addf %res2_1, %acc2_2 : vector<4xf32>
    %res2 = arith.addf %res2_2, %acc2_3 : vector<4xf32>

    // Compute row 3 of result (v11 in the assembly)
    %a3_0 = vector.extract %a_row3[0] : vector<4xf32>
    %a3_1 = vector.extract %a_row3[1] : vector<4xf32>
    %a3_2 = vector.extract %a_row3[2] : vector<4xf32>
    %a3_3 = vector.extract %a_row3[3] : vector<4xf32>

    %mul3_0 = vector.splat %a3_0 : vector<4xf32>
    %mul3_1 = vector.splat %a3_1 : vector<4xf32>
    %mul3_2 = vector.splat %a3_2 : vector<4xf32>
    %mul3_3 = vector.splat %a3_3 : vector<4xf32>

    %prod3_0 = arith.mulf %mul3_0, %b_col0_full : vector<4xf32>
    %acc3_1 = arith.mulf %mul3_1, %b_col1_full : vector<4xf32>
    %acc3_2 = arith.mulf %mul3_2, %b_col2_full : vector<4xf32>
    %acc3_3 = arith.mulf %mul3_3, %b_col3_full : vector<4xf32>

    %res3_1 = arith.addf %prod3_0, %acc3_1 : vector<4xf32>
    %res3_2 = arith.addf %res3_1, %acc3_2 : vector<4xf32>
    %res3 = arith.addf %res3_2, %acc3_3 : vector<4xf32>

    // Store result matrix
    vector.store %res0, %C_flat[%c0] : memref<?xf32>, vector<4xf32>
    vector.store %res1, %C_flat[%c4] : memref<?xf32>, vector<4xf32>
    vector.store %res2, %C_flat[%c8] : memref<?xf32>, vector<4xf32>
    vector.store %res3, %C_flat[%c12] : memref<?xf32>, vector<4xf32>

    return
  }
}
