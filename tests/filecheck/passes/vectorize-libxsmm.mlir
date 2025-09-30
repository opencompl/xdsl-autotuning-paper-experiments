// RUN: xdsl-opt -p vectorize-libxsmm %s | filecheck %s

func.func @matmul(
  %A: memref<3x42xf64>,
  %B: memref<42x16xf64>,
  %C: memref<3x16xf64>
) {
  linalg.matmul ins(%A, %B: memref<3x42xf64>, memref<42x16xf64>) outs(%C: memref<3x16xf64>)
  return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @matmul(%A : memref<3x42xf64>, %B : memref<42x16xf64>, %C : memref<3x16xf64>) {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %c3 = arith.constant 3 : index
// CHECK-NEXT:      %c4 = arith.constant 4 : index
// CHECK-NEXT:      %c5 = arith.constant 5 : index
// CHECK-NEXT:      %c6 = arith.constant 6 : index
// CHECK-NEXT:      %c7 = arith.constant 7 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c9 = arith.constant 9 : index
// CHECK-NEXT:      %c10 = arith.constant 10 : index
// CHECK-NEXT:      %c11 = arith.constant 11 : index
// CHECK-NEXT:      %c12 = arith.constant 12 : index
// CHECK-NEXT:      %c13 = arith.constant 13 : index
// CHECK-NEXT:      %c14 = arith.constant 14 : index
// CHECK-NEXT:      %c15 = arith.constant 15 : index
// CHECK-NEXT:      %c16 = arith.constant 16 : index
// CHECK-NEXT:      %K = arith.constant 42 : index
// CHECK-NEXT:      %N = arith.constant 16 : index
// CHECK-NEXT:      %a_leading_stride = arith.constant 42 : index
// CHECK-NEXT:      %b_leading_stride = arith.constant 16 : index
// CHECK-NEXT:      %vec_size = arith.constant 4 : index
// CHECK-NEXT:      %a_ptr = ptr_xdsl.to_ptr %A : memref<3x42xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %b_ptr = ptr_xdsl.to_ptr %B : memref<42x16xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:      %element_bytes = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:      %a_leading = arith.muli %element_bytes, %a_leading_stride : index
// CHECK-NEXT:      %c_vector_bytes = arith.muli %element_bytes, %vec_size : index
// CHECK-NEXT:      %c_row_bytes = arith.muli %element_bytes, %N : index
// CHECK-NEXT:      %b_leading_bytes = arith.muli %element_bytes, %b_leading_stride : index
// CHECK-NEXT:      %b_increment = arith.subi %b_leading_bytes, %c_row_bytes : index
// CHECK-NEXT:      %c_0_0_init = vector.load %C[%c0, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_1_0_init = vector.load %C[%c1, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_2_0_init = vector.load %C[%c2, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_0_4_init = vector.load %C[%c0, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_1_4_init = vector.load %C[%c1, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_2_4_init = vector.load %C[%c2, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_0_8_init = vector.load %C[%c0, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_1_8_init = vector.load %C[%c1, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_2_8_init = vector.load %C[%c2, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_0_12_init = vector.load %C[%c0, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_1_12_init = vector.load %C[%c1, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %c_2_12_init = vector.load %C[%c2, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      %a_ptr_out, %b_ptr_out, %c_0_0_res, %c_1_0_res, %c_2_0_res, %c_0_4_res, %c_1_4_res, %c_2_4_res, %c_0_8_res, %c_1_8_res, %c_2_8_res, %c_0_12_res, %c_1_12_res, %c_2_12_res = scf.for %k = %c0 to %K step %c1 iter_args(%a_0_k_ptr = %a_ptr, %b_k_0_ptr = %b_ptr, %c_0_0_in = %c_0_0_init, %c_1_0_in = %c_1_0_init, %c_2_0_in = %c_2_0_init, %c_0_4_in = %c_0_4_init, %c_1_4_in = %c_1_4_init, %c_2_4_in = %c_2_4_init, %c_0_8_in = %c_0_8_init, %c_1_8_in = %c_1_8_init, %c_2_8_in = %c_2_8_init, %c_0_12_in = %c_0_12_init, %c_1_12_in = %c_1_12_init, %c_2_12_in = %c_2_12_init) -> (!ptr_xdsl.ptr, !ptr_xdsl.ptr, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>) {
// CHECK-NEXT:        %a_1_k_ptr = ptr_xdsl.ptradd %a_0_k_ptr, %a_leading : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %a_2_k_ptr = ptr_xdsl.ptradd %a_1_k_ptr, %a_leading : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %a_0_k = ptr_xdsl.load %a_0_k_ptr : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %a_1_k = ptr_xdsl.load %a_1_k_ptr : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %a_2_k = ptr_xdsl.load %a_2_k_ptr : !ptr_xdsl.ptr -> f64
// CHECK-NEXT:        %a_col_vector = vector.broadcast %a_0_k : f64 to vector<4xf64>
// CHECK-NEXT:        %a_col_vector_1 = vector.broadcast %a_1_k : f64 to vector<4xf64>
// CHECK-NEXT:        %a_col_vector_2 = vector.broadcast %a_2_k : f64 to vector<4xf64>
// CHECK-NEXT:        %b_vector = ptr_xdsl.load %b_k_0_ptr : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %c_0_0_out = vector.fma %a_col_vector, %b_vector, %c_0_0_in : vector<4xf64>
// CHECK-NEXT:        %c_1_0_out = vector.fma %a_col_vector_1, %b_vector, %c_1_0_in : vector<4xf64>
// CHECK-NEXT:        %c_2_0_out = vector.fma %a_col_vector_2, %b_vector, %c_2_0_in : vector<4xf64>
// CHECK-NEXT:        %b_vector_ptr = ptr_xdsl.ptradd %b_k_0_ptr, %c_vector_bytes : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %b_vector_1 = ptr_xdsl.load %b_vector_ptr : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %c_0_4_out = vector.fma %a_col_vector, %b_vector_1, %c_0_4_in : vector<4xf64>
// CHECK-NEXT:        %c_1_4_out = vector.fma %a_col_vector_1, %b_vector_1, %c_1_4_in : vector<4xf64>
// CHECK-NEXT:        %c_2_4_out = vector.fma %a_col_vector_2, %b_vector_1, %c_2_4_in : vector<4xf64>
// CHECK-NEXT:        %b_vector_ptr_1 = ptr_xdsl.ptradd %b_vector_ptr, %c_vector_bytes : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %b_vector_2 = ptr_xdsl.load %b_vector_ptr_1 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %c_0_8_out = vector.fma %a_col_vector, %b_vector_2, %c_0_8_in : vector<4xf64>
// CHECK-NEXT:        %c_1_8_out = vector.fma %a_col_vector_1, %b_vector_2, %c_1_8_in : vector<4xf64>
// CHECK-NEXT:        %c_2_8_out = vector.fma %a_col_vector_2, %b_vector_2, %c_2_8_in : vector<4xf64>
// CHECK-NEXT:        %b_vector_ptr_2 = ptr_xdsl.ptradd %b_vector_ptr_1, %c_vector_bytes : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %b_vector_3 = ptr_xdsl.load %b_vector_ptr_2 : !ptr_xdsl.ptr -> vector<4xf64>
// CHECK-NEXT:        %c_0_12_out = vector.fma %a_col_vector, %b_vector_3, %c_0_12_in : vector<4xf64>
// CHECK-NEXT:        %c_1_12_out = vector.fma %a_col_vector_1, %b_vector_3, %c_1_12_in : vector<4xf64>
// CHECK-NEXT:        %c_2_12_out = vector.fma %a_col_vector_2, %b_vector_3, %c_2_12_in : vector<4xf64>
// CHECK-NEXT:        %b_vector_ptr_3 = ptr_xdsl.ptradd %b_vector_ptr_2, %c_vector_bytes : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %b_vector_ptr_4 = ptr_xdsl.ptradd %b_vector_ptr_3, %b_increment : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        %a_0_k_plus_one_ptr = ptr_xdsl.ptradd %a_0_k_ptr, %element_bytes : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:        scf.yield %a_0_k_plus_one_ptr, %b_vector_ptr_4, %c_0_0_out, %c_1_0_out, %c_2_0_out, %c_0_4_out, %c_1_4_out, %c_2_4_out, %c_0_8_out, %c_1_8_out, %c_2_8_out, %c_0_12_out, %c_1_12_out, %c_2_12_out : !ptr_xdsl.ptr, !ptr_xdsl.ptr, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>, vector<4xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      vector.store %c_0_0_res, %C[%c0, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_1_0_res, %C[%c1, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_2_0_res, %C[%c2, %c0] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_0_4_res, %C[%c0, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_1_4_res, %C[%c1, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_2_4_res, %C[%c2, %c4] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_0_8_res, %C[%c0, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_1_8_res, %C[%c1, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_2_8_res, %C[%c2, %c8] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_0_12_res, %C[%c0, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_1_12_res, %C[%c1, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      vector.store %c_2_12_res, %C[%c2, %c12] : memref<3x16xf64>, vector<4xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
