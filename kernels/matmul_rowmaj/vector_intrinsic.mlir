func.func @matmul(
    %arg0: memref<{{M}}x{{N}}xf32> {llvm.noalias},
    %arg1: memref<{{M}}x{{K}}xf32> {llvm.noalias},
    %arg2: memref<{{K}}x{{N}}xf32> {llvm.noalias}
) {
    %c0 = arith.constant 0 : index
    %A = vector.load %arg1[%c0, %c0] : memref<{{M}}x{{K}}xf32>, vector<16xf32>
    %B = vector.load %arg2[%c0, %c0] : memref<{{K}}x{{N}}xf32>, vector<16xf32>
    %C = vector.matrix_multiply %A, %B {
        lhs_rows = {{M}}: i32,
        lhs_columns = {{K}}: i32,
        rhs_columns = {{N}}: i32
    } : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vector.store %C, %arg0[%c0, %c0] : memref<{{M}}x{{N}}xf32>, vector<16xf32>
    return
}
