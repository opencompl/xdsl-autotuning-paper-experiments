func.func @matmul_colmaj(
    %arg0: memref<{{N}}x{{M}}xf32> {llvm.noalias},
    %arg1: memref<{{K}}x{{M}}xf32> {llvm.noalias},
    %arg2: memref<{{N}}x{{K}}xf32> {llvm.noalias}
) {
    %c0 = arith.constant 0 : index
    %A = vector.load %arg1[%c0, %c0] : memref<{{K}}x{{M}}xf32>, vector<{{MK}}xf32>
    %B = vector.load %arg2[%c0, %c0] : memref<{{N}}x{{K}}xf32>, vector<{{KN}}xf32>
    %C_in = vector.load %arg0[%c0, %c0] : memref<{{N}}x{{M}}xf32>, vector<{{MN}}xf32>

    %C_acc = vector.matrix_multiply %A, %B {
        lhs_rows = {{M}}: i32,
        lhs_columns = {{K}}: i32,
        rhs_columns = {{N}}: i32
    } : (vector<{{MK}}xf32>, vector<{{KN}}xf32>) -> vector<{{MN}}xf32>

    %C_out = arith.addf %C_in, %C_acc : vector<{{MN}}xf32>
    vector.store %C_out, %arg0[%c0, %c0] : memref<{{N}}x{{M}}xf32>, vector<{{MN}}xf32>
    return
}
