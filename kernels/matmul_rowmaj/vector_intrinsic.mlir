func.func @matmul(
    %arg0: memref<{{wildcards.m}}x{{wildcards.n}}xf32> {llvm.noalias},
    %arg1: memref<{{wildcards.m}}x{{wildcards.k}}xf32> {llvm.noalias},
    %arg2: memref<{{wildcards.k}}x{{wildcards.n}}xf32> {llvm.noalias}
) {
    %c0 = arith.constant 0 : index
    %A = vector.load %arg1[%c0, %c0] : memref<{{wildcards.m}}x{{wildcards.k}}xf32>, vector<{{wildcards.m|int * wildcards.k|int}}xf32>
    %B = vector.load %arg2[%c0, %c0] : memref<{{wildcards.k}}x{{wildcards.n}}xf32>, vector<{{wildcards.k|int * wildcards.n|int}}xf32>
    %C_in = vector.load %arg0[%c0, %c0] : memref<{{wildcards.m}}x{{wildcards.n}}xf32>, vector<{{wildcards.m|int * wildcards.n|int}}xf32>

    %C_acc = vector.matrix_multiply %B, %A {
        lhs_rows = {{wildcards.n}}: i32,
        lhs_columns = {{wildcards.k}}: i32,
        rhs_columns = {{wildcards.m}}: i32
    } : (vector<{{wildcards.k|int * wildcards.n|int}}xf32>, vector<{{wildcards.m|int * wildcards.k|int}}xf32>) -> vector<{{wildcards.m|int * wildcards.n|int}}xf32>

    %C_out = arith.addf %C_in, %C_acc : vector<{{wildcards.m|int * wildcards.n|int}}xf32>
    vector.store %C_out, %arg0[%c0, %c0] : memref<{{wildcards.m}}x{{wildcards.n}}xf32>, vector<{{wildcards.m|int * wildcards.n|int}}xf32>
    return
}
