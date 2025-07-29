func.func @matmul_colmaj(
    %arg0: memref<{{wildcards.n}}x{{wildcards.m}}xf32> {llvm.noalias},
    %arg1: memref<{{wildcards.k}}x{{wildcards.m}}xf32> {llvm.noalias},
    %arg2: memref<{{wildcards.n}}x{{wildcards.k}}xf32> {llvm.noalias}
) {
    %c0 = arith.constant 0 : index
    %A = vector.load %arg1[%c0, %c0] : memref<{{wildcards.k}}x{{wildcards.m}}xf32>, vector<{{wildcards.m|int * wildcards.k|int}}xf32>
    %B = vector.load %arg2[%c0, %c0] : memref<{{wildcards.n}}x{{wildcards.k}}xf32>, vector<{{wildcards.k|int * wildcards.n|int}}xf32>
    %C_in = vector.load %arg0[%c0, %c0] : memref<{{wildcards.n}}x{{wildcards.m}}xf32>, vector<{{wildcards.m|int * wildcards.n|int}}xf32>

    %C_acc = vector.matrix_multiply %A, %B {
        lhs_rows = {{wildcards.m}}: i32,
        lhs_columns = {{wildcards.k}}: i32,
        rhs_columns = {{wildcards.n}}: i32
    } : (vector<{{wildcards.m|int * wildcards.k|int}}xf32>, vector<{{wildcards.k|int * wildcards.n|int}}xf32>) -> vector<{{wildcards.m|int * wildcards.n|int}}xf32>

    %C_out = arith.addf %C_in, %C_acc : vector<{{wildcards.m|int * wildcards.n|int}}xf32>
    vector.store %C_out, %arg0[%c0, %c0] : memref<{{wildcards.n}}x{{wildcards.m}}xf32>, vector<{{wildcards.m|int * wildcards.n|int}}xf32>
    return
}
