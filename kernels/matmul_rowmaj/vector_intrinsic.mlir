func.func @matmul(
    %arg0: memref<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg1: memref<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg2: memref<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}> {llvm.noalias}
) {
    %c0 = arith.constant 0 : index
    %A = vector.load %arg1[%c0, %c0] : memref<{{wildcards.m}}x{{wildcards.k}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.k|int}}x{{wildcards.dtype}}>
    %B = vector.load %arg2[%c0, %c0] : memref<{{wildcards.k}}x{{wildcards.n}}x{{wildcards.dtype}}>, vector<{{wildcards.k|int * wildcards.n|int}}x{{wildcards.dtype}}>
    %C_in = vector.load %arg0[%c0, %c0] : memref<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>

    %C_acc = vector.matrix_multiply %B, %A {
        lhs_rows = {{wildcards.n}}: i32,
        lhs_columns = {{wildcards.k}}: i32,
        rhs_columns = {{wildcards.m}}: i32
    } : (vector<{{wildcards.k|int * wildcards.n|int}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.k|int}}x{{wildcards.dtype}}>) -> vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>

    %C_out = arith.addf %C_in, %C_acc : vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>
    vector.store %C_out, %arg0[%c0, %c0] : memref<{{wildcards.m}}x{{wildcards.n}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>
    return
}
