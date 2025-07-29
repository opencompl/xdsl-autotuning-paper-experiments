func.func @matmul_colmaj(
    %arg0: memref<{{wildcards.n}}x{{wildcards.m}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg1: memref<{{wildcards.k}}x{{wildcards.m}}x{{wildcards.dtype}}> {llvm.noalias},
    %arg2: memref<{{wildcards.n}}x{{wildcards.k}}x{{wildcards.dtype}}> {llvm.noalias}
) {
    %c0 = arith.constant 0 : index
    %A = vector.load %arg1[%c0, %c0] : memref<{{wildcards.k}}x{{wildcards.m}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.k|int}}x{{wildcards.dtype}}>
    %B = vector.load %arg2[%c0, %c0] : memref<{{wildcards.n}}x{{wildcards.k}}x{{wildcards.dtype}}>, vector<{{wildcards.k|int * wildcards.n|int}}x{{wildcards.dtype}}>
    %C_in = vector.load %arg0[%c0, %c0] : memref<{{wildcards.n}}x{{wildcards.m}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>

    %C_acc = vector.matrix_multiply %A, %B {
        lhs_rows = {{wildcards.m}}: i32,
        lhs_columns = {{wildcards.k}}: i32,
        rhs_columns = {{wildcards.n}}: i32
    } : (vector<{{wildcards.m|int * wildcards.k|int}}x{{wildcards.dtype}}>, vector<{{wildcards.k|int * wildcards.n|int}}x{{wildcards.dtype}}>) -> vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>

    %C_out = arith.addf %C_in, %C_acc : vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>
    vector.store %C_out, %arg0[%c0, %c0] : memref<{{wildcards.n}}x{{wildcards.m}}x{{wildcards.dtype}}>, vector<{{wildcards.m|int * wildcards.n|int}}x{{wildcards.dtype}}>
    return
}
