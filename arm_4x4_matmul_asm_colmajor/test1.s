.global _matrix_mul_4x4_asm

///
//  Function for 4x4 matrix multiplication of 32-bit floating-point values (A * B = C).
//  Args are passed in registers x0, x1, ... and the return value is stored in x0.
//  Therefore, the result address (of matrix C) is passed as the first arg (x0).
//  The address of Matrix A is passed as x1. 
//  The address of Matrix B is passed as x2.
///

_matrix_mul_4x4_asm:

    // Load one column (4 elements) of matrix A into each of registers V0-V3
    ld1  {v0.4S, v1.4S, v2.4S, v3.4S}, [x1]

    // Load one column (4 elements) of matrix B into each of registers V4-V7
    LD1  {V4.4S, V5.4S, V6.4S, V7.4S}, [X2]

    // Multiply each column of A by the first element of each column of B
    // FMUL is used to initialize the accumulators, FMULA is used to update them.
    FMUL V8.4S, V0.4S, V4.S[0]
    FMUL V9.4S, V0.4S, V5.S[0]
    FMUL V10.4S, V0.4S, V6.S[0]
    FMUL V11.4S, V0.4S, V7.S[0]

    // Multiply the rest of the columns of A by the remaining elements in each column of B
    FMLA V8.4S, V1.4S, V4.S[1]
    FMLA V9.4S, V1.4S, V5.S[1]
    FMLA V10.4S, V1.4S, V6.S[1]
    FMLA V11.4S, V1.4S, V7.S[1]

    FMLA V8.4S, V2.4S,  V4.S[2]
    FMLA V9.4S, V2.4S,  V5.S[2]
    FMLA V10.4S, V2.4S, V6.S[2]
    FMLA V11.4S, V2.4S, V7.S[2]

    FMLA V8.4S, V3.4S, V4.S[3]
    FMLA V9.4S, V3.4S, V5.S[3]
    FMLA V10.4S, V3.4S, V6.S[3]
    FMLA V11.4S, V3.4S, V7.S[3]

    ST1  {V8.4S, V9.4S, V10.4S, V11.4S}, [X0]
    RET