.global _matrix_mul_4x4_asm

///
//  Function for 4x4 matrix multiplication of 32-bit floating-point values (C += A * B).
//  Args are passed in registers x0, x1, ... and the return value is stored in x0.
//  Therefore, the result address (of matrix C) is passed as the first arg (x0).
//  The address of Matrix C is passed as x0.
//  The address of Matrix A is passed as x1.
//  The address of Matrix B is passed as x2.
///

_matrix_mul_4x4_asm:
    // Load one column (4 elements) of matrix A into each of registers v0-v3
    ld1  {v0.4s, v1.4s, v2.4s, v3.4s}, [x1]

    // Load one column (4 elements) of matrix B into each of registers v4-v7
    ld1  {v4.4s, v5.4s, v6.4s, v7.4s}, [x2]

    // Load one column (4 elements) of matrix C into each of registers v8-v11
    ld1 {V8.4S, V9.4S, V10.4S, V11.4S}, [X0]

    // Row 0 of result
    fmla v8.4s, v4.4s, v0.s[0]
    fmla v8.4s, v5.4s, v0.s[1]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v8.4s, v7.4s, v0.s[3]

    // Row 1 of result
    fmla v9.4s, v4.4s, v1.s[0]
    fmla v9.4s, v5.4s, v1.s[1]
    fmla v9.4s, v6.4s, v1.s[2]
    fmla v9.4s, v7.4s, v1.s[3]

    // Row 2 of result
    fmla v10.4s, v4.4s, v2.s[0]
    fmla v10.4s, v5.4s, v2.s[1]
    fmla v10.4s, v6.4s, v2.s[2]
    fmla v10.4s, v7.4s, v2.s[3]

    // Row 3 of result
    fmla v11.4s, v4.4s, v3.s[0]
    fmla v11.4s, v5.4s, v3.s[1]
    fmla v11.4s, v6.4s, v3.s[2]
    fmla v11.4s, v7.4s, v3.s[3]

    // store result matrix C
    st1  {v8.4s, v9.4s, v10.4s, v11.4s}, [x0]

    ret
