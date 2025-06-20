# RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -o %t kernels/matmul_colmaj/main.c %s && %t | filecheck %s

.global _matmul_colmaj

///
//  Function for 4x4 matrix multiplication of 32-bit floating-point values (C += A * B).
//  Args are passed in registers x0, x1, ... and the return value is stored in x0.
//  Therefore, the result address (of matrix C) is passed as the first arg (x0).
//  The address of Matrix C is passed as x0.
//  The address of Matrix A is passed as x1.
//  The address of Matrix B is passed as x2.
///

_matmul_colmaj:

    // Load one column (4 elements) of matrix C into each of registers V8-V11
    ld1 {V8.4S, V9.4S, V10.4S, V11.4S}, [X0]

    // Load one column (4 elements) of matrix A into each of registers V0-V3
    ld1  {v0.4S, v1.4S, v2.4S, v3.4S}, [x1]

    // Load one column (4 elements) of matrix B into each of registers V4-V7
    ld1  {V4.4S, V5.4S, V6.4S, V7.4S}, [X2]

    // Multiply each column of A by the first element of each column of B
    FMLA V8.4S, V0.4S, V4.S[0]
    FMLA V9.4S, V0.4S, V5.S[0]
    FMLA V10.4S, V0.4S, V6.S[0]
    FMLA V11.4S, V0.4S, V7.S[0]

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


# CHECK:       A

# CHECK-NEXT:  [0.00, 1.00, 5.00, 5.00;
# CHECK-NEXT:   3.00, 4.00, 4.00, 4.00;
# CHECK-NEXT:   4.00, 3.00, 6.00, 3.00;
# CHECK-NEXT:   3.00, 0.00, 4.00, 2.00]

# CHECK-NEXT: B

# CHECK-NEXT:  [8.00, 3.00, 8.00, 6.00;
# CHECK-NEXT:   9.00, 4.00, 5.00, 3.00;
# CHECK-NEXT:   2.00, 7.00, 0.00, 7.00;
# CHECK-NEXT:   5.00, 5.00, 6.00, 2.00]

# CHECK-NEXT: C

# CHECK-NEXT:  [6.00, 9.00, 3.00, 3.00;
# CHECK-NEXT:   4.00, 0.00, 5.00, 0.00;
# CHECK-NEXT:   6.00, 2.00, 4.00, 0.00;
# CHECK-NEXT:   3.00, 7.00, 8.00, 1.00]

# CHECK-NEXT: A_colmaj

# CHECK-NEXT:  [0.00, 1.00, 5.00, 5.00;
# CHECK-NEXT:   3.00, 4.00, 4.00, 4.00;
# CHECK-NEXT:   4.00, 3.00, 6.00, 3.00;
# CHECK-NEXT:   3.00, 0.00, 4.00, 2.00]

# CHECK-NEXT: B_colmaj

# CHECK-NEXT:  [8.00, 3.00, 8.00, 6.00;
# CHECK-NEXT:   9.00, 4.00, 5.00, 3.00;
# CHECK-NEXT:   2.00, 7.00, 0.00, 7.00;
# CHECK-NEXT:   5.00, 5.00, 6.00, 2.00]

# CHECK-NEXT: C_colmaj

# CHECK-NEXT:  [6.00, 9.00, 3.00, 3.00;
# CHECK-NEXT:   4.00, 0.00, 5.00, 0.00;
# CHECK-NEXT:   6.00, 2.00, 4.00, 0.00;
# CHECK-NEXT:   3.00, 7.00, 8.00, 1.00]

# CHECK-NEXT: C out

# CHECK-NEXT:  [50.00, 73.00, 38.00, 51.00;
# CHECK-NEXT:   92.00, 73.00, 73.00, 66.00;
# CHECK-NEXT:   92.00, 83.00, 69.00, 81.00;
# CHECK-NEXT:   45.00, 54.00, 44.00, 51.00]

# CHECK-NEXT: C_asm out

# CHECK-NEXT:  [50.00, 73.00, 38.00, 51.00;
# CHECK-NEXT:   92.00, 73.00, 73.00, 66.00;
# CHECK-NEXT:   92.00, 83.00, 69.00, 81.00;
# CHECK-NEXT:   45.00, 54.00, 44.00, 51.00]

# CHECK-EMPTY:
# CHECK-NEXT: Test Passed: The results are equal!
