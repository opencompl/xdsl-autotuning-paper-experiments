# RUN: clang-20 -DCROWS=4 -DCCOLS=4 -DINNER=4 -o %t kernels/matmul_rowmaj/test.c %s && %t | filecheck %s

.global _matmul

///
//  Function for 4x4 matrix multiplication of 32-bit floating-point values (C += A * B).
//  Args are passed in registers x0, x1, ... and the return value is stored in x0.
//  Therefore, the result address (of matrix C) is passed as the first arg (x0).
//  The address of Matrix C is passed as x0.
//  The address of Matrix A is passed as x1.
//  The address of Matrix B is passed as x2.
///

_matmul:
    // Load one row (4 elements) of matrix C into each of registers v8-v11
    ld1 {v8.4S, v9.4S, v10.4S, v11.4S}, [X0]

    // Load one row (4 elements) of matrix A into each of registers v0-v3
    ld1  {v0.4s, v1.4s, v2.4s, v3.4s}, [x1]

    // Load one row (4 elements) of matrix B into each of registers v4-v7
    ld1  {v4.4s, v5.4s, v6.4s, v7.4s}, [x2]

    // Row 0 of result
    fmla v8.4s, v4.4s, v0.s[0]   // C[0,:] += B[0,:] * A[0,0]
    fmla v8.4s, v5.4s, v0.s[1]   // C[0,:] += B[1,:] * A[0,1]
    fmla v8.4s, v6.4s, v0.s[2]   // C[0,:] += B[2,:] * A[0,2]
    fmla v8.4s, v7.4s, v0.s[3]   // C[0,:] += B[3,:] * A[0,3]

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


# CHECK:       A

# CHECK-NEXT:  [0.00, 1.00, 5.00, 5.00],
# CHECK-NEXT:  [3.00, 4.00, 4.00, 4.00],
# CHECK-NEXT:  [4.00, 3.00, 6.00, 3.00],
# CHECK-NEXT:  [3.00, 0.00, 4.00, 2.00]

# CHECK-NEXT: B

# CHECK-NEXT:  [8.00, 3.00, 8.00, 6.00],
# CHECK-NEXT:  [9.00, 4.00, 5.00, 3.00],
# CHECK-NEXT:  [2.00, 7.00, 0.00, 7.00],
# CHECK-NEXT:  [5.00, 5.00, 6.00, 2.00]

# CHECK-NEXT: C

# CHECK-NEXT:  [6.00, 9.00, 3.00, 3.00],
# CHECK-NEXT:  [4.00, 0.00, 5.00, 0.00],
# CHECK-NEXT:  [6.00, 2.00, 4.00, 0.00],
# CHECK-NEXT:  [3.00, 7.00, 8.00, 1.00]

# CHECK-NEXT: C out

# CHECK-NEXT:  [50.00, 73.00, 38.00, 51.00],
# CHECK-NEXT:  [92.00, 73.00, 73.00, 66.00],
# CHECK-NEXT:  [92.00, 83.00, 69.00, 81.00],
# CHECK-NEXT:  [45.00, 54.00, 44.00, 51.00]


# CHECK-NEXT: C_asm out

# CHECK-NEXT:  [50.00, 73.00, 38.00, 51.00],
# CHECK-NEXT:  [92.00, 73.00, 73.00, 66.00],
# CHECK-NEXT:  [92.00, 83.00, 69.00, 81.00],
# CHECK-NEXT:  [45.00, 54.00, 44.00, 51.00]

# CHECK-EMPTY:
# CHECK-NEXT: Test Passed: The results are equal!
