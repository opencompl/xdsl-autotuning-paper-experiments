# RUN: clang -o %t arm_4x4_matmul_asm_colmajor/main.c arm_4x4_matmul_asm_colmajor/test1.s && %t | filecheck %s

# A

# CHECK:       [0.00, 1.00, 5.00, 5.00;
# CHECK-NEXT:   3.00, 4.00, 4.00, 4.00;
# CHECK-NEXT:   4.00, 3.00, 6.00, 3.00;
# CHECK-NEXT:   3.00, 0.00, 4.00, 2.00]

# B

# CHECK-NEXT:  [8.00, 3.00, 8.00, 6.00;
# CHECK-NEXT:   9.00, 4.00, 5.00, 3.00;
# CHECK-NEXT:   2.00, 7.00, 0.00, 7.00;
# CHECK-NEXT:   5.00, 5.00, 6.00, 2.00]

# C

# CHECK-NEXT:  [6.00, 9.00, 3.00, 3.00;
# CHECK-NEXT:   4.00, 0.00, 5.00, 0.00;
# CHECK-NEXT:   6.00, 2.00, 4.00, 0.00;
# CHECK-NEXT:   3.00, 7.00, 8.00, 1.00]

# A_colmaj

# CHECK-NEXT:  [0.00, 3.00, 4.00, 3.00;
# CHECK-NEXT:   1.00, 4.00, 3.00, 0.00;
# CHECK-NEXT:   5.00, 4.00, 6.00, 4.00;
# CHECK-NEXT:   5.00, 4.00, 3.00, 2.00]

# A_colmaj

# CHECK-NEXT:  [8.00, 9.00, 2.00, 5.00;
# CHECK-NEXT:   3.00, 4.00, 7.00, 5.00;
# CHECK-NEXT:   8.00, 5.00, 0.00, 6.00;
# CHECK-NEXT:   6.00, 3.00, 7.00, 2.00]

# A_colmaj

# CHECK-NEXT:  [6.00, 4.00, 6.00, 3.00;
# CHECK-NEXT:   9.00, 0.00, 2.00, 7.00;
# CHECK-NEXT:   3.00, 5.00, 4.00, 8.00;
# CHECK-NEXT:   3.00, 0.00, 0.00, 1.00]

# C

# CHECK-NEXT:  [44.00, 64.00, 35.00, 48.00;
# CHECK-NEXT:   88.00, 73.00, 68.00, 66.00;
# CHECK-NEXT:   86.00, 81.00, 65.00, 81.00;
# CHECK-NEXT:   42.00, 47.00, 36.00, 50.00]


# C_colmaj transposed

# CHECK-NEXT:  [44.00, 64.00, 35.00, 48.00;
# CHECK-NEXT:   88.00, 73.00, 68.00, 66.00;
# CHECK-NEXT:   86.00, 81.00, 65.00, 81.00;
# CHECK-NEXT:   42.00, 47.00, 36.00, 50.00]

# CHECK-EMPTY:
# CHECK-NEXT: Test Passed: The results are equal!
