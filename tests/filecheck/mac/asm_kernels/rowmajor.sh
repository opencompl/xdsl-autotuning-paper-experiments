# RUN: clang -o %t arm_4x4_matmul_asm_rowmajor/main.c arm_4x4_matmul_asm_rowmajor/test1.s && %t | filecheck %s

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

# CHECK: Test Passed: The results are equal!
