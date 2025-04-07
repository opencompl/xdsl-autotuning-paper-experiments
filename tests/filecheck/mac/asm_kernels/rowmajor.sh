# RUN: clang -o %t arm_4x4_matmul_asm_rowmajor/main.c arm_4x4_matmul_asm_rowmajor/test1.s && %t | filecheck %s

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
