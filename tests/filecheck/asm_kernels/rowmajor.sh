# RUN: bash %s %t | filecheck %s

clang -o $1 arm_4x4_matmul_asm_rowmajor/main.c arm_4x4_matmul_asm_rowmajor/test1.s
$1

# CHECK: Test Passed: The results are equal!
