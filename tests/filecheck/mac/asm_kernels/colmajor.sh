# RUN: clang -o %t arm_4x4_matmul_asm_colmajor/main.c arm_4x4_matmul_asm_colmajor/test1.s && %t | filecheck %s

# CHECK: Test Passed: The results are equal!
