# RUN: bash %s %t | filecheck %s

ROOT_DIR=$(dirname $(dirname $(dirname $(dirname $0))))
clang -o $1 $ROOT_DIR/arm_4x4_matmul_asm_colmajor/main.c $ROOT_DIR/arm_4x4_matmul_asm_colmajor/test1.s
$1

# CHECK: Test Passed: The results are equal!
