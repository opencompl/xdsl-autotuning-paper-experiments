# RUN: bash %s %t | filecheck %s
set -e

# Use bare pointer call convention for simpler LLVM IR
mlir-opt arm_4x4_matmul_asm_rowmajor/test2.mlir \
    -expand-strided-metadata \
    --convert-func-to-llvm='use-bare-ptr-memref-call-conv' \
    --finalize-memref-to-llvm='use-generic-functions' \
    --convert-vector-to-llvm="enable-arm-neon" \
    --convert-cf-to-llvm \
    --convert-arith-to-llvm \
    --reconcile-unrealized-casts \
    --cse \
    --canonicalize \
    -o arm_4x4_matmul_asm_rowmajor/test2.llvm.mlir

mlir-translate arm_4x4_matmul_asm_rowmajor/test2.llvm.mlir \
    --mlir-to-llvmir \
    -o arm_4x4_matmul_asm_rowmajor/test2.ll

# Suppress the target triple warning
clang -o $1 \
    -Wno-override-module \
    arm_4x4_matmul_asm_rowmajor/main.c arm_4x4_matmul_asm_rowmajor/test2.ll

# Run executable
$1

# CHECK: Test Passed: The results are equal!
