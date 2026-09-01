// REQUIRES: xtc
// RUN: libxtcmm-gemm dense %t matmul_bac 20 16 16 20 16 20 1 1 1 1 skx nopf DP && cat %t | filecheck %s
// RUN: libxtcmm-gemm dense %t matmul_bac 20 16 16 20 16 20 1 1 1 1 skx nopf DP --mask-tail && cat %t | filecheck %s --check-prefix MASK

// XTC source IR (linalg payload + transform-dialect schedule, before it is
// applied) reproducing LIBXSMM's 20x16x16 DP microkernel schedule where M is
// not a multiple of the vector length:
//   * by default, the 20 % 8 = 4 tail is a narrow unmasked vector
//   * --mask-tail pins the vector width to 8 so the tail is masked
//   * K = 16 <= 23 is fully unrolled

// CHECK:      func.func @matmul_bac({{.*}}memref<16x20xf64>
// CHECK:        linalg.matmul
// CHECK:      transform.structured.vectorize {{%.*}} : !transform.any_op

// MASK:       transform.structured.vectorize {{%.*}} vector_sizes [1, 8, 1]
