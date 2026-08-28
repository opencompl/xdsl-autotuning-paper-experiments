// REQUIRES: xtc
// RUN: libxtcmm-gemm dense %t matmul_bac 20 16 16 20 16 20 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// XTC source IR (linalg payload + transform-dialect schedule, before it is
// applied) reproducing LIBXSMM's 20x16x16 DP microkernel schedule where M is
// not a multiple of the vector length, so the tail lane is masked:
//   * M = 20 is one register block vectorized to a fixed width 8 with masking
//     (vectorize ... vector_sizes [1, 8, 1]) -> the 20 % 8 = 4 tail is masked,
//     exactly LIBXSMM's masked microkernel
//   * K = 16 <= 23 is fully unrolled

// CHECK:      func.func @matmul_bac({{.*}}memref<16x20xf64>
// CHECK:        linalg.matmul
// CHECK:      transform.structured.vectorize {{%.*}} vector_sizes [1, 8, 1]
