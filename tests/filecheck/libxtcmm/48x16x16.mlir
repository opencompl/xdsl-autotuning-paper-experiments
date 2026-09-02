// RUN: libxtcmm-gemm dense %t matmul_bac 48 16 16 48 16 48 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// XTC source IR (linalg payload + transform-dialect schedule, before it is
// applied) reproducing LIBXSMM's 48x16x16 DP microkernel schedule, where M has
// a remainder register block:
//   * M = 48 is split on the contiguous dim 1 after 32 -> a full 32-wide block
//     tier plus a 16-wide remainder tier
//   * K = 16 <= 23 is fully unrolled

// CHECK:      func.func @matmul_bac({{.*}}memref<16x48xf64>
// CHECK:        linalg.matmul
// CHECK:      transform.structured.split {{%.*}} after 32 {dimension = 1 : i64}
// CHECK:      transform.include @_vecto
