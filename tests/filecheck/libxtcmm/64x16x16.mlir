// REQUIRES: xtc
// RUN: libxtcmm-gemm dense %t matmul_bac 64 16 16 64 16 64 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// XTC source IR (linalg payload + transform-dialect schedule, before it is
// applied) reproducing LIBXSMM's 64x16x16 DP microkernel schedule, where M is
// a single tier of several register blocks:
//   * M = 64 is one tier of 32-wide register blocks (tile_sizes [0, 32, 0]) ->
//     a rolled M loop over 2 blocks, no M split
//   * K = 16 <= 23 is fully unrolled

// CHECK:      func.func @matmul_bac({{.*}}memref<16x64xf64>
// CHECK:        linalg.matmul
// CHECK:      transform.structured.tile_using_for {{.*}} tile_sizes [0, 32, 0]
// CHECK:      transform.include @_vecto
