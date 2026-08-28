// REQUIRES: xtc
// RUN: libxtcmm-gemm dense %t matmul_bac 16 16 100 16 100 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// XTC source IR (linalg payload + transform-dialect schedule, before it is
// applied) reproducing LIBXSMM's 16x16x100 DP microkernel schedule where K is
// above the unroll threshold, so it becomes a rolled reduction loop:
//   * K = 100 > 23 is tiled by the blocking factor 4 (tile_sizes [0, 0, 4] on
//     the "./K" loop) with a body of 4 unrolled steps -> a rolled K loop, not
//     a full unroll
//   * M = 16 is one register block, N = 16 a single tier

// CHECK:      func.func @matmul_bac({{.*}}memref<16x100xf64>{{.*}}memref<100x16xf64>
// CHECK:        linalg.matmul
// CHECK:      transform.structured.tile_using_for {{.*}} tile_sizes [0, 0, 4]
// CHECK-NEXT: transform.annotate {{.*}} "./K"
// CHECK:      transform.loop.unroll {{.*}} {factor = 4 : i64}
