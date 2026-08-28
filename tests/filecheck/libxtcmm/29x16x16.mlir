// REQUIRES: xtc
// RUN: libxtcmm-gemm dense %t matmul_bac 16 29 16 16 16 16 1 1 1 1 skx nopf DP && cat %t | filecheck %s

// XTC source IR (linalg payload + transform-dialect schedule, before it is
// applied) reproducing LIBXSMM's 16x29x16 DP microkernel schedule:
//   * N = 29 is split into two equalized register tiers: 20 (step 10) + 9
//   * M = 16 is one register block, vectorized (no masking -> bare vectorize)
//   * K = 16 <= 23 is fully unrolled (k_step unroll factor = 16)

// CHECK:      module attributes {transform.with_named_sequence}
// CHECK:      func.func @matmul_bac({{.*}}memref<29x16xf64>{{.*}}memref<16x16xf64>{{.*}}memref<29x16xf64>
// CHECK:        linalg.matmul
// CHECK:      transform.structured.split {{%.*}} after 20 {dimension = 0 : i64}
// CHECK:      transform.include @_vecto
// CHECK:      transform.loop.unroll {{.*}} {factor = 16 : i64}
