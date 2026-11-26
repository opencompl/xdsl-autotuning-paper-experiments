// An explicit schedule decides the passes over memory: each pass stores all
// 64 points of 16 bytes once, whatever vector width it picks.

// RUN: fft-dft %t 64 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{schedule=8,8},FFT_LOWERING' -t x86-asm | STORED_BYTES | filecheck %s --check-prefix TWO
// TWO: {{^}}2048{{$}}

// RUN: fft-dft %t 64 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{schedule=2,4,8},FFT_LOWERING' -t x86-asm | STORED_BYTES | filecheck %s --check-prefix THREE
// THREE: {{^}}3072{{$}}

// RUN: fft-dft %t 64 DP && not xdsl-opt %t -f mlir -p 'fft-apply-schedule{schedule=8,4},FFT_LOWERING' 2>&1 | filecheck %s --check-prefix PRODUCT
// PRODUCT: the schedule (8, 4) does not multiply to 64 points

// RUN: fft-dft %t 64 DP && not xdsl-opt %t -f mlir -p 'fft-apply-schedule{max_leaf=8 schedule=16,4},FFT_LOWERING' 2>&1 | filecheck %s --check-prefix LEAF
// LEAF: no register kernel for a pass of 16 points (max_leaf=8)
