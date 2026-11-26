// Each rewrite expands one fft operation in place.

// A 64-point DFT splits into two passes over memory, the first tiled into
// vectors of transforms, each loaded, transformed, and stored.
// RUN: fft-dft %t 64 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{schedule=8,8}' | filecheck %s --check-prefix EXPANDED
// EXPANDED-NOT: fft.
// EXPANDED:     x86_func.ret

// Constants sit beside the code that uses them, in mergeable sections, so
// no register holds their address.
// RUN: fft-dft %t 8 DP && xdsl-opt %t -f mlir -p FFT_PIPELINE -t x86-asm | filecheck %s --check-prefix CONSTANTS
// CONSTANTS:     .pushsection .rodata.cst8,"aM",@progbits,8
// CONSTANTS:     [rip + .Lfft{{[0-9]+}}]{1to2}
// CONSTANTS-NOT: lea

// 256 points in two passes of 16-point kernels: each pass stores all 256
// points of 16 bytes once, and no pass unpacks.
// RUN: fft-dft %t 256 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{max_leaf=16 schedule=16,16},FFT_LOWERING' -t x86-asm > %t.s
// RUN: STORED_BYTES < %t.s | filecheck %s --check-prefix BYTES
// RUN: awk '/vunpck/ { n++ } END { print n + 0 }' < %t.s | filecheck %s --check-prefix UNPACKS
// BYTES: {{^}}8192{{$}}
// UNPACKS: {{^}}0{{$}}

// A batch the vector width does not divide ends in narrower vectors, and a
// side whose transforms are neither contiguous nor transposable moves each
// lane on its own. 15 points as 3 DFTs of 5, then 5 twiddled DFTs of 3.
// RUN: fft-dft %t 15 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{schedule=5,3},FFT_LOWERING' -t x86-asm | filecheck %s --check-prefix RAGGED
// RAGGED:     vextractf128 [rsi+{{[0-9]+}}], ymm{{[0-9]+}}, 1
// RAGGED:     vmovupd [rsi+{{[0-9]+}}], zmm{{[0-9]+}}
// RAGGED:     vmovupd [rsi+{{[0-9]+}}], xmm{{[0-9]+}}

// Duplicated twiddle factors for 1024 points of f64 take 32 KiB, as much as
// the data, so beside it they overflow a 32 KiB L1 and each factor is stored
// once instead, its parts duplicated as they load.
// RUN: fft-dft %t 1024 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{max_leaf=16},FFT_LOWERING' -t x86-asm | filecheck %s --check-prefix COMPACT
// RUN: fft-dft %t 1024 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{max_leaf=16 l1_bytes=65536},FFT_LOWERING' -t x86-asm | filecheck %s --check-prefix DUPLICATED
// COMPACT:        vpermilpd zmm{{[0-9]+}}, [rip + .Lfft{{[0-9]+}}], 255
// DUPLICATED-NOT: vpermilpd zmm{{[0-9]+}}, [rip
