// The interleaved register layout holds a point in one register, so a
// 16-point kernel fits in the 32 zmm registers, and data moves between memory
// and registers as it is, without unpacking real and imaginary parts.

// 256 points in two passes of 16-point kernels: each pass stores all 256
// points of 16 bytes once, and no pass unpacks.
// RUN: fft-dft %t 256 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{layout=interleaved max_leaf=16 schedule=16,16},FFT_LOWERING' -t x86-asm > %t.s
// RUN: STORED_BYTES < %t.s | filecheck %s --check-prefix BYTES
// RUN: awk '/vunpck/ { n++ } END { print n + 0 }' < %t.s | filecheck %s --check-prefix UNPACKS
// BYTES: {{^}}8192{{$}}
// UNPACKS: {{^}}0{{$}}

// The split layout runs out of registers on the same schedule.
// RUN: fft-dft %t 256 DP && not xdsl-opt %t -f mlir -p 'fft-apply-schedule{max_leaf=16 schedule=16,16},FFT_LOWERING' -t x86-asm 2>&1 | filecheck %s --check-prefix SPLIT
// SPLIT: OutOfRegisters

// A batch the vector width does not divide ends in narrower vectors, and a
// side whose transforms are neither contiguous nor transposable moves each
// lane on its own. 15 points as 3 DFTs of 5, then 5 twiddled DFTs of 3: the
// first pass stores its 2 + 1 transforms lane by lane, the second runs its 5
// as a zmm vector of 4 and an xmm vector of 1.
// RUN: fft-dft %t 15 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{layout=interleaved schedule=5,3},FFT_LOWERING' -t x86-asm | filecheck %s --check-prefix RAGGED
// RAGGED:     vextractf128 [rsi+{{[0-9]+}}], ymm{{[0-9]+}}, 1
// RAGGED:     vmovupd [rsi+{{[0-9]+}}], zmm{{[0-9]+}}
// RAGGED:     vmovupd [rsi+{{[0-9]+}}], xmm{{[0-9]+}}

// Duplicated twiddle factors for 1024 points of f64 take 32 KiB, as much as
// the data, so beside it they overflow a 32 KiB L1 and the pool stores each
// factor once instead, its parts duplicated as they load.
// RUN: fft-dft %t 1024 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{layout=interleaved max_leaf=16},FFT_LOWERING' -t x86-asm | filecheck %s --check-prefix COMPACT
// RUN: fft-dft %t 1024 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{layout=interleaved max_leaf=16 l1_bytes=65536},FFT_LOWERING' -t x86-asm | filecheck %s --check-prefix DUPLICATED
// COMPACT:        vpermilpd zmm{{[0-9]+}}, [rax+{{[0-9]+}}], 255
// DUPLICATED-NOT: vpermilpd zmm{{[0-9]+}}, [rax

// A pass that stores fewer bytes at a time than the next one loads leaves the
// second waiting on the store buffer, which a DFT this small cannot hide: 10
// points stay one kernel and store their 160 bytes once.
// RUN: fft-dft %t 10 DP && xdsl-opt %t -f mlir -p 'fft-apply-schedule{layout=interleaved max_leaf=16},FFT_LOWERING' -t x86-asm | STORED_BYTES | filecheck %s --check-prefix SMALL
// SMALL: {{^}}160{{$}}
