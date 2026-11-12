// The radix-4 and radix-8 nano-kernels as adds and FMAs: multiplying by -i is
// free, the sqrt(1/2) of the radix-8 twiddles folds into FMAs, and no negation
// survives as an XOR or a multiply.

// RUN: fft-dft %t 4 DP && xdsl-opt %t -f mlir -p FFT_PIPELINE -t x86-asm | ARITHMETIC | filecheck %s --check-prefix RADIX4
// RADIX4:      {{^ *}}8 vaddpd
// RADIX4-NEXT: {{^ *}}8 vsubpd
// RADIX4-EMPTY:

// RUN: fft-dft %t 8 DP && xdsl-opt %t -f mlir -p FFT_PIPELINE -t x86-asm | ARITHMETIC | filecheck %s --check-prefix RADIX8
// RADIX8:      {{^ *}}22 vaddpd
// RADIX8-NEXT: {{^ *}}4 vfmadd231pd
// RADIX8-NEXT: {{^ *}}4 vfnmadd231pd
// RADIX8-NEXT: {{^ *}}22 vsubpd
// RADIX8-EMPTY:
