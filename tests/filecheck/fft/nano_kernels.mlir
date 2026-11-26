// The radix-4 and radix-8 nano-kernels as adds and FMAs: multiplying by -i
// folds into vfmaddsub/vfmsubadd, the sqrt(1/2) of the radix-8 twiddles folds
// into FMAs, and no negation survives as an XOR or a multiply.

// RUN: fft-dft %t 4 DP && xdsl-opt %t -f mlir -p FFT_PIPELINE -t x86-asm | ARITHMETIC | filecheck %s --check-prefix RADIX4
// RADIX4:      {{^ *}}3 vaddpd
// RADIX4-NEXT: {{^ *}}1 vfmaddsub231pd
// RADIX4-NEXT: {{^ *}}1 vfmsubadd231pd
// RADIX4-NEXT: {{^ *}}3 vsubpd
// RADIX4-EMPTY:

// RUN: fft-dft %t 8 DP && xdsl-opt %t -f mlir -p FFT_PIPELINE -t x86-asm | ARITHMETIC | filecheck %s --check-prefix RADIX8
// RADIX8:      {{^ *}}7 vaddpd
// RADIX8-NEXT: {{^ *}}2 vfmadd231pd
// RADIX8-NEXT: {{^ *}}4 vfmaddsub231pd
// RADIX8-NEXT: {{^ *}}4 vfmsubadd231pd
// RADIX8-NEXT: {{^ *}}2 vfnmadd231pd
// RADIX8-NEXT: {{^ *}}7 vsubpd
// RADIX8-EMPTY:
