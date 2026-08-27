// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'xsmm-tile-n-m{arch=hsw}' | filecheck %s

builtin.module {}

// CHECK: xsmm-tile-n-m currently supports AVX-512 architectures only
