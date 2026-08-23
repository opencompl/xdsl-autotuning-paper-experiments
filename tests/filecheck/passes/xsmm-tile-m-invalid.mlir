// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'xsmm-tile-m{arch=hsw}' | filecheck %s

builtin.module {}

// CHECK: xsmm-tile-m currently supports AVX-512 architectures only
