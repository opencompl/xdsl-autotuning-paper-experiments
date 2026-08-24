// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'xsmm-tile-n{arch=hsw}' | filecheck %s

builtin.module {}

// CHECK: xsmm-tile-n currently supports AVX-512 architectures only
