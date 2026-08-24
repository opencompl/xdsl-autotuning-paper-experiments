// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'xsmm-split-n{arch=hsw}' | filecheck %s

builtin.module {}

// CHECK: xsmm-split-n currently supports AVX-512 architectures only
