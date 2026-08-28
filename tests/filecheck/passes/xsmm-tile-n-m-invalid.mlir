// RUN: xdsl-opt %s --split-input-file --verify-diagnostics -p 'xsmm-tile-n-m{strategy=unknown}' | filecheck %s

builtin.module {}

// CHECK: unknown XSMM strategy 'unknown'; expected one of: libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst
