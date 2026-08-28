// RUN: xdsl-opt %s --verify-diagnostics -p 'convert-xsmm-to-x86{strategy=unknown}' | filecheck %s

builtin.module {}

// CHECK: unknown XSMM strategy 'unknown'; expected one of: libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst
