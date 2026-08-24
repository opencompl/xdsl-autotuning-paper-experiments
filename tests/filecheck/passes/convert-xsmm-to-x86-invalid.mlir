// RUN: xdsl-opt %s --verify-diagnostics -p 'convert-xsmm-to-x86{nano-kernel=unknown}' | filecheck %s

builtin.module {}

// CHECK: unknown SKX nano-kernel 'unknown'; expected one of: libxsmm, skx-fsdbcst, skx-nofsdbcst
