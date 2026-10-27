// RUN: xdsl-opt %s --verify-diagnostics -p 'xsmm-apply-schedule{strategy=unknown}' | filecheck %s

builtin.module {}

// CHECK: unknown XSMM strategy 'unknown'; expected one of: libxsmm-cpx-bf16, libxsmm-cpx-bf16-vnni, libxsmm-skx, libxsmm-skx-fsdbcst, libxsmm-skx-nofsdbcst
