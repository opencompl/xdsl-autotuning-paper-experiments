# Lighthouse pipeline descriptors

The `x86_64/` tree is vendored from
`llvm/lighthouse@d50a962b181bb0fa6fec1b8026da7f52bff25be2`
(`examples/KernelBench/schedules/x86_64/`).

YAML includes such as `bufferization.yaml` and `llvm_lowering.yaml` are resolved
from the installed `lighthouse` package at build time.
