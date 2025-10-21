from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.x86_func import FuncOp

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch


@dataclass
class GeneratedCode:
    func_op: FuncOp
    builder: Builder
    arch: Arch
