from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import SSAValue

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch


@dataclass
class GeneratedCode:
    func_op: FuncOp
    builder: Builder
    arch: Arch
    current_a: SSAValue
    current_b: SSAValue
    current_c: SSAValue

    @property
    def current_block(self):
        return self.builder.insertion_point.block
