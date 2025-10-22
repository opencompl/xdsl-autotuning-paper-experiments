from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Attribute, SSAValue

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch


@dataclass
class GeneratedCode:
    func_op: FuncOp
    builder: Builder
    arch: Arch

    current_val_by_reg: dict[Attribute, SSAValue]

    @property
    def current_block(self):
        return self.builder.insertion_point.block
