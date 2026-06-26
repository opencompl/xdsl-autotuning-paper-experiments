from dataclasses import dataclass
from typing import cast

from xdsl.builder import Builder
from xdsl.dialects import x86
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Attribute, OperationInvT, SSAValue

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch


@dataclass
class GeneratedCode:
    func_op: FuncOp
    builder: Builder
    arch: Arch

    current_val_by_reg: dict[Attribute, SSAValue]

    def get_val(self, reg: x86.ops.R1InvT) -> SSAValue[x86.ops.R1InvT]:
        return self.current_val_by_reg[reg]  # pyright: ignore

    @property
    def current_block(self):
        return self.builder.insertion_point.block

    def insert(self, op: OperationInvT) -> OperationInvT:
        self.builder.insert(op)
        for res in op.results:
            self.current_val_by_reg[res.type] = res
        return op
