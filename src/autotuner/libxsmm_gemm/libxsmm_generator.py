from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Attribute, OperationInvT, SSAValue

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

    def insert(self, op: OperationInvT) -> OperationInvT:
        self.builder.insert(op)
        for res in op.results:
            self.current_val_by_reg[res.type] = res
        return op
