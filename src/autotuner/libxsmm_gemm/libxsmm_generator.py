from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects.x86_func import FuncOp


@dataclass
class GeneratedCode:
    func_op: FuncOp
    builder: Builder
