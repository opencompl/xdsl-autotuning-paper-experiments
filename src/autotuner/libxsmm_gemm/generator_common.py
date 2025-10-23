from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86.registers import (
    RDI,
    RSI,
    RDX,
)
from xdsl.ir import Block, Region
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.rewriter import InsertPoint, Rewriter


def libxsmm_mmfunction_signature(module: ModuleOp, routine_name: str) -> FuncOp:
    operand_types = (RDI, RSI, RDX)

    func = FuncOp(
        routine_name,
        Region(Block((RetOp(),), arg_types=operand_types)),
        (operand_types, ()),
        "public",
    )

    Rewriter.insert_op(func, InsertPoint.at_end(module.body.block))

    return func


@dataclass
class MicroKernelConfig:
    """libxsmm_micro_kernel_config"""
