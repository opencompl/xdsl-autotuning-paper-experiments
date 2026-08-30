from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulIterator, MatmulOp
from autotuner.schedules import matmul_m_to_reg
from autotuner.strategy import get_xsmm_strategy


@dataclass
class ConvertMatmulMToRegPattern(RewritePattern):
    """Expose the register body and preserve the M iterator semantics."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        if op.iterator.data != MatmulIterator.M:
            return
        if (
            len(op.ins) > 1
            or op.outs
            or (
                op.ins
                and not isinstance(op.ins[0].type, x86.registers.AVX512MaskRegisterType)
            )
        ):
            raise PassFailedException(
                "xsmm-matmul-m-to-reg currently supports only one mask out"
            )
        m_blocking = op.m.value.data
        vector_length = 512 // op.datatype.bitwidth
        if m_blocking % vector_length and not op.ins:
            raise PassFailedException(
                "xsmm-matmul-m-to-reg requires a mask for a partial M vector; "
                "run tile_m first"
            )
        matmul_m_to_reg(rewriter, op)


@dataclass(frozen=True)
class XsmmMatmulMToRegPass(ModulePass):
    """Lower an M-iterating matmul to C accesses and matmul_reg.

    The generated adjustments preserve the M operation's pointer semantics:
    A and C advance by the M block, while B remains unchanged. These results do
    not depend on whether a later transformation tiles the generated matmul_reg.
    """

    name = "xsmm-matmul-m-to-reg"

    strategy: str = "libxsmm-skx"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            ConvertMatmulMToRegPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
