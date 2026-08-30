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
from autotuner.schedules import matmul_m_to_k
from autotuner.strategy import get_xsmm_strategy


@dataclass
class ConvertMatmulMToKPattern(RewritePattern):
    """Expose the K body and preserve the pointer semantics of matmul_m."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        if op.iterator.data != MatmulIterator.M:
            return
        if (
            op.ins
            or len(op.outs) > 1
            or (
                op.outs
                and not isinstance(
                    op.outs[0].type, x86.registers.AVX512MaskRegisterType
                )
            )
        ):
            raise PassFailedException(
                "xsmm-matmul-m-to-k currently supports only one mask out"
            )
        m_blocking = op.m.value.data
        vector_length = 512 // op.datatype.bitwidth
        if m_blocking % vector_length and not op.outs:
            raise PassFailedException(
                "xsmm-matmul-m-to-k requires a mask for a partial M vector; "
                "run tile_m first"
            )
        matmul_m_to_k(rewriter, op)


@dataclass(frozen=True)
class XsmmMatmulMToKPass(ModulePass):
    """Lower matmul_m to C accesses, matmul_k, and pointer adjustments.

    The generated adjustments preserve the M operation's pointer semantics:
    A and C advance by the M block, while B remains unchanged. These results do
    not depend on whether a later transformation tiles the generated matmul_k.
    """

    name = "xsmm-matmul-m-to-k"

    strategy: str = "libxsmm-skx"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            ConvertMatmulMToKPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
