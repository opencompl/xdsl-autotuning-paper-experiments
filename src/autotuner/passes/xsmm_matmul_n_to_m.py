from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from autotuner.dialects.xsmm import MatmulNOp
from autotuner.schedules import matmul_n_to_m


@dataclass
class ConvertMatmulNToMPattern(RewritePattern):
    """Expose the M body and preserve the pointer semantics of matmul_n."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulNOp, rewriter: PatternRewriter, /) -> None:
        matmul_n_to_m(rewriter, op)


@dataclass(frozen=True)
class XsmmMatmulNToMPass(ModulePass):
    """Lower matmul_n to matmul_m and N pointer adjustments.

    The generated adjustments preserve the N operation's pointer semantics:
    A remains unchanged, while B and C advance by the N block. These results do
    not depend on how later transformations tile the generated matmul_m.
    """

    name = "xsmm-matmul-n-to-m"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertMatmulNToMPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
