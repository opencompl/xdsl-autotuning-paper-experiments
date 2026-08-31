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
from autotuner.schedules import matmul_k_to_reg
from autotuner.strategy import get_xsmm_strategy


@dataclass
class ConvertMatmulKToRegPattern(RewritePattern):
    """Expose the register body of a K-iterating matmul."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        if op.iterator.data != MatmulIterator.K:
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
                "xsmm-matmul-k-to-reg currently supports only one mask in"
            )
        m_blocking = op.m.value.data
        vector_length = 512 // op.datatype.bitwidth
        if m_blocking % vector_length and not op.ins:
            raise PassFailedException(
                "xsmm-matmul-k-to-reg requires a mask for a partial M vector; "
                "run tile_m first"
            )
        matmul_k_to_reg(rewriter, op)


@dataclass(frozen=True)
class XsmmMatmulKToRegPass(ModulePass):
    """Lower a K-iterating matmul to C accesses and matmul_reg.

    The operation already carries its complete pointer-result contract, so the
    conversion only moves C through vector registers and leaves its pointer
    unchanged.
    """

    name = "xsmm-matmul-k-to-reg"

    strategy: str = "libxsmm-skx"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            ConvertMatmulKToRegPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
