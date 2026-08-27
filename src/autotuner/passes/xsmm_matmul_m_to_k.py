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
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulMOp
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.schedules import matmul_m_to_k


@dataclass
class ConvertMatmulMToKPattern(RewritePattern):
    """Expose the K body and preserve the pointer semantics of matmul_m."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulMOp, rewriter: PatternRewriter, /) -> None:
        m_blocking = op.m_blocking.value.data
        vector_length = 512 // op.datatype.bitwidth
        if m_blocking % vector_length and op.mask is None:
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

    arch: str = "skx"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            arch = ARCH_BY_CODE[self.arch]
        except KeyError as error:
            raise PassFailedException(
                f"unknown architecture code '{self.arch}'"
            ) from error
        if not (Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT):
            raise PassFailedException(
                "xsmm-matmul-m-to-k currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            ConvertMatmulMToKPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
