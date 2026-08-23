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

from autotuner.dialects.xsmm import MatmulMOp, MatmulNOp


@dataclass
class ConvertMatmulNToMPattern(RewritePattern):
    """Expose the M body and preserve the pointer semantics of matmul_n."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulNOp, rewriter: PatternRewriter, /) -> None:
        m = op.m.value.data
        n_blocking = op.n_blocking.value.data
        ldb = op.ldb.value.data
        ldc = op.ldc.value.data
        element_size = op.datatype.bitwidth // 8

        matmul_m = MatmulMOp(
            op.a,
            op.b,
            op.c,
            op.rbp,
            op.rsp,
            None,
            m_blocking=m,
            n_blocking=n_blocking,
            k=op.k.value.data,
            lda=op.lda.value.data,
            ldb=ldb,
            ldc=ldc,
            datatype=op.datatype,
            aligned_a=bool(op.aligned_a),
            aligned_c=bool(op.aligned_c),
        )
        c_out_op = x86.ops.RI_AddOp(
            matmul_m.c_out,
            (n_blocking * ldc - m) * element_size,
            register_out=op.c_out.type,
        )
        b_out_op = x86.ops.RI_AddOp(
            matmul_m.b_out,
            n_blocking * ldb * element_size,
            register_out=op.b_out.type,
        )
        a_out_op = x86.ops.RI_SubOp(
            matmul_m.a_out,
            m * element_size,
            register_out=op.a_out.type,
        )

        rewriter.replace(
            op,
            (matmul_m, c_out_op, b_out_op, a_out_op),
            (
                a_out_op.register_out,
                b_out_op.register_out,
                c_out_op.register_out,
                matmul_m.rbp_out,
                matmul_m.rsp_out,
            ),
        )


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
