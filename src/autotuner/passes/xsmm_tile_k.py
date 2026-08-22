from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_scf
from xdsl.dialects.x86 import ops, registers
from xdsl.ir import Block, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from autotuner.dialects.xsmm import MatmulKOp

K_BLOCKING = 4
K_THRESHOLD = 23


def _matmul_k_with_inputs(
    op: MatmulKOp, inputs: Sequence[SSAValue], k_blocking: int
) -> MatmulKOp:
    a, b, c, rbp, rsp, *rest = inputs
    if op.mask is None:
        mask = None
        accumulators = rest
    else:
        mask, *accumulators = rest

    return MatmulKOp(
        a,
        b,
        c,
        rbp,
        rsp,
        mask,
        accumulators,
        m_blocking=op.m_blocking.value.data,
        n_blocking=op.n_blocking.value.data,
        k_blocking=k_blocking,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
    )


@dataclass
class TileMatmulKPattern(RewritePattern):
    """Tile a matmul_k without changing its pointer results.

    Each tiled body advances A and B through its portion of K. The loop-carried
    results therefore already have the same pointer values as the original op;
    the transformation must not reset either pointer after the loop.
    """

    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulKOp, rewriter: PatternRewriter, /) -> None:
        k = op.k_blocking.value.data
        if k <= K_THRESHOLD:
            return

        max_blocked_k = (k // K_BLOCKING) * K_BLOCKING
        kloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R12
        )
        k_init = ops.DI_MovOp(0, destination=kloop_register)

        body = Block(
            arg_types=(k_init.destination.type, *(value.type for value in op.operands))
        )
        tiled_matmul = _matmul_k_with_inputs(op, body.args[1:], k_blocking=K_BLOCKING)
        body.add_ops((tiled_matmul, x86_scf.YieldOp(*tiled_matmul.results)))
        kloop = x86_scf.ForOp(
            k_init.destination,
            builtin.IntegerAttr(max_blocked_k, ops.si32),
            builtin.IntegerAttr(K_BLOCKING, ops.si32),
            op.operands,
            body,
        )

        loop_results = tuple(kloop.results[1:])
        new_ops: list = [k_init, kloop]
        if remainder := k % K_BLOCKING:
            remainder_matmul = _matmul_k_with_inputs(
                op, loop_results, k_blocking=remainder
            )
            new_ops.append(remainder_matmul)
            results = tuple(remainder_matmul.results)
        else:
            results = loop_results

        rewriter.replace(op, new_ops, results)


@dataclass(frozen=True)
class XsmmTileKPass(ModulePass):
    """Tile large xsmm.matmul_k operations while preserving pointer results.

    This applies the current LIBXSMM K policy, but the chosen representation does
    not change the operation semantics: in particular, ``b_out`` advances by the
    full K extent for both tiled and untiled operations.
    """

    name = "xsmm-tile-k"

    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            TileMatmulKPattern(self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
