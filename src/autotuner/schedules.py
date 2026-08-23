from xdsl import ir
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import MatmulNOp


def split_n(
    rewriter: PatternRewriter, op: MatmulNOp, n_split: int
) -> tuple[MatmulNOp, MatmulNOp]:
    """
    Split the matrix multiplication in two along the N dimension.
    Precondition: `0 < n_split < n_blocking`.
    """
    n_blocking = op.n_blocking.value.data
    assert 0 < n_split < n_blocking, (
        f"Invalid n_split value {n_split} outside of n_blocking range {n_blocking}"
    )
    first = MatmulNOp(
        *op.operands,
        n_start=op.n_start.value.data,
        n_blocking=n_split,
        m=op.m.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
    )
    second = MatmulNOp(
        *first.results,
        n_start=op.n_start.value.data + n_split,
        n_blocking=n_blocking - n_split,
        m=op.m.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
    )
    rewriter.replace(op, (first, second), second.results)
    return first, second


def tile_n(
    rewriter: PatternRewriter,
    op: MatmulNOp,
    n_tile: int,
    nloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> MatmulNOp:
    n = op.n_blocking.value.data
    n_start = op.n_start.value.data
    n_init = rewriter.insert(
        x86.ops.DI_MovOp(n_start, destination=nloop_register),
        insertion_point=InsertPoint.before(op),
    )
    inputs = tuple(op.operands)
    body = ir.Block(
        arg_types=(n_init.destination.type, *(value.type for value in inputs))
    )
    a, b, c, rbp, rsp = body.args[1:]
    tiled_matmul = MatmulNOp(
        a,
        b,
        c,
        rbp,
        rsp,
        m=op.m.value.data,
        n_start=n_start,
        n_blocking=n_tile,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
    )
    body.add_ops((tiled_matmul, x86_scf.YieldOp(*tiled_matmul.results)))
    nloop = rewriter.insert(
        x86_scf.ForOp(
            n_init.destination,
            builtin.IntegerAttr(n_start + n, x86.ops.si32),
            builtin.IntegerAttr(n_tile, x86.ops.si32),
            inputs,
            body,
        ),
        insertion_point=InsertPoint.before(op),
    )
    rewriter.replace(op, [], tuple(nloop.results[1:]))
    return tiled_matmul
