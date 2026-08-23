from xdsl.pattern_rewriter import PatternRewriter

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
