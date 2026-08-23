from xdsl import ir
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import MatmulMOp, MatmulNOp


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


def matmul_n_to_m(rewriter: PatternRewriter, op: MatmulNOp) -> MatmulMOp:
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

    # MatmulNOp increments by n_blocking, and MatmulMOp increments by m_blocking
    # Add ops to adjust A, B and C pointers as appropriate, so that the results at the
    # end of this transform have the expected values.
    c_out_op = x86.ops.RI_AddOp(
        matmul_m.c_out,
        (n_blocking * ldc - m) * element_size,
        register_out=matmul_m.c_out.type,
    )
    b_out_op = x86.ops.RI_AddOp(
        matmul_m.b_out,
        n_blocking * ldb * element_size,
        register_out=matmul_m.b_out.type,
    )
    a_out_op = x86.ops.RI_SubOp(
        matmul_m.a_out,
        m * element_size,
        register_out=matmul_m.a_out.type,
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
    return matmul_m
