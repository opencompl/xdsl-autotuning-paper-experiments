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


def _insert_m_loop(
    rewriter: PatternRewriter,
    op: MatmulMOp,
    inputs: tuple[ir.SSAValue, ...],
    *,
    lower_bound: int,
    upper_bound: int,
    m_blocking: int,
    mask: ir.SSAValue | None,
    mloop_register: x86.registers.GeneralRegisterType,
) -> tuple[tuple[ir.SSAValue, ...], MatmulMOp]:
    m_init = rewriter.insert(
        x86.ops.DI_MovOp(lower_bound, destination=mloop_register),
        insertion_point=InsertPoint.before(op),
    )
    iter_args = (*inputs, *((mask,) if mask is not None else ()))
    body = ir.Block(
        arg_types=(m_init.destination.type, *(value.type for value in iter_args))
    )
    a, b, c, rbp, rsp, *rest = body.args[1:]
    tiled_matmul = MatmulMOp(
        a,
        b,
        c,
        rbp,
        rsp,
        rest[0] if rest else None,
        m_blocking=m_blocking,
        n_blocking=op.n_blocking.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
    )
    body.add_ops((tiled_matmul, x86_scf.YieldOp(*tiled_matmul.results)))
    mloop = rewriter.insert(
        x86_scf.ForOp(
            m_init.destination,
            builtin.IntegerAttr(upper_bound, x86.ops.si32),
            builtin.IntegerAttr(m_blocking, x86.ops.si32),
            iter_args,
            body,
        ),
        insertion_point=InsertPoint.before(op),
    )
    return tuple(mloop.results[1:6]), tiled_matmul


def _initialize_avx512_mask(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    gp_reg_tmp: x86.registers.GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
    mask_count: int,
    datatype: builtin.Float64Type | builtin.Float32Type,
) -> ir.SSAValue[x86.registers.AVX512MaskRegisterType]:
    match datatype:
        case builtin.Float64Type():
            mask = 0xFF
            op_type = x86.ops.KS_KMovBOp
        case builtin.Float32Type():
            mask = 0xFFFF
            op_type = x86.ops.KS_KMovWOp

    # shift right by "inverse" remainder
    mask = mask >> mask_count

    # /* move mask to GP register */
    mask_tmp_val = rewriter.insert(
        x86.ops.DI_MovOp(mask, destination=gp_reg_tmp), insertion_point=insert_point
    ).destination

    # loading the mask register
    return rewriter.insert(
        op_type(
            mask_tmp_val,
            destination=mask_reg,
        ),
        insertion_point=insert_point,
    ).destination


def tile_m(
    rewriter: PatternRewriter,
    op: MatmulMOp,
    *,
    m_blocking: int,
    mloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_tmp_reg: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> tuple[MatmulMOp, MatmulMOp | None]:
    """
    Tiles M, inserting a for loop and an optional remainder matmul.
    Returns the new tiled matmul and optional remainder matmul.
    """
    match op.datatype:
        case builtin.Float64Type():
            vector_length = 8
        case builtin.Float32Type():
            vector_length = 16

    m = op.m_blocking.value.data
    blocked_end = m // m_blocking * m_blocking
    inputs = tuple(op.operands[:5])

    mask = None
    if m_blocking % vector_length:
        mask = _initialize_avx512_mask(
            rewriter,
            InsertPoint.before(op),
            mask_tmp_reg,
            mask_reg,
            vector_length - m_blocking % vector_length,
            op.datatype,
        )
    inputs, tiled_matmul = _insert_m_loop(
        rewriter,
        op,
        inputs,
        lower_bound=0,
        upper_bound=blocked_end,
        m_blocking=m_blocking,
        mask=mask,
        mloop_register=mloop_register,
    )

    if remainder := m - blocked_end:
        mask = None
        if remainder % vector_length:
            mask = _initialize_avx512_mask(
                rewriter,
                InsertPoint.before(op),
                mask_tmp_reg,
                mask_reg,
                vector_length - remainder % vector_length,
                op.datatype,
            )

        inputs, remainder_matmul = _insert_m_loop(
            rewriter,
            op,
            inputs,
            lower_bound=blocked_end,
            upper_bound=m,
            m_blocking=remainder,
            mask=mask,
            mloop_register=mloop_register,
        )
    else:
        remainder_matmul = None

    rewriter.replace(op, [], inputs)

    return tiled_matmul, remainder_matmul
