from collections.abc import Sequence
from typing import cast
from xdsl import ir
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import MatmulIterator, MatmulKOp, MatmulOp


def split_n(
    rewriter: PatternRewriter, op: MatmulOp, n_split: int
) -> tuple[MatmulOp, MatmulOp]:
    """
    Split the matrix multiplication in two along the N dimension.
    Precondition: `0 < n_split < n_blocking`.
    """
    assert op.iterator.data == MatmulIterator.N
    n_blocking = op.n.value.data
    assert 0 < n_split < n_blocking, (
        f"Invalid n_split value {n_split} outside of n_blocking range {n_blocking}"
    )
    first = MatmulOp(
        op.a,
        op.b,
        op.c,
        op.rbp,
        op.rsp,
        op.ins,
        op.outs,
        n_start=op.n_start.value.data,
        n=n_split,
        m=op.m.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=MatmulIterator.N,
    )
    second = MatmulOp(
        first.a_out,
        first.b_out,
        first.c_out,
        first.rbp_out,
        first.rsp_out,
        op.ins,
        first.out_results,
        n_start=op.n_start.value.data + n_split,
        n=n_blocking - n_split,
        m=op.m.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=MatmulIterator.N,
    )
    rewriter.replace(op, (first, second), second.results)
    return first, second


def tile_n(
    rewriter: PatternRewriter,
    op: MatmulOp,
    n_tile: int,
    nloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> MatmulOp:
    assert op.iterator.data == MatmulIterator.N
    n = op.n.value.data
    n_start = op.n_start.value.data
    n_init = rewriter.insert(
        x86.ops.DI_MovOp(n_start, destination=nloop_register),
        insertion_point=InsertPoint.before(op),
    )
    inputs = (op.a, op.b, op.c, op.rbp, op.rsp, *op.outs)
    body = ir.Block(
        arg_types=(n_init.destination.type, *(value.type for value in inputs))
    )
    a, b, c, rbp, rsp, *outs = body.args[1:]
    tiled_matmul = MatmulOp(
        a,
        b,
        c,
        rbp,
        rsp,
        op.ins,
        outs,
        m=op.m.value.data,
        n_start=n_start,
        n=n_tile,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=MatmulIterator.N,
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


def matmul_n_to_m(rewriter: PatternRewriter, op: MatmulOp) -> MatmulOp:
    assert op.iterator.data == MatmulIterator.N
    m = op.m.value.data
    n = op.n.value.data
    ldb = op.ldb.value.data
    ldc = op.ldc.value.data
    element_size = op.datatype.bitwidth // 8

    matmul_m = MatmulOp(
        op.a,
        op.b,
        op.c,
        op.rbp,
        op.rsp,
        op.ins,
        op.outs,
        m=m,
        n_start=op.n_start.value.data,
        n=n,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=ldb,
        ldc=ldc,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=MatmulIterator.M,
    )

    # The N iterator increments by n, and the M iterator increments by m.
    # Add ops to adjust A, B and C pointers as appropriate, so that the results at the
    # end of this transform have the expected values.
    c_out_op = x86.ops.RI_AddOp(
        matmul_m.c_out,
        (n * ldc - m) * element_size,
        register_out=matmul_m.c_out.type,
    )
    b_out_op = x86.ops.RI_AddOp(
        matmul_m.b_out,
        n * ldb * element_size,
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
            *matmul_m.out_results,
        ),
    )
    return matmul_m


def matmul_m_to_k(rewriter: PatternRewriter, op: MatmulOp) -> MatmulKOp:
    assert op.iterator.data == MatmulIterator.M
    assert not op.ins, "matmul_m_to_k does not yet support ins"
    assert len(op.outs) <= 1, "matmul_m_to_k supports at most one mask out"
    m_blocking = op.m.value.data
    n_blocking = op.n.value.data
    k = op.k.value.data
    lda = op.lda.value.data
    vector_length = 512 // op.datatype.bitwidth
    insert_point = InsertPoint.before(op)

    mask = (
        None
        if not op.outs
        else ir.SSAValue.get(op.outs[0], type=x86.registers.AVX512MaskRegisterType)
    )
    accumulators = load_c(
        rewriter,
        insert_point,
        m_blocking,
        n_blocking,
        x86.registers.AVX512RegisterType,
        op.datatype,
        ldc=op.ldc.value.data,
        vector_length=vector_length,
        vector_reg_count=32,
        use_masking_a_c=mask is not None,
        aligned_c=bool(op.aligned_c),
        c_val=ir.SSAValue.get(op.c, type=x86.registers.GeneralRegisterType),
        mask_k1=mask,
    )
    matmul_k = rewriter.insert(
        MatmulKOp(
            op.a,
            op.b,
            op.c,
            op.rbp,
            op.rsp,
            mask,
            accumulators,
            m_blocking=m_blocking,
            n_blocking=n_blocking,
            k_blocking=k,
            lda=lda,
            ldb=op.ldb.value.data,
            datatype=op.datatype,
            aligned_a=bool(op.aligned_a),
        ),
        insertion_point=insert_point,
    )

    element_size = op.datatype.size
    b_out = rewriter.insert(
        x86.ops.RI_SubOp(
            matmul_k.b_out,
            k * element_size,
            register_out=op.b_out.type,
        ),
        insertion_point=insert_point,
    ).register_out

    store_c(
        rewriter,
        insert_point,
        m_blocking,
        n_blocking,
        op.datatype,
        ldc=op.ldc.value.data,
        vector_length=vector_length,
        use_masking_a_c=mask is not None,
        aligned_c=bool(op.aligned_c),
        c_val=ir.SSAValue.get(matmul_k.c_out, type=x86.registers.GeneralRegisterType),
        acc_vectors=tuple(matmul_k.accumulator_outs),
        mask_k1=(
            None
            if matmul_k.mask_out is None
            else ir.SSAValue.get(
                matmul_k.mask_out, type=x86.registers.AVX512MaskRegisterType
            )
        ),
    )

    c_out = rewriter.insert(
        x86.ops.RI_AddOp(
            matmul_k.c_out,
            m_blocking * element_size,
            register_out=op.c_out.type,
        ),
        insertion_point=insert_point,
    ).register_out
    a_out = rewriter.insert(
        x86.ops.RI_SubOp(
            matmul_k.a_out,
            (k * lda - m_blocking) * element_size,
            register_out=op.a_out.type,
        ),
        insertion_point=insert_point,
    ).register_out

    rewriter.replace(
        op,
        [],
        (
            a_out,
            b_out,
            c_out,
            matmul_k.rbp_out,
            matmul_k.rsp_out,
            *((matmul_k.mask_out,) if matmul_k.mask_out is not None else ()),
        ),
    )
    return matmul_k


def _matmul_k_with_inputs(
    op: MatmulKOp, inputs: Sequence[ir.SSAValue], k_blocking: int
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


def tile_k(
    rewriter: PatternRewriter,
    op: MatmulKOp,
    k_tile: int,
    kloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulKOp, MatmulKOp | None]:
    """Tile K, inserting an exact loop and an optional remainder matmul.

    Each tiled body advances A and B through its portion of K. The loop-carried
    results therefore already have the same pointer values as the original op;
    neither pointer is reset after the loop.

    Returns the matmul in the loop body and the optional remainder matmul.
    """
    k = op.k_blocking.value.data
    assert 0 < k_tile <= k, f"Invalid K tile {k_tile} for K extent {k}"

    insert_point = InsertPoint.before(op)
    blocked_end = k // k_tile * k_tile
    k_init = rewriter.insert(
        x86.ops.DI_MovOp(0, destination=kloop_register),
        insertion_point=insert_point,
    )

    inputs = tuple(op.operands)
    body = ir.Block(
        arg_types=(k_init.destination.type, *(value.type for value in inputs))
    )
    tiled_matmul = _matmul_k_with_inputs(op, body.args[1:], k_tile)
    body.add_ops((tiled_matmul, x86_scf.YieldOp(*tiled_matmul.results)))
    kloop = rewriter.insert(
        x86_scf.ForOp(
            k_init.destination,
            builtin.IntegerAttr(blocked_end, x86.ops.si32),
            builtin.IntegerAttr(k_tile, x86.ops.si32),
            inputs,
            body,
        ),
        insertion_point=insert_point,
    )

    loop_results = tuple(kloop.results[1:])
    if remainder := k % k_tile:
        remainder_matmul = rewriter.insert(
            _matmul_k_with_inputs(op, loop_results, remainder),
            insertion_point=insert_point,
        )
        results = tuple(remainder_matmul.results)
    else:
        remainder_matmul = None
        results = loop_results

    rewriter.replace(op, [], results)
    return tiled_matmul, remainder_matmul


def _insert_m_loop(
    rewriter: PatternRewriter,
    op: MatmulOp,
    inputs: tuple[ir.SSAValue, ...],
    *,
    lower_bound: int,
    upper_bound: int,
    m_blocking: int,
    mask: ir.SSAValue | None,
    mloop_register: x86.registers.GeneralRegisterType,
) -> tuple[tuple[ir.SSAValue, ...], MatmulOp]:
    m_init = rewriter.insert(
        x86.ops.DI_MovOp(lower_bound, destination=mloop_register),
        insertion_point=InsertPoint.before(op),
    )
    iter_args = (*inputs, *((mask,) if mask is not None else ()))
    body = ir.Block(
        arg_types=(m_init.destination.type, *(value.type for value in iter_args))
    )
    a, b, c, rbp, rsp, *rest = body.args[1:]
    tiled_matmul = MatmulOp(
        a,
        b,
        c,
        rbp,
        rsp,
        (),
        rest,
        m=m_blocking,
        n_start=op.n_start.value.data,
        n=op.n.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=MatmulIterator.M,
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
    op: MatmulOp,
    *,
    m_blocking: int,
    mloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_tmp_reg: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> tuple[MatmulOp, MatmulOp | None]:
    """
    Tiles M, inserting a for loop and an optional remainder matmul.
    Returns the new tiled matmul and optional remainder matmul.
    """
    match op.datatype:
        case builtin.Float64Type():
            vector_length = 8
        case builtin.Float32Type():
            vector_length = 16

    assert op.iterator.data == MatmulIterator.M
    assert not op.ins and not op.outs
    m = op.m.value.data
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


AccVals = tuple[ir.SSAValue[x86.registers.AVX512RegisterType], ...]


def load_vector(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: ir.SSAValue[x86.registers.GeneralRegisterType],
    offset: int,
    mask: ir.SSAValue[x86.registers.AVX512MaskRegisterType] | None = None,
    *,
    aligned: bool,
    destination: x86.registers.AVX512RegisterType,
) -> ir.SSAValue[x86.registers.AVX512RegisterType]:
    if mask is None:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.DM_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.DM_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.DM_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.DM_VmovupdOp

        res_vec = rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                destination=destination,
            ),
            insertion_point=insert_point,
        ).destination
        res_vec = cast(ir.SSAValue[x86.registers.AVX512RegisterType], res_vec)
    else:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.DMK_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.DMK_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.DMK_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.DMK_VmovupdOp

        res_vec = rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                destination=destination,
                mask_reg=mask,
                z=True,
            ),
            insertion_point=insert_point,
        ).destination
    return res_vec


def load_c(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    m_blocking: int,
    n_blocking: int,
    dest_type: type[x86.registers.AVX512RegisterType],
    datatype: builtin.Float32Type | builtin.Float64Type,
    *,
    ldc: int,
    vector_length: int,
    vector_reg_count: int,
    use_masking_a_c: bool,
    aligned_c: bool,
    c_val: ir.SSAValue[x86.registers.GeneralRegisterType],
    mask_k1: ir.SSAValue[x86.registers.AVX512MaskRegisterType] | None = None,
) -> tuple[ir.SSAValue[x86.registers.AVX512RegisterType], ...]:
    result: list[ir.SSAValue[x86.registers.AVX512RegisterType]] = []
    # register blocking counter in n
    n = 0
    # register blocking counter in m
    m = 0

    datatype_size_out = datatype.size

    # deriving register blocking from kernel config
    m_blocking = (
        m_blocking // vector_length
        if (m_blocking % vector_length == 0)
        else (m_blocking // vector_length) + 1
    )
    # start register of accumulator
    vec_reg_acc_start = vector_reg_count - n_blocking * m_blocking

    # load C accumulator
    # Beta=1
    # adding to C, so let's load C
    for n in range(n_blocking):
        for m in range(m_blocking):
            c_vec_reg = dest_type.from_index(vec_reg_acc_start + m + (m_blocking * n))
            last_iteration = m == m_blocking - 1
            use_masking = use_masking_a_c and last_iteration

            displacement = (n * ldc + m * vector_length) * datatype_size_out
            if use_masking:
                assert mask_k1 is not None

            res_vec = load_vector(
                rewriter,
                insert_point,
                datatype,
                c_val,
                displacement,
                mask_k1 if use_masking else None,
                aligned=aligned_c,
                destination=c_vec_reg,
            )

            result.append(res_vec)

    return tuple(result)


def store_vector(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    datatype: builtin.Float32Type | builtin.Float64Type,
    pointer: ir.SSAValue[x86.registers.GeneralRegisterType],
    offset: int,
    source: ir.SSAValue[x86.registers.AVX512RegisterType],
    mask: ir.SSAValue[x86.registers.AVX512MaskRegisterType] | None = None,
    *,
    aligned: bool,
) -> None:
    if mask is None:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.MS_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.MS_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.MS_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.MS_VmovupdOp

        rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                source=source,
            ),
            insertion_point=insert_point,
        )
    else:
        match (datatype, aligned):
            case (builtin.Float32Type(), True):
                move_op_type = x86.ops.MSK_VmovapsOp
            case (builtin.Float32Type(), False):
                move_op_type = x86.ops.MSK_VmovupsOp
            case (builtin.Float64Type(), True):
                move_op_type = x86.ops.MSK_VmovapdOp
            case (builtin.Float64Type(), False):
                move_op_type = x86.ops.MSK_VmovupdOp

        rewriter.insert(
            move_op_type(
                memory=pointer,
                memory_offset=offset,
                source=source,
                mask_reg=mask,
            ),
            insertion_point=insert_point,
        )


def store_c(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    m_blocking: int,
    n_blocking: int,
    datatype: builtin.Float32Type | builtin.Float64Type,
    *,
    ldc: int,
    vector_length: int,
    use_masking_a_c: bool,
    aligned_c: bool,
    c_val: ir.SSAValue[x86.registers.GeneralRegisterType],
    acc_vectors: tuple[ir.SSAValue[x86.registers.AVX512RegisterType], ...],
    mask_k1: ir.SSAValue[x86.registers.AVX512MaskRegisterType] | None = None,
) -> None:
    datatype_size_out = datatype.size

    m_blocking = (
        m_blocking // vector_length
        if m_blocking % vector_length == 0
        else m_blocking // vector_length + 1
    )
    assert len(acc_vectors) == n_blocking * m_blocking

    for n in range(n_blocking):
        for m in range(m_blocking):
            accumulator = acc_vectors[m + m_blocking * n]
            use_masking = use_masking_a_c and m == m_blocking - 1
            displacement = (n * ldc + m * vector_length) * datatype_size_out

            if use_masking:
                assert mask_k1 is not None

            store_vector(
                rewriter,
                insert_point,
                datatype,
                c_val,
                displacement,
                accumulator,
                mask_k1 if use_masking else None,
                aligned=aligned_c,
            )
