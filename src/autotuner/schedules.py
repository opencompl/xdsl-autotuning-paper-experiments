from collections.abc import Sequence
from typing import cast
from xdsl import ir
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import (
    MatmulIterator,
    MatmulOp,
    MatmulRegOp,
    matmul_reg_pointer_offsets,
)


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


def _matmul_pointer_offsets_abc(
    op: MatmulOp, iterator: MatmulIterator
) -> tuple[int, int, int]:
    """Return the A, B, and C pointer advances in bytes for an iterator."""
    element_size = op.datatype.bitwidth // 8
    match iterator:
        case MatmulIterator.NONE:
            return 0, 0, 0
        case MatmulIterator.M:
            m_bytes = op.m.value.data * element_size
            return m_bytes, 0, m_bytes
        case MatmulIterator.N:
            n = op.n.value.data
            return (
                0,
                n * op.ldb.value.data * element_size,
                n * op.ldc.value.data * element_size,
            )
        case MatmulIterator.K:
            return (
                op.k.value.data * op.lda.value.data * element_size,
                op.k.value.data * element_size,
                0,
            )


def set_matmul_iterator(
    rewriter: PatternRewriter, op: MatmulOp, iterator: MatmulIterator
) -> MatmulOp:
    """Change a matmul iterator without changing its externally visible results.

    The replacement matmul has the requested pointer-result contract. Pointer
    adjustments after it compensate for the difference from the old contract,
    so existing users of the operation's results remain valid.
    """
    old_iterator = MatmulIterator(op.iterator.data)
    if old_iterator == iterator:
        return op

    matmul = MatmulOp(
        op.a,
        op.b,
        op.c,
        op.rbp,
        op.rsp,
        op.ins,
        op.outs,
        m=op.m.value.data,
        n_start=op.n_start.value.data,
        n=op.n.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=iterator,
    )

    # Preserve the existing C, B, A adjustment order and zero adjustments in
    # generated code, so changing the iterator does not change the schedule.
    old_offsets_cba = reversed(_matmul_pointer_offsets_abc(op, old_iterator))
    new_offsets_cba = reversed(_matmul_pointer_offsets_abc(op, iterator))
    offsets_cba = tuple(
        old_offset - new_offset
        for old_offset, new_offset in zip(old_offsets_cba, new_offsets_cba, strict=True)
    )
    pointer_results_cba = [matmul.c_out, matmul.b_out, matmul.a_out]
    offset_ops = [
        (x86.ops.RI_SubOp if offset < 0 else x86.ops.RI_AddOp)(
            pointer_result,
            abs(offset),
            register_out=pointer_result.type,
        )
        for offset, pointer_result in zip(offsets_cba, pointer_results_cba)
    ]

    rewriter.replace(
        op,
        (matmul, *offset_ops),
        (
            *(offset_op.register_out for offset_op in reversed(offset_ops)),
            matmul.rbp_out,
            matmul.rsp_out,
            *matmul.out_results,
        ),
    )
    return matmul


def matmul_m_to_reg(rewriter: PatternRewriter, op: MatmulOp) -> MatmulRegOp:
    assert op.iterator.data == MatmulIterator.M
    assert not op.ins, "matmul_m_to_reg does not yet support ins"
    assert len(op.outs) <= 1, "matmul_m_to_reg supports at most one mask out"
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
    matmul_reg = rewriter.insert(
        MatmulRegOp(
            op.a,
            op.b,
            op.rbp,
            op.rsp,
            () if mask is None else (mask,),
            accumulators,
            m=m_blocking,
            n=n_blocking,
            k=k,
            lda=lda,
            ldb=op.ldb.value.data,
            datatype=op.datatype,
            aligned_a=bool(op.aligned_a),
            iterator=MatmulIterator(op.iterator.data),
        ),
        insertion_point=insert_point,
    )

    element_size = op.datatype.size
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
        c_val=ir.SSAValue.get(op.c, type=x86.registers.GeneralRegisterType),
        acc_vectors=tuple(
            ir.SSAValue.get(accumulator, type=x86.registers.AVX512RegisterType)
            for accumulator in matmul_reg.out_results
        ),
        mask_k1=mask,
    )

    c_out = rewriter.insert(
        x86.ops.RI_AddOp(
            op.c,
            m_blocking * element_size,
            register_out=op.c_out.type,
        ),
        insertion_point=insert_point,
    ).register_out
    rewriter.replace(
        op,
        [],
        (
            matmul_reg.a_out,
            matmul_reg.b_out,
            c_out,
            matmul_reg.rbp_out,
            matmul_reg.rsp_out,
            *op.outs,
        ),
    )
    return matmul_reg


def _matmul_reg_with_inputs(
    op: MatmulRegOp, inputs: Sequence[ir.SSAValue], k: int
) -> MatmulRegOp:
    a, b, rbp, rsp, *outs = inputs
    return MatmulRegOp(
        a,
        b,
        rbp,
        rsp,
        op.ins,
        outs,
        m=op.m.value.data,
        n=op.n.value.data,
        k=k,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        iterator=MatmulIterator.K,
    )


def _offset_pointer(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    pointer: ir.SSAValue,
    byte_offset: int,
    *,
    emit_zero_sub: bool = False,
) -> ir.SSAValue[x86.registers.GeneralRegisterType]:
    pointer = ir.SSAValue.get(pointer, type=x86.registers.GeneralRegisterType)
    if byte_offset == 0 and not emit_zero_sub:
        return pointer
    op_type = x86.ops.RI_SubOp if byte_offset <= 0 else x86.ops.RI_AddOp
    return rewriter.insert(
        op_type(pointer, abs(byte_offset)),
        insertion_point=insert_point,
    ).register_out


def tile_k(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    k_tile: int,
    kloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulRegOp, MatmulRegOp | None]:
    """Tile K, inserting an exact loop and an optional remainder matmul.

    Each tiled body uses the K iterator to advance through its reduction slice.
    The combined K pointer results are normalized once after the loop to the
    original operation's iterator contract.

    Returns the matmul in the loop body and the optional remainder matmul.
    """
    k = op.k.value.data
    assert 0 < k_tile <= k, f"Invalid K tile {k_tile} for K extent {k}"

    insert_point = InsertPoint.before(op)
    blocked_end = k // k_tile * k_tile
    k_init = rewriter.insert(
        x86.ops.DI_MovOp(0, destination=kloop_register),
        insertion_point=insert_point,
    )

    inputs = (op.a, op.b, op.rbp, op.rsp, *op.outs)
    body = ir.Block(
        arg_types=(k_init.destination.type, *(value.type for value in inputs))
    )
    tiled_matmul = _matmul_reg_with_inputs(op, body.args[1:], k_tile)
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
            _matmul_reg_with_inputs(op, loop_results, remainder),
            insertion_point=insert_point,
        )
        results = tuple(remainder_matmul.results)
    else:
        remainder_matmul = None
        results = loop_results

    desired_a_offset, desired_b_offset = matmul_reg_pointer_offsets(op)
    k_a_offset, k_b_offset = matmul_reg_pointer_offsets(op, MatmulIterator.K)
    a, b, *rest = results
    # Keep the B-before-A adjustment order used by the existing schedule.
    b = _offset_pointer(rewriter, insert_point, b, desired_b_offset - k_b_offset)
    # Delay the independent A normalization until its first same-block use so
    # intervening C stores retain their established instruction order.
    a_insert_point = insert_point
    if (
        a_user := op.a_out.get_user_of_unique_use()
    ) is not None and a_user.parent_block() is op.parent_block():
        a_insert_point = InsertPoint.before(a_user)
    a = _offset_pointer(
        rewriter,
        a_insert_point,
        a,
        desired_a_offset - k_a_offset,
        emit_zero_sub=op.iterator.data == MatmulIterator.M,
    )

    rewriter.replace(op, [], (a, b, *rest))
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
