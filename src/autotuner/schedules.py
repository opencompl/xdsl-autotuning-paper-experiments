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


def _matmul_like(
    op: MatmulOp,
    *,
    a: ir.SSAValue | None = None,
    b: ir.SSAValue | None = None,
    c: ir.SSAValue | None = None,
    rbp: ir.SSAValue | None = None,
    rsp: ir.SSAValue | None = None,
    ins: Sequence[ir.SSAValue] | None = None,
    outs: Sequence[ir.SSAValue] | None = None,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    aligned_a: bool | None = None,
    aligned_c: bool | None = None,
    iterator: MatmulIterator | None = None,
) -> MatmulOp:
    """Copy a matmul while replacing only explicitly supplied fields."""
    return MatmulOp(
        op.a if a is None else a,
        op.b if b is None else b,
        op.c if c is None else c,
        op.rbp if rbp is None else rbp,
        op.rsp if rsp is None else rsp,
        op.ins if ins is None else ins,
        op.outs if outs is None else outs,
        m=op.m.value.data if m is None else m,
        n=op.n.value.data if n is None else n,
        k=op.k.value.data if k is None else k,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a) if aligned_a is None else aligned_a,
        aligned_c=bool(op.aligned_c) if aligned_c is None else aligned_c,
        iterator=(MatmulIterator(op.iterator.data) if iterator is None else iterator),
    )


def _matmul_reg_like(
    op: MatmulRegOp,
    *,
    a: ir.SSAValue | None = None,
    b: ir.SSAValue | None = None,
    rbp: ir.SSAValue | None = None,
    rsp: ir.SSAValue | None = None,
    ins: Sequence[ir.SSAValue] | None = None,
    outs: Sequence[ir.SSAValue] | None = None,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    aligned_a: bool | None = None,
    iterator: MatmulIterator | None = None,
) -> MatmulRegOp:
    """Copy a matmul_reg while replacing only explicitly supplied fields."""
    return MatmulRegOp(
        op.a if a is None else a,
        op.b if b is None else b,
        op.rbp if rbp is None else rbp,
        op.rsp if rsp is None else rsp,
        op.ins if ins is None else ins,
        op.outs if outs is None else outs,
        m=op.m.value.data if m is None else m,
        n=op.n.value.data if n is None else n,
        k=op.k.value.data if k is None else k,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a) if aligned_a is None else aligned_a,
        iterator=(MatmulIterator(op.iterator.data) if iterator is None else iterator),
    )


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


def _normalize_pointers(
    pointers: Sequence[ir.SSAValue],
    desired_offsets: Sequence[int],
    actual_offsets: Sequence[int],
    zero_op_types: Sequence[type[x86.ops.RI_AddOp] | type[x86.ops.RI_SubOp] | None],
) -> tuple[tuple[ir.SSAValue, ...], tuple[ir.Operation, ...]]:
    """Normalize pointer results in operand order."""
    normalized: list[ir.SSAValue] = []
    adjustments: list[ir.Operation] = []
    for pointer, desired, actual, zero_op_type in zip(
        pointers, desired_offsets, actual_offsets, zero_op_types, strict=True
    ):
        offset = desired - actual
        if offset == 0 and zero_op_type is None:
            normalized.append(pointer)
            continue
        pointer = ir.SSAValue.get(pointer, type=x86.registers.GeneralRegisterType)
        op_type = (
            zero_op_type
            if offset == 0
            else x86.ops.RI_SubOp
            if offset < 0
            else x86.ops.RI_AddOp
        )
        assert op_type is not None
        adjustment = op_type(pointer, abs(offset), register_out=pointer.type)
        adjustments.append(adjustment)
        normalized.append(adjustment.register_out)
    return tuple(normalized), tuple(adjustments)


def _normalize_matmul_results(
    op: MatmulOp,
    results: Sequence[ir.SSAValue],
    actual_iterator: MatmulIterator,
) -> tuple[tuple[ir.SSAValue, ...], tuple[ir.Operation, ...]]:
    pointers, adjustments = _normalize_pointers(
        results[:3],
        _matmul_pointer_offsets_abc(op, MatmulIterator(op.iterator.data)),
        _matmul_pointer_offsets_abc(op, actual_iterator),
        (
            (x86.ops.RI_AddOp,) * 3
            if op.iterator.data != actual_iterator
            else (None,) * 3
        ),
    )
    return (*pointers, *results[3:]), adjustments


def _normalize_matmul_reg_results(
    op: MatmulRegOp,
    results: Sequence[ir.SSAValue],
    actual_iterator: MatmulIterator,
) -> tuple[tuple[ir.SSAValue, ...], tuple[ir.Operation, ...]]:
    pointers, adjustments = _normalize_pointers(
        results[:2],
        matmul_reg_pointer_offsets(op),
        matmul_reg_pointer_offsets(op, actual_iterator),
        (
            (x86.ops.RI_SubOp, None)
            if op.iterator.data == MatmulIterator.M
            and actual_iterator == MatmulIterator.K
            else (None, None)
        ),
    )
    return (*pointers, *results[2:]), adjustments


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

    matmul = _matmul_like(op, iterator=iterator)
    results, adjustments = _normalize_matmul_results(op, matmul.results, iterator)
    rewriter.replace(op, (matmul, *adjustments), results)
    return matmul


def set_matmul_reg_iterator(
    rewriter: PatternRewriter, op: MatmulRegOp, iterator: MatmulIterator
) -> MatmulRegOp:
    """Change a matmul_reg iterator while preserving its result contract."""
    if op.iterator.data == iterator:
        return op
    matmul = _matmul_reg_like(op, iterator=iterator)
    results, adjustments = _normalize_matmul_reg_results(op, matmul.results, iterator)
    rewriter.replace(op, (matmul, *adjustments), results)
    return matmul


def split_matmul(
    rewriter: PatternRewriter,
    op: MatmulOp,
    dimension: MatmulIterator,
    first_size: int,
) -> tuple[MatmulOp, MatmulOp]:
    """Split a matmul along M or N and preserve its external pointer contract."""
    assert dimension in (MatmulIterator.M, MatmulIterator.N)
    extent = op.m.value.data if dimension == MatmulIterator.M else op.n.value.data
    assert 0 < first_size < extent, (
        f"Invalid {dimension} split {first_size} for extent {extent}"
    )
    vector_length = 512 // op.datatype.bitwidth
    second_aligned_a = bool(op.aligned_a) and (
        dimension != MatmulIterator.M or first_size % vector_length == 0
    )
    second_aligned_c = bool(op.aligned_c) and (
        dimension != MatmulIterator.M or first_size % vector_length == 0
    )
    first = _matmul_like(
        op,
        m=first_size if dimension == MatmulIterator.M else None,
        n=first_size if dimension == MatmulIterator.N else None,
        iterator=dimension,
    )
    second = _matmul_like(
        op,
        a=first.a_out,
        b=first.b_out,
        c=first.c_out,
        rbp=first.rbp_out,
        rsp=first.rsp_out,
        outs=first.out_results,
        m=extent - first_size if dimension == MatmulIterator.M else None,
        n=extent - first_size if dimension == MatmulIterator.N else None,
        aligned_a=second_aligned_a,
        aligned_c=second_aligned_c,
        iterator=dimension,
    )
    results, adjustments = _normalize_matmul_results(op, second.results, dimension)
    rewriter.replace(op, (first, second, *adjustments), results)
    return first, second


def loop_matmul(
    rewriter: PatternRewriter,
    op: MatmulOp,
    dimension: MatmulIterator,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    *,
    lower_bound: int | None = None,
) -> MatmulOp:
    """Materialize an exact M or N tiled loop and preserve result semantics."""
    assert dimension in (MatmulIterator.M, MatmulIterator.N)
    extent = op.m.value.data if dimension == MatmulIterator.M else op.n.value.data
    assert 0 < tile_size <= extent and extent % tile_size == 0

    if lower_bound is None:
        lower_bound = 0
    init = x86.ops.DI_MovOp(lower_bound, destination=loop_register)
    inputs = (op.a, op.b, op.c, op.rbp, op.rsp, *op.outs)
    body = ir.Block(arg_types=(init.destination.type, *(v.type for v in inputs)))
    a, b, c, rbp, rsp, *outs = body.args[1:]
    tiled = _matmul_like(
        op,
        a=a,
        b=b,
        c=c,
        rbp=rbp,
        rsp=rsp,
        outs=outs,
        m=tile_size if dimension == MatmulIterator.M else None,
        n=tile_size if dimension == MatmulIterator.N else None,
        iterator=dimension,
    )
    body.add_ops((tiled, x86_scf.YieldOp(*tiled.results)))
    loop = x86_scf.ForOp(
        init.destination,
        builtin.IntegerAttr(lower_bound + extent, x86.ops.si32),
        builtin.IntegerAttr(tile_size, x86.ops.si32),
        inputs,
        body,
    )
    results, adjustments = _normalize_matmul_results(
        op, tuple(loop.results[1:]), dimension
    )
    rewriter.replace(op, (init, loop, *adjustments), results)
    return tiled


def tile_matmul(
    rewriter: PatternRewriter,
    op: MatmulOp,
    dimension: MatmulIterator,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulOp, MatmulOp | None]:
    """Tile M or N with an exact loop and an optional direct remainder."""
    assert dimension in (MatmulIterator.M, MatmulIterator.N)
    extent = op.m.value.data if dimension == MatmulIterator.M else op.n.value.data
    assert 0 < tile_size <= extent
    if tile_size == extent:
        return loop_matmul(rewriter, op, dimension, tile_size, loop_register), None

    blocked_end = extent // tile_size * tile_size
    if blocked_end != extent:
        main, remainder = split_matmul(rewriter, op, dimension, blocked_end)
    else:
        main, remainder = op, None
    tiled = loop_matmul(rewriter, main, dimension, tile_size, loop_register)
    return tiled, remainder


def split_n(
    rewriter: PatternRewriter, op: MatmulOp, n_split: int
) -> tuple[MatmulOp, MatmulOp]:
    """Compatibility wrapper for the existing XSMM N-tiling pass."""
    assert op.iterator.data == MatmulIterator.N
    return split_matmul(rewriter, op, MatmulIterator.N, n_split)


def tile_n(
    rewriter: PatternRewriter,
    op: MatmulOp,
    n_tile: int,
    nloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> MatmulOp:
    """Compatibility wrapper for the existing XSMM N-tiling pass."""
    assert op.iterator.data == MatmulIterator.N
    return loop_matmul(rewriter, op, MatmulIterator.N, n_tile, nloop_register)


def matmul_m_to_reg(rewriter: PatternRewriter, op: MatmulOp) -> MatmulRegOp:
    assert op.iterator.data == MatmulIterator.M
    assert len(op.ins) <= 1, "matmul_m_to_reg supports at most one mask in"
    assert not op.outs, "matmul_m_to_reg does not yet support outs"
    m_blocking = op.m.value.data
    n_blocking = op.n.value.data
    k = op.k.value.data
    lda = op.lda.value.data
    vector_length = 512 // op.datatype.bitwidth
    insert_point = InsertPoint.before(op)

    if op.ins:
        mask = ir.SSAValue.get(op.ins[0], type=x86.registers.AVX512MaskRegisterType)
        ins = (mask,)
    else:
        mask = None
        ins = ()
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
            ins,
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
        ),
    )
    return matmul_reg


def split_matmul_reg(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    dimension: MatmulIterator,
    first_size: int,
) -> tuple[MatmulRegOp, MatmulRegOp]:
    """Split a matmul_reg reduction and preserve its pointer contract."""
    assert dimension == MatmulIterator.K
    extent = op.k.value.data
    assert 0 < first_size < extent, f"Invalid K split {first_size} for extent {extent}"
    first = _matmul_reg_like(op, k=first_size, iterator=dimension)
    second = _matmul_reg_like(
        op,
        a=first.a_out,
        b=first.b_out,
        rbp=first.rbp_out,
        rsp=first.rsp_out,
        outs=first.out_results,
        k=extent - first_size,
        iterator=dimension,
    )
    results, adjustments = _normalize_matmul_reg_results(op, second.results, dimension)
    rewriter.replace(op, (first, second, *adjustments), results)
    return first, second


def loop_matmul_reg(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    dimension: MatmulIterator,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> MatmulRegOp:
    """Materialize an exact K loop and preserve the original result contract."""
    assert dimension == MatmulIterator.K
    extent = op.k.value.data
    assert 0 < tile_size <= extent and extent % tile_size == 0

    init = x86.ops.DI_MovOp(0, destination=loop_register)
    inputs = (op.a, op.b, op.rbp, op.rsp, *op.outs)
    body = ir.Block(arg_types=(init.destination.type, *(v.type for v in inputs)))
    a, b, rbp, rsp, *outs = body.args[1:]
    tiled = _matmul_reg_like(
        op,
        a=a,
        b=b,
        rbp=rbp,
        rsp=rsp,
        outs=outs,
        k=tile_size,
        iterator=dimension,
    )
    body.add_ops((tiled, x86_scf.YieldOp(*tiled.results)))
    loop = x86_scf.ForOp(
        init.destination,
        builtin.IntegerAttr(extent, x86.ops.si32),
        builtin.IntegerAttr(tile_size, x86.ops.si32),
        inputs,
        body,
    )
    results, adjustments = _normalize_matmul_reg_results(
        op, tuple(loop.results[1:]), dimension
    )
    rewriter.replace(op, (init, loop, *adjustments), results)
    return tiled


def tile_matmul_reg(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    dimension: MatmulIterator,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulRegOp, MatmulRegOp | None]:
    """Tile matmul_reg K with an exact loop and an optional direct remainder."""
    assert dimension == MatmulIterator.K
    extent = op.k.value.data
    assert 0 < tile_size <= extent
    if tile_size == extent:
        return loop_matmul_reg(rewriter, op, dimension, tile_size, loop_register), None

    blocked_end = extent // tile_size * tile_size
    if blocked_end != extent:
        main, remainder = split_matmul_reg(rewriter, op, dimension, blocked_end)
    else:
        main, remainder = op, None
    tiled = loop_matmul_reg(rewriter, main, dimension, tile_size, loop_register)
    return tiled, remainder


def tile_k(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    k_tile: int,
    kloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulRegOp, MatmulRegOp | None]:
    """Compatibility wrapper for the existing XSMM K-tiling pass."""
    return tile_matmul_reg(rewriter, op, MatmulIterator.K, k_tile, kloop_register)


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


def _attach_avx512_mask(
    rewriter: PatternRewriter,
    op: MatmulOp,
    *,
    active_elements: int,
    vector_length: int,
    mask_tmp_reg: x86.registers.GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> MatmulOp:
    """Attach one loop-invariant AVX-512 tail mask as a read-only input."""
    assert not op.ins
    mask = _initialize_avx512_mask(
        rewriter,
        InsertPoint.before(op),
        mask_tmp_reg,
        mask_reg,
        vector_length - active_elements % vector_length,
        op.datatype,
    )
    masked = _matmul_like(op, ins=(mask,))
    rewriter.replace(op, (masked,), masked.results)
    return masked


def tile_m(
    rewriter: PatternRewriter,
    op: MatmulOp,
    *,
    m_blocking: int,
    mloop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_tmp_reg: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> tuple[MatmulOp, MatmulOp | None]:
    """Tile M structurally, attaching AVX-512 masks as read-only inputs."""
    match op.datatype:
        case builtin.Float64Type():
            vector_length = 8
        case builtin.Float32Type():
            vector_length = 16

    assert not op.ins
    m = op.m.value.data
    assert 0 < m_blocking <= m
    blocked_end = m // m_blocking * m_blocking
    if remainder := m % m_blocking:
        main, remainder_matmul = split_matmul(
            rewriter, op, MatmulIterator.M, blocked_end
        )
    else:
        main = op
        remainder_matmul = None

    if m_blocking % vector_length:
        main = _attach_avx512_mask(
            rewriter,
            main,
            active_elements=m_blocking,
            vector_length=vector_length,
            mask_tmp_reg=mask_tmp_reg,
            mask_reg=mask_reg,
        )
    tiled_matmul, unexpected_remainder = tile_matmul(
        rewriter,
        main,
        MatmulIterator.M,
        m_blocking,
        mloop_register,
    )
    assert unexpected_remainder is None

    if remainder_matmul is not None:
        if remainder % vector_length:
            remainder_matmul = _attach_avx512_mask(
                rewriter,
                remainder_matmul,
                active_elements=remainder,
                vector_length=vector_length,
                mask_tmp_reg=mask_tmp_reg,
                mask_reg=mask_reg,
            )
        remainder_matmul = loop_matmul(
            rewriter,
            remainder_matmul,
            MatmulIterator.M,
            remainder,
            mloop_register,
            lower_bound=blocked_end,
        )

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
