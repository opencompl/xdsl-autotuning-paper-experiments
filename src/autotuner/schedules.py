"""Generic, semantics-preserving transformations for XSMM operations."""

from collections.abc import Sequence

from xdsl import ir
from xdsl.dialects import builtin, x86, x86_scf
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from autotuner.dialects.xsmm import (
    MatmulIterator,
    MatmulOp,
    MatmulRegOp,
)
from autotuner.instructions import load_mask


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


def split_matmul(
    rewriter: PatternRewriter,
    op: MatmulOp,
    first_size: int,
) -> tuple[MatmulOp, MatmulOp]:
    """Split a matmul along its M or N iterator."""
    iterator = MatmulIterator(op.iterator.data)
    assert iterator in (MatmulIterator.M, MatmulIterator.N)
    extent = op.m.value.data if iterator == MatmulIterator.M else op.n.value.data
    assert 0 < first_size < extent, (
        f"Invalid {iterator} split {first_size} for extent {extent}"
    )
    vector_length = 512 // op.datatype.bitwidth
    second_aligned_a = bool(op.aligned_a) and (
        iterator != MatmulIterator.M or first_size % vector_length == 0
    )
    second_aligned_c = bool(op.aligned_c) and (
        iterator != MatmulIterator.M or first_size % vector_length == 0
    )
    first = _matmul_like(
        op,
        m=first_size if iterator == MatmulIterator.M else None,
        n=first_size if iterator == MatmulIterator.N else None,
    )
    second = _matmul_like(
        op,
        a=first.a_out,
        b=first.b_out,
        c=first.c_out,
        rbp=first.rbp_out,
        rsp=first.rsp_out,
        outs=first.out_results,
        m=extent - first_size if iterator == MatmulIterator.M else None,
        n=extent - first_size if iterator == MatmulIterator.N else None,
        aligned_a=second_aligned_a,
        aligned_c=second_aligned_c,
    )
    rewriter.replace(op, (first, second), second.results)
    return first, second


def loop_matmul(
    rewriter: PatternRewriter,
    op: MatmulOp,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> MatmulOp:
    """Materialize an exact loop along a matmul's M or N iterator."""
    iterator = MatmulIterator(op.iterator.data)
    assert iterator in (MatmulIterator.M, MatmulIterator.N)
    extent = op.m.value.data if iterator == MatmulIterator.M else op.n.value.data
    assert 0 < tile_size <= extent and extent % tile_size == 0

    init = x86.ops.DI_MovOp(0, destination=loop_register)
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
        m=tile_size if iterator == MatmulIterator.M else None,
        n=tile_size if iterator == MatmulIterator.N else None,
    )
    body.add_ops((tiled, x86_scf.YieldOp(*tiled.results)))
    loop = x86_scf.ForOp(
        init.destination,
        builtin.IntegerAttr(extent, x86.ops.si32),
        builtin.IntegerAttr(tile_size, x86.ops.si32),
        inputs,
        body,
    )
    rewriter.replace(op, (init, loop), tuple(loop.results[1:]))
    return tiled


def tile_matmul(
    rewriter: PatternRewriter,
    op: MatmulOp,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulOp, MatmulOp | None]:
    """Tile along a matmul's M or N iterator with an optional remainder."""
    iterator = MatmulIterator(op.iterator.data)
    assert iterator in (MatmulIterator.M, MatmulIterator.N)
    extent = op.m.value.data if iterator == MatmulIterator.M else op.n.value.data
    assert 0 < tile_size <= extent
    if tile_size == extent:
        return loop_matmul(rewriter, op, tile_size, loop_register), None

    blocked_end = extent // tile_size * tile_size
    if blocked_end != extent:
        main, remainder = split_matmul(rewriter, op, blocked_end)
    else:
        main, remainder = op, None
    tiled = loop_matmul(rewriter, main, tile_size, loop_register)
    return tiled, remainder


def split_matmul_reg(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    first_size: int,
) -> tuple[MatmulRegOp, MatmulRegOp]:
    """Split a matmul_reg K reduction and preserve its pointer contract."""
    extent = op.k.value.data
    assert 0 < first_size < extent, f"Invalid K split {first_size} for extent {extent}"
    first = _matmul_reg_like(op, k=first_size)
    second = _matmul_reg_like(
        op,
        a=first.a_out,
        b=first.b_out,
        rbp=first.rbp_out,
        rsp=first.rsp_out,
        outs=first.out_results,
        k=extent - first_size,
    )
    rewriter.replace(op, (first, second), second.results)
    return first, second


def loop_matmul_reg(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> MatmulRegOp:
    """Materialize an exact K loop and preserve the result contract."""
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
    )
    body.add_ops((tiled, x86_scf.YieldOp(*tiled.results)))
    loop = x86_scf.ForOp(
        init.destination,
        builtin.IntegerAttr(extent, x86.ops.si32),
        builtin.IntegerAttr(tile_size, x86.ops.si32),
        inputs,
        body,
    )
    rewriter.replace(op, (init, loop), tuple(loop.results[1:]))
    return tiled


def tile_matmul_reg(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    tile_size: int,
    loop_register: x86.registers.GeneralRegisterType = x86.registers.UNALLOCATED_REG64,
) -> tuple[MatmulRegOp, MatmulRegOp | None]:
    """Tile matmul_reg K with an exact loop and an optional direct remainder."""
    extent = op.k.value.data
    assert 0 < tile_size <= extent
    if tile_size == extent:
        return loop_matmul_reg(rewriter, op, tile_size, loop_register), None

    blocked_end = extent // tile_size * tile_size
    if blocked_end != extent:
        main, remainder = split_matmul_reg(rewriter, op, blocked_end)
    else:
        main, remainder = op, None
    tiled = loop_matmul_reg(rewriter, main, tile_size, loop_register)
    return tiled, remainder


def attach_mask(
    rewriter: PatternRewriter,
    op: MatmulOp,
    vectorize_dim: MatmulIterator,
    *,
    mask_tmp_reg: x86.registers.GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> MatmulOp:
    """
    Attach one loop-invariant AVX-512 tail mask as a read-only input, if necessary.
    """
    assert not op.ins
    match vectorize_dim:
        case MatmulIterator.N:
            active_elements = op.n.value.data
        case MatmulIterator.M:
            active_elements = op.m.value.data
        case MatmulIterator.K:
            active_elements = op.k.value.data
        case MatmulIterator.NONE:
            assert False

    match op.datatype:
        case builtin.Float64Type():
            vector_length = 8
        case builtin.Float32Type():
            vector_length = 16

    remainder = active_elements % vector_length
    if not remainder:
        return op

    mask = load_mask(
        rewriter,
        InsertPoint.before(op),
        mask_tmp_reg,
        mask_reg,
        vector_length - remainder,
        op.datatype,
    )
    masked = _matmul_like(op, ins=(mask,))
    rewriter.replace(op, (masked,), masked.results)
    return masked
