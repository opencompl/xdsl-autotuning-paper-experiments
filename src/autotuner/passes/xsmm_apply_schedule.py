"""Apply a concrete XSMM-compatible schedule and lower it to x86."""

from dataclasses import dataclass
from typing import cast

from xdsl import ir
from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulIterator, MatmulOp, MatmulRegOp
from autotuner.nano_kernel import GemmDescriptor, ISAInfo, NanoKernel
from autotuner.schedules import (
    loop_matmul,
    set_matmul_iterator,
    split_matmul,
    tile_matmul,
    tile_matmul_reg,
)
from autotuner.strategy import get_xsmm_strategy
from autotuner.tiling import TilingStrategy, compute_tiling_strategy

K_BLOCKING = 4
K_THRESHOLD = 23


def load_mask(
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
    gp_reg_tmp: x86.registers.GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
    mask_count: int,
    datatype: builtin.Float64Type | builtin.Float32Type,
) -> ir.SSAValue[x86.registers.AVX512MaskRegisterType]:
    """Materialize the AVX-512 tail mask expected by the XSMM kernels."""
    match datatype:
        case builtin.Float64Type():
            mask = 0xFF
            op_type = x86.ops.KS_KMovBOp
        case builtin.Float32Type():
            mask = 0xFFFF
            op_type = x86.ops.KS_KMovWOp

    mask = mask >> mask_count
    mask_tmp_val = rewriter.insert(
        x86.ops.DI_MovOp(mask, destination=gp_reg_tmp), insertion_point=insert_point
    ).destination
    return rewriter.insert(
        op_type(mask_tmp_val, destination=mask_reg), insertion_point=insert_point
    ).destination


def _attach_mask(
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
    mask = load_mask(
        rewriter,
        InsertPoint.before(op),
        mask_tmp_reg,
        mask_reg,
        vector_length - active_elements % vector_length,
        op.datatype,
    )
    masked = MatmulOp(
        op.a,
        op.b,
        op.c,
        op.rbp,
        op.rsp,
        (mask,),
        op.outs,
        m=op.m.value.data,
        n=op.n.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        iterator=MatmulIterator(op.iterator.data),
    )
    rewriter.replace(op, (masked,), masked.results)
    return masked


def tile_m(
    rewriter: PatternRewriter,
    op: MatmulOp,
    *,
    m_blocking: int,
    mloop_register: x86.registers.GeneralRegisterType,
    mask_tmp_reg: x86.registers.GeneralRegisterType,
    mask_reg: x86.registers.AVX512MaskRegisterType,
) -> tuple[MatmulOp, MatmulOp | None]:
    """Apply the XSMM M-tiling and masked-remainder policy."""
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
        main = _attach_mask(
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
            remainder_matmul = _attach_mask(
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


def _tile_n_m(
    rewriter: PatternRewriter,
    op: MatmulOp,
    strategy: TilingStrategy,
    *,
    nloop_register: x86.registers.GeneralRegisterType,
    mloop_register: x86.registers.GeneralRegisterType,
    mask_tmp_register: x86.registers.GeneralRegisterType,
    mask_register: x86.registers.AVX512MaskRegisterType,
) -> list[MatmulOp]:
    if len(strategy.n_ranges) == 2:
        first_range, second_range = strategy.n_ranges
        first, second = split_matmul(rewriter, op, MatmulIterator.N, first_range.extent)
        n_ranges = (
            (first, first_range.tile_size),
            (second, second_range.tile_size),
        )
    else:
        assert len(strategy.n_ranges) == 1
        n_ranges = ((op, strategy.n_ranges[0].tile_size),)

    result: list[MatmulOp] = []
    for n_range, n_tile in n_ranges:
        tiled_n = loop_matmul(
            rewriter,
            n_range,
            MatmulIterator.N,
            n_tile,
            nloop_register,
        )
        matmul_m = set_matmul_iterator(rewriter, tiled_n, MatmulIterator.M)
        tiled_m, remainder_m = tile_m(
            rewriter,
            matmul_m,
            m_blocking=strategy.m_tile_size,
            mloop_register=mloop_register,
            mask_tmp_reg=mask_tmp_register,
            mask_reg=mask_register,
        )
        result.append(set_matmul_iterator(rewriter, tiled_m, MatmulIterator.K))
        if remainder_m is not None:
            result.append(set_matmul_iterator(rewriter, remainder_m, MatmulIterator.K))
    return result


def _matmul_k_to_reg(rewriter: PatternRewriter, op: MatmulOp) -> MatmulRegOp:
    if (
        len(op.ins) > 1
        or op.outs
        or (
            op.ins
            and not isinstance(op.ins[0].type, x86.registers.AVX512MaskRegisterType)
        )
    ):
        raise PassFailedException(
            "xsmm-apply-schedule currently supports only one mask in"
        )
    m_blocking = op.m.value.data
    vector_length = 512 // op.datatype.bitwidth
    if m_blocking % vector_length and not op.ins:
        raise PassFailedException(
            "xsmm-apply-schedule requires a mask for a partial M vector"
        )

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
        op.n.value.data,
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
            n=op.n.value.data,
            k=op.k.value.data,
            lda=op.lda.value.data,
            ldb=op.ldb.value.data,
            datatype=op.datatype,
            aligned_a=bool(op.aligned_a),
            iterator=MatmulIterator(op.iterator.data),
        ),
        insertion_point=insert_point,
    )
    store_c(
        rewriter,
        insert_point,
        m_blocking,
        op.n.value.data,
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
    rewriter.replace(
        op,
        [],
        (
            matmul_reg.a_out,
            matmul_reg.b_out,
            op.c,
            matmul_reg.rbp_out,
            matmul_reg.rsp_out,
        ),
    )
    return matmul_reg


def _tile_k(
    rewriter: PatternRewriter,
    op: MatmulRegOp,
    *,
    kloop_register: x86.registers.GeneralRegisterType,
) -> tuple[MatmulRegOp, ...]:
    if op.k.value.data <= K_THRESHOLD:
        return (op,)
    tiled, remainder = tile_matmul_reg(
        rewriter,
        op,
        MatmulIterator.K,
        K_BLOCKING,
        kloop_register,
    )
    return (tiled,) if remainder is None else (tiled, remainder)


@dataclass
class ApplySchedulePattern(RewritePattern):
    """Apply the complete XSMM schedule to an N-iterating matmul."""

    isa_info: ISAInfo
    nano_kernel: NanoKernel
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        if op.iterator.data != MatmulIterator.N:
            return
        descriptor = GemmDescriptor(
            m=op.m.value.data,
            n=op.n.value.data,
            k=op.k.value.data,
            lda=op.lda.value.data,
            ldb=op.ldb.value.data,
            ldc=op.ldc.value.data,
            datatype=op.datatype,
            aligned_a=bool(op.aligned_a),
            aligned_c=bool(op.aligned_c),
        )
        try:
            strategy = compute_tiling_strategy(
                descriptor, self.isa_info, self.nano_kernel
            )
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        if self.disable_regalloc:
            nloop_register = x86.registers.UNALLOCATED_REG64
            mloop_register = x86.registers.UNALLOCATED_REG64
            kloop_register = x86.registers.UNALLOCATED_REG64
            mask_tmp_reg = x86.registers.UNALLOCATED_REG64
        else:
            nloop_register = x86.registers.R11
            mloop_register = x86.registers.R10
            kloop_register = x86.registers.R12
            mask_tmp_reg = x86.registers.R15

        for tiled_m in _tile_n_m(
            rewriter,
            op,
            strategy,
            nloop_register=nloop_register,
            mloop_register=mloop_register,
            mask_tmp_register=mask_tmp_reg,
            mask_register=x86.registers.K1,
        ):
            matmul_reg = _matmul_k_to_reg(rewriter, tiled_m)
            for tiled_k in _tile_k(rewriter, matmul_reg, kloop_register=kloop_register):
                self.nano_kernel.rewrite(
                    rewriter,
                    tiled_k,
                    self.isa_info,
                    disable_regalloc=self.disable_regalloc,
                )


@dataclass(frozen=True)
class XsmmApplySchedulePass(ModulePass):
    """Apply an XSMM schedule and lower its operations to x86 instructions."""

    name = "xsmm-apply-schedule"

    strategy: str = "libxsmm-skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            strategy = get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            ApplySchedulePattern(
                strategy.isa_info, strategy.nano_kernel, self.disable_regalloc
            ),
            apply_recursively=False,
        ).rewrite_module(op)
