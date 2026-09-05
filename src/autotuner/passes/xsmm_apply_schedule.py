"""Apply a concrete XSMM-compatible schedule and lower it to x86."""

from dataclasses import dataclass

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
from autotuner.instructions import load_vector, store_vector
from autotuner.nano_kernel import GemmDescriptor, ISAInfo, NanoKernel
from autotuner.schedules import (
    attach_mask,
    loop_matmul,
    set_matmul_iterator,
    split_matmul,
    tile_matmul_reg,
)
from autotuner.skx_nano_kernel_utils import vector_register
from autotuner.strategy import get_xsmm_strategy
from autotuner.tiling import BlockingRange, TilingStrategy, compute_tiling_strategy


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
        first, second = split_matmul(rewriter, op, first_range.extent)
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
            n_tile,
            nloop_register,
        )
        matmul_m = set_matmul_iterator(rewriter, tiled_n, MatmulIterator.M)
        assert not matmul_m.ins
        m = matmul_m.m.value.data
        m_blocking = strategy.m_tile_size
        assert 0 < m_blocking <= m
        blocked_end = m // m_blocking * m_blocking
        if remainder := m % m_blocking:
            tiled_m, remainder_m = split_matmul(rewriter, matmul_m, blocked_end)
            remainder_m = attach_mask(
                rewriter,
                remainder_m,
                vectorize_dim=MatmulIterator.M,
                mask_tmp_reg=mask_tmp_register,
                mask_reg=mask_register,
            )
            remainder_m = loop_matmul(
                rewriter,
                remainder_m,
                remainder,
                mloop_register,
            )
        else:
            tiled_m = matmul_m
            remainder_m = None

        tiled_m = attach_mask(
            rewriter,
            tiled_m,
            vectorize_dim=MatmulIterator.M,
            mask_tmp_reg=mask_tmp_register,
            mask_reg=mask_register,
        )
        tiled_m = loop_matmul(
            rewriter,
            tiled_m,
            m_blocking,
            mloop_register,
        )

        result.append(set_matmul_iterator(rewriter, tiled_m, MatmulIterator.K))
        if remainder_m is not None:
            result.append(set_matmul_iterator(rewriter, remainder_m, MatmulIterator.K))
    return result


def _matmul_k_to_reg(
    rewriter: PatternRewriter,
    op: MatmulOp,
    *,
    disable_regalloc: bool,
) -> MatmulRegOp:
    if op.iterator.data != MatmulIterator.K:
        raise PassFailedException(
            "xsmm-apply-schedule requires K iteration before matmul_reg"
        )
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
    m_vectors = m_blocking // vector_length + bool(m_blocking % vector_length)
    c_val = ir.SSAValue.get(op.c, type=x86.registers.GeneralRegisterType)
    c_accesses = tuple(
        (
            (n * op.ldc.value.data + m * vector_length) * op.datatype.size,
            mask if m == m_vectors - 1 else None,
        )
        for n in range(op.n.value.data)
        for m in range(m_vectors)
    )
    accumulator_start = 32 - len(c_accesses)
    accumulators = tuple(
        load_vector(
            rewriter,
            insert_point,
            op.datatype,
            c_val,
            offset,
            destination=vector_register(
                accumulator_start + index,
                disable_regalloc=disable_regalloc,
            ),
            aligned=bool(op.aligned_c),
            mask=access_mask,
        )
        for index, (offset, access_mask) in enumerate(c_accesses)
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
        ),
        insertion_point=insert_point,
    )
    for accumulator, (offset, access_mask) in zip(
        matmul_reg.out_results, c_accesses, strict=True
    ):
        store_vector(
            rewriter,
            insert_point,
            op.datatype,
            c_val,
            offset,
            ir.SSAValue.get(accumulator, type=x86.registers.AVX512RegisterType),
            aligned=bool(op.aligned_c),
            mask=access_mask,
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
    tile_size: int,
    kloop_register: x86.registers.GeneralRegisterType,
) -> tuple[MatmulRegOp, ...]:
    tiled, remainder = tile_matmul_reg(
        rewriter,
        op,
        tile_size,
        kloop_register,
    )
    return (tiled,) if remainder is None else (tiled, remainder)


@dataclass
class ApplySchedulePattern(RewritePattern):
    """Apply the complete XSMM schedule to an N-iterating matmul."""

    isa_info: ISAInfo
    nano_kernel: NanoKernel
    disable_regalloc: bool
    disable_loop_construction: bool

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
        if self.disable_loop_construction:
            strategy = TilingStrategy(
                m_tile_size=descriptor.m,
                n_ranges=(BlockingRange(descriptor.n, descriptor.n),),
                k_tile_size=descriptor.k,
            )
        else:
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
            matmul_reg = _matmul_k_to_reg(
                rewriter,
                tiled_m,
                disable_regalloc=self.disable_regalloc,
            )
            for tiled_k in _tile_k(
                rewriter,
                matmul_reg,
                tile_size=strategy.k_tile_size,
                kloop_register=kloop_register,
            ):
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
    disable_loop_construction: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            strategy = get_xsmm_strategy(self.strategy)
        except ValueError as error:
            raise PassFailedException(str(error)) from error

        PatternRewriteWalker(
            ApplySchedulePattern(
                strategy.isa_info,
                strategy.nano_kernel,
                self.disable_regalloc,
                self.disable_loop_construction,
            ),
            apply_recursively=False,
        ).rewrite_module(op)
