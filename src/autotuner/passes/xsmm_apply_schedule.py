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
from autotuner.nano_kernel import GemmDescriptor, ISAInfo, NanoKernel
from autotuner.schedules import (
    load_c,
    set_matmul_iterator,
    split_n,
    store_c,
    tile_m,
    tile_matmul_reg,
    tile_n,
)
from autotuner.strategy import get_xsmm_strategy
from autotuner.tiling import TilingStrategy, compute_tiling_strategy

K_BLOCKING = 4
K_THRESHOLD = 23


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
        first, second = split_n(rewriter, op, first_range.extent)
        n_ranges = (
            (first, first_range.tile_size),
            (second, second_range.tile_size),
        )
    else:
        assert len(strategy.n_ranges) == 1
        n_ranges = ((op, strategy.n_ranges[0].tile_size),)

    result: list[MatmulOp] = []
    for n_range, n_tile in n_ranges:
        tiled_n = tile_n(rewriter, n_range, n_tile, nloop_register=nloop_register)
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
