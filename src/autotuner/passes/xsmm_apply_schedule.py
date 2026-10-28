"""Apply a concrete XSMM-compatible schedule and lower it to x86."""

from dataclasses import dataclass
from functools import partial

from xdsl import ir
from xdsl.context import Context
from xdsl.dialects import builtin, x86, x86_func
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException

from autotuner.amx import MAX_TILE_ROWS, load_tile_config
from autotuner.dialects.xsmm import MatmulIterator, MatmulOp, MatmulRegOp
from autotuner.instructions import load_mask
from autotuner.nano_kernel import GemmDescriptor, ISAInfo, NanoKernel
from autotuner.schedules import (
    add_matmul_ins,
    loop_matmul,
    set_matmul_iterator,
    split_matmul,
    tile_matmul_reg,
)
from autotuner.skx_nano_kernel_utils import vector_register
from autotuner.spr_amx_bf16_nano_kernel import (
    SCRATCH_BYTES,
    SCRATCH_LD,
    SprAmxBf16NanoKernel,
    amx_code_blocks,
)
from autotuner.strategy import get_xsmm_strategy
from autotuner.tiling import BlockingRange, TilingStrategy, compute_tiling_strategy


def _tile_n_m(
    rewriter: PatternRewriter,
    op: MatmulOp,
    strategy: TilingStrategy,
    nano_kernel: NanoKernel,
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
            remainder_m = nano_kernel.attach_mask(
                rewriter,
                remainder_m,
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

        tiled_m = nano_kernel.attach_mask(
            rewriter,
            tiled_m,
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
    nano_kernel: NanoKernel,
    isa_info: ISAInfo,
    *,
    disable_regalloc: bool,
) -> MatmulRegOp:
    if op.iterator.data != MatmulIterator.K:
        raise PassFailedException(
            "xsmm-apply-schedule requires K iteration before matmul_reg"
        )
    if op.outs or not all(
        isinstance(value.type, x86.registers.AVX512MaskRegisterType) for value in op.ins
    ):
        raise PassFailedException("xsmm-apply-schedule supports only mask ins")
    m_blocking = op.m.value.data
    # The nano-kernel picks the register type one M vector lands in, so the C
    # accumulators are loaded and stored in that same type.
    vector_type = nano_kernel.vector_type(
        m_blocking, op.datatype, isa_info, op.ab_datatype
    )
    vector_length = vector_type.bitwidth() // op.datatype.bitwidth
    # The mask of C's tail lanes comes first; the nano-kernel may add its own.
    if m_blocking % vector_length:
        if not op.ins:
            raise PassFailedException(
                "xsmm-apply-schedule requires a mask for a partial M vector"
            )
        c_mask = ir.SSAValue.get(op.ins[0], type=x86.registers.AVX512MaskRegisterType)
    else:
        c_mask = None

    insert_point = InsertPoint.before(op)
    m_vectors = m_blocking // vector_length + bool(m_blocking % vector_length)
    c_val = ir.SSAValue.get(op.c, type=x86.registers.GeneralRegisterType)
    c_accesses = tuple(
        (
            (n * op.ldc.value.data + m * vector_length) * op.datatype.size,
            c_mask if m == m_vectors - 1 else None,
        )
        for n in range(op.n.value.data)
        for m in range(m_vectors)
    )
    accumulator_start = 32 - len(c_accesses)
    accumulators = nano_kernel.load_accumulators(
        rewriter,
        insert_point,
        op,
        c_val,
        c_accesses,
        tuple(
            vector_register(
                accumulator_start + index,
                vector_type,
                disable_regalloc=disable_regalloc,
            )
            for index in range(len(c_accesses))
        ),
        disable_regalloc=disable_regalloc,
    )
    matmul_reg = rewriter.insert(
        MatmulRegOp(
            op.a,
            op.b,
            op.ins,
            accumulators,
            m=m_blocking,
            n=op.n.value.data,
            k=op.k.value.data,
            lda=op.lda.value.data,
            ldb=op.ldb.value.data,
            datatype=op.datatype,
            ab_datatype=op.ab_datatype,
            vnni_a=bool(op.vnni_a),
            aligned_a=bool(op.aligned_a),
        ),
        insertion_point=insert_point,
    )
    nano_kernel.store_accumulators(
        rewriter,
        insert_point,
        op,
        c_val,
        c_accesses,
        tuple(
            ir.SSAValue.get(accumulator, type=x86.registers.X86VectorRegisterType)
            for accumulator in matmul_reg.out_results
        ),
        disable_regalloc=disable_regalloc,
    )
    rewriter.replace(
        op,
        [],
        (
            matmul_reg.a_out,
            matmul_reg.b_out,
            op.c,
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


def _descriptor(op: MatmulOp) -> GemmDescriptor:
    return GemmDescriptor(
        m=op.m.value.data,
        n=op.n.value.data,
        k=op.k.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.ldc.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a),
        aligned_c=bool(op.aligned_c),
        ab_datatype=op.ab_datatype,
        vnni_a=bool(op.vnni_a),
        c_datatype=op.c_datatype,
    )


def _function_name(op: ir.Operation) -> str:
    parent = op.parent_op()
    while parent is not None and not isinstance(parent, x86_func.FuncOp):
        parent = parent.parent_op()
    assert parent is not None, "xsmm.matmul outside a function"
    return parent.sym_name.data


def _apply_amx_schedule(
    rewriter: PatternRewriter,
    op: MatmulOp,
    nano_kernel: SprAmxBf16NanoKernel,
    isa_info: ISAInfo,
    *,
    disable_regalloc: bool,
    first_config: int,
) -> int:
    """Lower `op` the way LIBXSMM's AMX kernel does; return how many tile
    configurations it loads.

    The caller's tile configuration is saved on the stack and restored at the
    end. M is cut into LIBXSMM's code blocks, each of which loads the
    configuration of its tiles, then loops over N and M blocks whose
    accumulator tiles the nano-kernel loads, multiplies into over all of K and
    stores. Tile loads take their row strides from registers set once, which
    ride along as the matmul's `ins`.

    A BF16 C moves through a scratch below the caller's configuration, aligned
    to 64 bytes like LIBXSMM's GEMM scratch: the third stride is the scratch's
    instead of C's, and the scratch and each block's masks join the `ins`. The
    address of the caller's configuration is kept in a register meanwhile.
    """
    m, n, k = op.m.value.data, op.n.value.data, op.k.value.data
    blocks = amx_code_blocks(m, n, k)
    bf16_c = op.c_datatype is not None
    if disable_regalloc:
        nloop_register = mloop_register = x86.registers.UNALLOCATED_REG64
        stride_registers = (x86.registers.UNALLOCATED_REG64,) * 3
        frame_register = temp_register = x86.registers.UNALLOCATED_REG64
    else:
        nloop_register = x86.registers.R11
        mloop_register = x86.registers.R10
        stride_registers = (x86.registers.R13, x86.registers.R14, x86.registers.R15)
        frame_register, temp_register = x86.registers.R12, x86.registers.R10

    name = _function_name(op)
    following = op.next_op
    assert following is not None
    rsp = rewriter.insert(
        x86.ops.GetRegisterOp(x86.registers.RSP), InsertPoint.before(op)
    ).result
    caller_config = rewriter.insert(
        x86.ops.RI_SubOp(rsp, 64, register_out=x86.registers.RSP),
        InsertPoint.before(op),
    ).register_out
    rewriter.insert(x86.ops.M_SttilecfgOp(caller_config, 0), InsertPoint.before(op))
    if bf16_c:
        frame = rewriter.insert(
            x86.ops.DS_MovOp(caller_config, destination=frame_register),
            InsertPoint.before(op),
        ).destination
        below = rewriter.insert(
            x86.ops.RI_SubOp(
                caller_config, SCRATCH_BYTES, register_out=x86.registers.RSP
            ),
            InsertPoint.before(op),
        ).register_out
        alignment = rewriter.insert(
            x86.ops.DI_MovOp(-64, destination=temp_register), InsertPoint.before(op)
        ).destination
        scratch: tuple[ir.SSAValue, ...] = (
            rewriter.insert(
                x86.ops.RS_AndOp(below, alignment, register_out=x86.registers.RSP),
                InsertPoint.before(op),
            ).register_out,
        )
        c_stride = SCRATCH_LD
    else:
        frame = caller_config
        scratch = ()
        c_stride = op.ldc.value.data
    strides = tuple(
        rewriter.insert(
            x86.ops.DI_MovOp(leading_dimension, destination=register),
            InsertPoint.before(op),
        ).destination
        for leading_dimension, register in zip(
            (op.lda.value.data, op.ldb.value.data, c_stride),
            stride_registers,
            strict=True,
        )
    )
    op = add_matmul_ins(rewriter, op, (*strides, *scratch))

    if len(blocks) == 1:
        parts = (op,)
    else:
        parts = split_matmul(
            rewriter, set_matmul_iterator(rewriter, op, MatmulIterator.M), blocks[0].m
        )

    mx1 = n <= MAX_TILE_ROWS
    for index, (block, part) in enumerate(zip(blocks, parts, strict=True)):
        load_tile_config(
            partial(rewriter.insert, insertion_point=InsertPoint.before(part)),
            block.layout().tile_config(),
            f".L{name}_tileconfig{first_config + index}",
        )
        if bf16_c:
            # LIBXSMM's masks of a partial tile, F32 in k2 and BF16 in k1.
            masks = tuple(
                load_mask(
                    rewriter,
                    InsertPoint.before(part),
                    temp_register,
                    mask_register,
                    lanes,
                    lane_type,
                )
                for (lanes, lane_type), mask_register in zip(
                    nano_kernel.bf16_c_masks(block.layout(), mx1=mx1),
                    (x86.registers.K2, x86.registers.K1),
                )
            )
            part = add_matmul_ins(rewriter, part, masks)
        matmul_n = set_matmul_iterator(rewriter, part, MatmulIterator.N)
        tiled_n = loop_matmul(rewriter, matmul_n, block.n_blocking, nloop_register)
        matmul_m = set_matmul_iterator(rewriter, tiled_n, MatmulIterator.M)
        tiled_m = loop_matmul(rewriter, matmul_m, block.m_blocking, mloop_register)
        matmul_k = set_matmul_iterator(rewriter, tiled_m, MatmulIterator.K)
        matmul_reg = nano_kernel.accumulate_in_tiles(
            rewriter, matmul_k, mx1=mx1, disable_regalloc=disable_regalloc
        )
        nano_kernel.rewrite(
            rewriter, matmul_reg, isa_info, disable_regalloc=disable_regalloc
        )

    if bf16_c:
        caller_config = rewriter.insert(
            x86.ops.DS_MovOp(frame, destination=x86.registers.RSP),
            InsertPoint.before(following),
        ).destination
    rewriter.insert(
        x86.ops.M_LdtilecfgOp(caller_config, 0), InsertPoint.before(following)
    )
    rewriter.insert(
        x86.ops.RI_AddOp(caller_config, 64, register_out=x86.registers.RSP),
        InsertPoint.before(following),
    )
    return len(blocks)


@dataclass
class ApplySchedulePattern(RewritePattern):
    """Apply the complete XSMM schedule to an N-iterating matmul."""

    isa_info: ISAInfo
    nano_kernel: NanoKernel
    disable_regalloc: bool
    disable_loop_construction: bool
    tile_configs: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter, /) -> None:
        if op.iterator.data != MatmulIterator.N:
            return
        if isinstance(self.nano_kernel, SprAmxBf16NanoKernel):
            if self.disable_loop_construction:
                raise PassFailedException(
                    "the AMX schedule always constructs LIBXSMM's loops"
                )
            if not self.nano_kernel.supports(_descriptor(op), self.isa_info):
                raise PassFailedException(
                    "nano-kernel does not support the GEMM descriptor"
                )
            self.tile_configs += _apply_amx_schedule(
                rewriter,
                op,
                self.nano_kernel,
                self.isa_info,
                disable_regalloc=self.disable_regalloc,
                first_config=self.tile_configs,
            )
            return
        descriptor = _descriptor(op)
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
            self.nano_kernel,
            nloop_register=nloop_register,
            mloop_register=mloop_register,
            mask_tmp_register=mask_tmp_reg,
            mask_register=x86.registers.K1,
        ):
            matmul_reg = _matmul_k_to_reg(
                rewriter,
                tiled_m,
                self.nano_kernel,
                self.isa_info,
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
