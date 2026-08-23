from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_scf
from xdsl.dialects.x86 import ops, registers
from xdsl.ir import Block, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulMOp
from autotuner.libxsmm_gemm.generator_common import LIBXSMM_X86_AVX512_MASK_REG
from autotuner.libxsmm_gemm.generator_common_x86 import (
    libxsmm_generator_initialize_avx512_mask,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


@dataclass
class TileMatmulMPattern(RewritePattern):
    """Tile M without changing the pointer results of matmul_m.

    Every generated body advances A and C by its M block and leaves B fixed.
    Composing the loop-carried results therefore advances A and C by the full
    original M extent, regardless of the chosen blocking or remainder mask.
    """

    arch: Arch
    disable_regalloc: bool

    def _insert_loop(
        self,
        generated_code: GeneratedCode,
        op: MatmulMOp,
        inputs: tuple[SSAValue, ...],
        *,
        lower_bound: int,
        upper_bound: int,
        m_blocking: int,
        mask: SSAValue | None,
    ) -> tuple[SSAValue, ...]:
        mloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R10
        )
        m_init = generated_code.insert(
            ops.DI_MovOp(lower_bound, destination=mloop_register)
        )
        iter_args = (*inputs, *((mask,) if mask is not None else ()))
        body = Block(
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
        mloop = generated_code.insert(
            x86_scf.ForOp(
                m_init.destination,
                builtin.IntegerAttr(upper_bound, ops.si32),
                builtin.IntegerAttr(m_blocking, ops.si32),
                iter_args,
                body,
            )
        )
        return tuple(mloop.results[1:6])

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulMOp, rewriter: PatternRewriter, /) -> None:
        if op.mask is not None:
            return

        if isinstance(op.datatype, builtin.Float32Type):
            datatype = Datatype.F32
            max_blocking = 64
            vector_length = 16
        elif isinstance(op.datatype, builtin.Float64Type):
            datatype = Datatype.F64
            max_blocking = 32
            vector_length = 8
        else:
            raise PassFailedException(
                f"unsupported xsmm.matmul_m datatype {op.datatype}"
            )

        m = op.m_blocking.value.data
        m_blocking = min(m, max_blocking)
        blocked_end = m // m_blocking * m_blocking
        generated_code = GeneratedCode(rewriter, self.arch)
        inputs = tuple(op.operands[:5])

        mask = None
        if m_blocking % vector_length:
            mask_tmp = (
                registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R15
            )
            mask = libxsmm_generator_initialize_avx512_mask(
                generated_code,
                mask_tmp,
                LIBXSMM_X86_AVX512_MASK_REG,
                vector_length - m_blocking % vector_length,
                datatype,
            )
        inputs = self._insert_loop(
            generated_code,
            op,
            inputs,
            lower_bound=0,
            upper_bound=blocked_end,
            m_blocking=m_blocking,
            mask=mask,
        )

        if remainder := m - blocked_end:
            mask = None
            if remainder % vector_length:
                mask_tmp = (
                    registers.UNALLOCATED_REG64
                    if self.disable_regalloc
                    else registers.R15
                )
                mask = libxsmm_generator_initialize_avx512_mask(
                    generated_code,
                    mask_tmp,
                    LIBXSMM_X86_AVX512_MASK_REG,
                    vector_length - remainder % vector_length,
                    datatype,
                )
            inputs = self._insert_loop(
                generated_code,
                op,
                inputs,
                lower_bound=blocked_end,
                upper_bound=m,
                m_blocking=remainder,
                mask=mask,
            )

        rewriter.replace(op, [], inputs)


@dataclass(frozen=True)
class XsmmTileMPass(ModulePass):
    """Apply the current AVX-512 F32/F64 M blocking policy to matmul_m.

    The pass materializes M loops, including single-iteration loops, and masks
    partial final vectors while preserving the original pointer results.
    """

    name = "xsmm-tile-m"

    arch: str = "skx"
    disable_regalloc: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        try:
            arch = ARCH_BY_CODE[self.arch]
        except KeyError as error:
            raise PassFailedException(
                f"unknown architecture code '{self.arch}'"
            ) from error
        if not (Arch.LIBXSMM_X86_AVX512_SKX <= arch <= Arch.LIBXSMM_X86_ALLFEAT):
            raise PassFailedException(
                "xsmm-tile-m currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            TileMatmulMPattern(arch, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
