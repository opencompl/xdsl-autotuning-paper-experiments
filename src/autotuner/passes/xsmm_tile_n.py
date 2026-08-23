from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86_scf
from xdsl.dialects.x86 import ops, registers
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulNOp
from autotuner.libxsmm_gemm.generator_common import MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
)
from autotuner.libxsmm_gemm.generator_gemm_sse_avx_avx2_avx512 import (
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking,
    libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


@dataclass
class TileMatmulNPattern(RewritePattern):
    """Tile N exactly while preserving the pointer results of matmul_n.

    The tile size is the largest divisor of the N range that does not exceed
    the current register-pressure limit. Every generated body leaves A fixed
    and advances B and C by its N tile, so composing the loop-carried results
    advances B and C by the complete original range without a remainder body.
    """

    arch: Arch
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulNOp, rewriter: PatternRewriter, /) -> None:
        if isinstance(op.datatype, builtin.Float32Type):
            datatype = Datatype.F32
        elif isinstance(op.datatype, builtin.Float64Type):
            datatype = Datatype.F64
        else:
            raise PassFailedException(
                f"unsupported xsmm.matmul_n datatype {op.datatype}"
            )

        flags = GEMMFlag.NONE
        if op.aligned_a.value.data:
            flags |= GEMMFlag.ALIGN_A
        if op.aligned_c.value.data:
            flags |= GEMMFlag.ALIGN_C
        desc = GEMMDescriptor(
            m=op.m.value.data,
            n=op.n_blocking.value.data,
            k=op.k.value.data,
            lda=op.lda.value.data,
            ldb=op.ldb.value.data,
            ldc=op.ldc.value.data,
            datatype=DescDatatype(datatype, datatype, datatype, datatype),
            flags=flags,
            prefetch=GEMMPrefetchType.NONE,
        )
        config = libxsmm_generator_gemm_init_micro_kernel_config(
            MicroKernelConfig(), self.arch, desc, use_masking_a_c=False
        )
        max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
            config, desc, self.arch
        )
        init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
            config, desc, self.arch, 0
        )
        init_m_blocks = (
            init_m_blocking + config.vector_length - 1
        ) // config.vector_length
        while (
            init_m_blocks * max_n_blocking + init_m_blocks + 1
        ) > config.vector_reg_count:
            max_n_blocking -= 1

        n = op.n_blocking.value.data
        n_tile = next(
            candidate
            for candidate in range(min(n, max_n_blocking), 0, -1)
            if n % candidate == 0
        )
        nloop_register = (
            registers.UNALLOCATED_REG64 if self.disable_regalloc else registers.R11
        )
        generated_code = GeneratedCode(rewriter, self.arch)
        n_start = op.n_start.value.data
        n_init = generated_code.insert(
            ops.DI_MovOp(n_start, destination=nloop_register)
        )
        inputs = tuple(op.operands)
        body = Block(
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
        nloop = generated_code.insert(
            x86_scf.ForOp(
                n_init.destination,
                builtin.IntegerAttr(n_start + n, ops.si32),
                builtin.IntegerAttr(n_tile, ops.si32),
                inputs,
                body,
            )
        )
        rewriter.replace(op, [], tuple(nloop.results[1:]))


@dataclass(frozen=True)
class XsmmTileNPass(ModulePass):
    """Tile matmul_n into an exact N loop using the current AVX-512 policy."""

    name = "xsmm-tile-n"

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
                "xsmm-tile-n currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            TileMatmulNPattern(arch, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
