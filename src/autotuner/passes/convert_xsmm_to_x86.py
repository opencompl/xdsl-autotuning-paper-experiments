from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
)
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

from autotuner.compxsmm_gemm.generator_gemm_avx512_microkernel import (
    compxsmm_generator_gemm_avx512_kloop_kernel,
)
from autotuner.dialects.xsmm import MatmulKOp
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode, KLoopVals
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


@dataclass
class ConvertMatmulKPattern(RewritePattern):
    arch: Arch
    disable_regalloc: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulKOp, rewriter: PatternRewriter, /) -> None:
        if isinstance(op.datatype, builtin.Float32Type):
            datatype = Datatype.F32
        elif isinstance(op.datatype, builtin.Float64Type):
            datatype = Datatype.F64
        else:
            raise PassFailedException(
                f"unsupported xsmm.matmul_k datatype {op.datatype}"
            )

        m_blocking = op.m_blocking.value.data
        n_blocking = op.n_blocking.value.data
        k_blocking = op.k_blocking.value.data
        flags = GEMMFlag.ALIGN_A if op.aligned_a.value.data else GEMMFlag.NONE
        desc = GEMMDescriptor(
            m=m_blocking,
            n=n_blocking,
            k=k_blocking,
            lda=op.lda.value.data,
            ldb=op.ldb.value.data,
            ldc=m_blocking,
            datatype=DescDatatype(datatype, datatype, datatype, datatype),
            flags=flags,
            prefetch=GEMMPrefetchType.NONE,
        )
        micro_kernel_config = libxsmm_generator_gemm_init_micro_kernel_config(
            MicroKernelConfig(),
            self.arch,
            desc,
            use_masking_a_c=op.mask is not None,
        )

        vals = compxsmm_generator_gemm_avx512_kloop_kernel(
            GeneratedCode(rewriter, self.arch),
            GPRegMapping(),
            micro_kernel_config,
            desc,
            m_blocking,
            n_blocking,
            k_blocking,
            KLoopVals(
                SSAValue.get(op.a, type=GeneralRegisterType),
                SSAValue.get(op.b, type=GeneralRegisterType),
                SSAValue.get(op.c, type=GeneralRegisterType),
                SSAValue.get(op.rbp, type=GeneralRegisterType),
                SSAValue.get(op.rsp, type=GeneralRegisterType),
                (
                    None
                    if op.mask is None
                    else SSAValue.get(op.mask, type=AVX512MaskRegisterType)
                ),
                tuple(
                    SSAValue.get(acc, type=AVX512RegisterType)
                    for acc in op.accumulators
                ),
            ),
            disable_regalloc=self.disable_regalloc,
        )
        rewriter.replace(op, [], vals.vals)


@dataclass(frozen=True)
class ConvertXsmmToX86Pass(ModulePass):
    """Lower XSMM scheduling operations to x86 instructions."""

    name = "convert-xsmm-to-x86"

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
                "convert-xsmm-to-x86 currently supports AVX-512 architectures only"
            )

        PatternRewriteWalker(
            ConvertMatmulKPattern(arch, self.disable_regalloc),
            apply_recursively=False,
        ).rewrite_module(op)
