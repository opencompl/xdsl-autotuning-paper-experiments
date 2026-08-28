from dataclasses import dataclass
from typing import Literal

from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
)
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.compxsmm_gemm.generator_gemm_avx512_microkernel import (
    compxsmm_generator_gemm_avx512_kloop_kernel,
)
from autotuner.dialects.xsmm import MatmulKOp
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE

from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode, KLoopVals
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor as LibxsmmGemmDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.nano_kernel import (
    FloatingPointType,
    GemmDescriptor,
    NanoKernel,
    RegisterCount,
    TargetInfo,
    TileSizes,
)


@dataclass(frozen=True)
class SkxTargetInfo(TargetInfo):
    """The register file and vector widths used by the SKX generator."""

    @property
    def arch(self) -> Literal["skx"]:
        return "skx"

    @property
    def register_capacity(self) -> RegisterCount:
        return RegisterCount(general=16, vector=32, mask=8)

    def vector_length(self, datatype: FloatingPointType) -> int:
        match datatype:
            case builtin.Float32Type():
                return 16
            case builtin.Float64Type():
                return 8
        raise ValueError(f"unsupported SKX datatype {datatype}")


class SkxNanoKernel(NanoKernel):
    """The current SKX fsdbcst/nofsdbcst nano-kernel family."""

    def supports(self, descriptor: GemmDescriptor, target: TargetInfo) -> bool:
        return isinstance(target, SkxTargetInfo) and isinstance(
            descriptor.datatype, builtin.Float32Type | builtin.Float64Type
        )

    def supports_tile(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> bool:
        if not self.supports(descriptor, target):
            return False
        if tile.m <= 0 or tile.n <= 0 or tile.k <= 0:
            return False
        vector_length = target.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        return m_vectors <= 4 and tile.n <= 28

    def register_usage(
        self,
        descriptor: GemmDescriptor,
        tile: TileSizes,
        target: TargetInfo,
    ) -> RegisterCount:
        if not self.supports_tile(descriptor, tile, target):
            raise ValueError("unsupported SKX nano-kernel tile")

        vector_length = target.vector_length(descriptor.datatype)
        m_vectors = (tile.m + vector_length - 1) // vector_length
        if m_vectors == 1:
            if tile.n >= 12:
                accumulator_sets = 1
            elif tile.n >= 6:
                accumulator_sets = 2
            else:
                accumulator_sets = 4
            accumulator_sets = min(accumulator_sets, tile.k)
            vector_registers = tile.n * accumulator_sets + min(tile.k, 2)
        else:
            vector_registers = m_vectors * tile.n + m_vectors + 1

        return RegisterCount(
            general=5,
            vector=vector_registers,
            mask=int(tile.m % vector_length != 0),
        )

    def rewrite(
        self,
        rewriter: PatternRewriter,
        op: MatmulKOp,
        target: TargetInfo,
        *,
        disable_regalloc: bool,
    ) -> None:
        if not isinstance(target, SkxTargetInfo):
            raise ValueError("SkxNanoKernel requires SkxTargetInfo")

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
        desc = LibxsmmGemmDescriptor(
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
        arch = ARCH_BY_CODE[target.arch]
        micro_kernel_config = libxsmm_generator_gemm_init_micro_kernel_config(
            MicroKernelConfig(),
            arch,
            desc,
            use_masking_a_c=op.mask is not None,
        )

        vals = compxsmm_generator_gemm_avx512_kloop_kernel(
            GeneratedCode(rewriter, arch),
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
            disable_regalloc=disable_regalloc,
        )
        rewriter.replace(op, [], vals.vals)
