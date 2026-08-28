from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.dialects.x86.registers import (
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
)
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.exceptions import PassFailedException

from autotuner.dialects.xsmm import MatmulKOp
from autotuner.libxsmm_gemm.generator_common import GPRegMapping, MicroKernelConfig
from autotuner.libxsmm_gemm.generator_gemm_common import (
    libxsmm_generator_gemm_init_micro_kernel_config,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode, KLoopVals
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor as LibxsmmGemmDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.nano_kernel import GemmDescriptor, TargetInfo, TileSizes


def tile_sizes_from_op(op: MatmulKOp) -> TileSizes:
    return TileSizes(
        op.m_blocking.value.data,
        op.n_blocking.value.data,
        op.k_blocking.value.data,
    )


def descriptor_from_op(op: MatmulKOp) -> GemmDescriptor:
    return GemmDescriptor(
        m=op.m_blocking.value.data,
        n=op.n_blocking.value.data,
        k=op.k_blocking.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=op.m_blocking.value.data,
        datatype=op.datatype,
        aligned_a=bool(op.aligned_a.value.data),
        aligned_c=False,
    )


@dataclass(frozen=True)
class SkxMatmulKContext:
    generated_code: GeneratedCode
    gp_reg_mapping: GPRegMapping
    micro_kernel_config: MicroKernelConfig
    descriptor: LibxsmmGemmDescriptor
    values: KLoopVals


def create_matmul_k_context(
    rewriter: PatternRewriter, op: MatmulKOp, target: TargetInfo
) -> SkxMatmulKContext:
    if target.arch != "skx":
        raise ValueError("SKX nano-kernels require an SKX target")

    if isinstance(op.datatype, builtin.Float32Type):
        datatype = Datatype.F32
    elif isinstance(op.datatype, builtin.Float64Type):
        datatype = Datatype.F64
    else:
        raise PassFailedException(f"unsupported xsmm.matmul_k datatype {op.datatype}")

    m_blocking = op.m_blocking.value.data
    descriptor = LibxsmmGemmDescriptor(
        m=m_blocking,
        n=op.n_blocking.value.data,
        k=op.k_blocking.value.data,
        lda=op.lda.value.data,
        ldb=op.ldb.value.data,
        ldc=m_blocking,
        datatype=DescDatatype(datatype, datatype, datatype, datatype),
        flags=GEMMFlag.ALIGN_A if op.aligned_a.value.data else GEMMFlag.NONE,
        prefetch=GEMMPrefetchType.NONE,
    )
    micro_kernel_config = libxsmm_generator_gemm_init_micro_kernel_config(
        MicroKernelConfig(),
        Arch.LIBXSMM_X86_AVX512_SKX,
        descriptor,
        use_masking_a_c=op.mask is not None,
    )
    values = KLoopVals(
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
        tuple(SSAValue.get(acc, type=AVX512RegisterType) for acc in op.accumulators),
    )
    return SkxMatmulKContext(
        GeneratedCode(rewriter, Arch.LIBXSMM_X86_AVX512_SKX),
        GPRegMapping(),
        micro_kernel_config,
        descriptor,
        values,
    )
