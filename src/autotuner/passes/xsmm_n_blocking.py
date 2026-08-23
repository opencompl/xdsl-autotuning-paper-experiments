from xdsl.dialects import builtin
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
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def get_max_n_blocking_for_matmul_n(op: MatmulNOp, arch: Arch) -> int:
    """Reconstruct the current AVX-512 N blocking limit for matmul_n."""
    if isinstance(op.datatype, builtin.Float32Type):
        datatype = Datatype.F32
    elif isinstance(op.datatype, builtin.Float64Type):
        datatype = Datatype.F64
    else:
        raise PassFailedException(f"unsupported xsmm.matmul_n datatype {op.datatype}")

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
        MicroKernelConfig(), arch, desc, use_masking_a_c=False
    )
    max_n_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_max_n_blocking(
        config, desc, arch
    )
    init_m_blocking = libxsmm_generator_gemm_sse_avx_avx2_avx512_get_m_blocking(
        config, desc, arch, 0
    )
    init_m_blocks = (init_m_blocking + config.vector_length - 1) // config.vector_length
    while (
        init_m_blocks * max_n_blocking + init_m_blocks + 1
    ) > config.vector_reg_count:
        max_n_blocking -= 1
    return max_n_blocking
