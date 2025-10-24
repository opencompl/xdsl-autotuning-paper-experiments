from unittest.mock import patch

from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Region

from autotuner.libxsmm_gemm import generator_gemm
from autotuner.libxsmm_gemm.generator_gemm import libxsmm_generator_gemm_kernel
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import (
    DescDatatype,
    GEMMDescriptor,
    GEMMFlag,
    GEMMPrefetchType,
)
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def test_libxsmm_generator_gemm_kernel():
    func_op = FuncOp("test", Region(), ((), ()))
    arch = Arch.LIBXSMM_X86_AVX512_SKX
    F64 = Datatype.F64
    dt = DescDatatype(F64, F64, F64, F64)
    desc = GEMMDescriptor(3, 16, 64, 16, 64, 16, dt, GEMMFlag(0), GEMMPrefetchType.NONE)

    # Mock the wrapper function to check what parameters it receives
    with patch.object(
        generator_gemm, "libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper"
    ) as mock_wrapper:
        libxsmm_generator_gemm_kernel(func_op, arch, desc)

        # Verify the wrapper was called with expected parameters
        mock_wrapper.assert_called_once_with(func_op, arch, desc)
