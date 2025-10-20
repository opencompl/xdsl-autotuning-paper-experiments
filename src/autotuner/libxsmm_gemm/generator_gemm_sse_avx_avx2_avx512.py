from xdsl.dialects.x86_func import FuncOp
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor


def libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper(
    func_op: FuncOp, arch: Arch, desc: GEMMDescriptor
) -> None:
    raise NotImplementedError
