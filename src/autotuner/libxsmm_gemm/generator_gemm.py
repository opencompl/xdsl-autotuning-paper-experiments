from pathlib import Path

from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor


def libxsmm_generator_gemm_directasm(
    file_out: Path, routine_name: str, desc: GEMMDescriptor, arch: Arch
):
    raise NotImplementedError
