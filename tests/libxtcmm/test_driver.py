import subprocess
import sys

import pytest


def _arguments(*, alpha: int = 1, beta: int = 1) -> list[str]:
    return [
        sys.executable,
        "-m",
        "autotuner.libxtcmm_gemm.libxtcmm_generator_gemm_driver",
        "dense",
        "unused.mlir",
        "matmul",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        str(alpha),
        str(beta),
        "1",
        "1",
        "skx",
        "nopf",
        "DP",
    ]


@pytest.mark.parametrize("arguments", [_arguments(alpha=-1), _arguments(beta=0)])
def test_rejects_unsupported_alpha_and_beta(arguments: list[str]):
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
