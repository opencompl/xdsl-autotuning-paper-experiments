from pathlib import Path

import pytest
import yaml

from autotuner.compxsmm_gemm import compxsmm_generator_gemm_driver
from autotuner.llvm_regalloc import kernel_mir, llvm_allocate


@pytest.mark.parametrize(
    ("shape", "precision"), [("10x10x64", "DP"), ("11x11x11", "SP")]
)
def test_kernel_converts_to_mir_llc_allocates(
    tmp_path: Path, shape: str, precision: str
) -> None:
    # 10x10x64 has a K loop with 22 loop-carried registers and a masked tail;
    # 11x11x11 zeroes its accumulators and broadcasts B from memory.
    m, n, k = shape.split("x")
    kernel = tmp_path / "kernel.mlir"
    compxsmm_generator_gemm_driver.main(
        ["dense", str(kernel), "matmul", m, n, k, m, k, m, "1", "1", "1", "1"]
        + ["skx", "nopf", precision, "--disable-regalloc"]
    )
    config = yaml.safe_load(Path("default.yaml").read_text())
    passes = ",".join(config["compxsmm-llvm-gemm-passes"]["avx512"])
    mir = kernel_mir(kernel, passes)

    # llc runs its machine verifier, so malformed MIR raises here.
    llvm_allocate(mir, "znver4")


def live_vectors_mir(count: int) -> str:
    regs = "\n".join(f"  - {{ id: {i + 1}, class: vr512 }}" for i in range(count))
    loads = "\n".join(
        f"    %{i + 1}:vr512 = VMOVUPDZrm %0, 1, $noreg, {64 * i}, $noreg"
        for i in range(count)
    )
    stores = "\n".join(
        f"    VMOVUPDZmr %0, 1, $noreg, {64 * i}, $noreg, %{i + 1}"
        for i in range(count)
    )
    return f"""---
name: f
tracksRegLiveness: true
registers:
{regs}
liveins:
  - {{ reg: '$rdi' }}
body: |
  bb.0:
    liveins: $rdi
    %0:gr64 = COPY $rdi
{loads}
{stores}
    RET64
...
"""


def test_spills_are_reported_once_vectors_exceed_registers() -> None:
    assert not llvm_allocate(live_vectors_mir(32), "znver4").spilled
    over = llvm_allocate(live_vectors_mir(33), "znver4")
    assert over.spilled and over.spills and over.reloads
