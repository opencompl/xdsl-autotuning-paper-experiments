import pytest
from xdsl.dialects.builtin import ModuleOp

from autotuner import analyzers
from autotuner.analyzers import LLVM_MCA


def test_llvm_mca_uses_arch_and_cpu_options(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_run_command(
        command: str,
        timeout: int | None = None,
        input: str | None = None,
    ) -> tuple[str, str]:
        commands.append(command)
        return "Block RThroughput: 1.5\n", ""

    monkeypatch.setattr(analyzers, "run_command", fake_run_command)

    cost = LLVM_MCA(arch="x86-64", cpu="skylake").evaluate(ModuleOp([]))

    assert commands == ["llvm-mca --march=x86-64 -mcpu=skylake"]
    assert cost.execution_time == 1.5
