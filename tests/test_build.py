import os
import platform
import shutil
from pathlib import Path

import pytest

from autotuner import build
from autotuner.datasets import Sample, machine_file

SAMPLE = Sample(3, 5, 7, "compxsmm", "f64")


@pytest.fixture
def tool(tmp_path: Path) -> build.Toolchain:
    return build.toolchain("rapper", root=tmp_path)


def test_the_builder_writes_where_the_datasets_say_it_should() -> None:
    # evaluate and the Snakefile both read paths off `datasets`; if the builder
    # ever spelled them differently, measurements would look for missing files.
    tool = build.toolchain("rapper")
    asm = build.asm_artifact(tool, SAMPLE)
    binary = build.binary_artifact(
        tool,
        SAMPLE,
        asm,
        build.driver_artifact(tool, SAMPLE.kernel, 3, 5, 7, "f64", "time"),
        "time",
    )

    assert str(binary.path) == SAMPLE.path("rapper", "time.o")
    assert str(asm.path) == machine_file(
        "S",
        machine="rapper",
        kernel=SAMPLE.kernel,
        m=3,
        n=5,
        k=7,
        variant="compxsmm",
        dtype="f64",
    )


def test_one_shape_compiles_the_harness_once(tool: build.Toolchain) -> None:
    # The harness object depends on the shape and the dtype, never the variant,
    # which is what made the per-variant compile redundant under Snakemake.
    variants = ("libxsmm", "aocl", "compxsmm")
    plan = build.plan_shape(tool, [Sample(3, 5, 7, v, "f64") for v in variants], "time")

    drivers = [a for a in plan if a.path.name == "time.f64.o"]
    binaries = [a for a in plan if a.path.name.endswith(".time.o") and a not in drivers]

    assert len(drivers) == 1
    assert len(binaries) == len(variants)
    # Every binary links against that one object.
    assert all(drivers[0].path in b.needs for b in binaries)


def test_a_binary_is_stale_when_its_assembly_recipe_changes(
    tool: build.Toolchain,
) -> None:
    obj = build.driver_artifact(tool, SAMPLE.kernel, 3, 5, 7, "f64", "time")
    asm = build.asm_artifact(tool, SAMPLE)
    other = build.Artifact(asm.path, asm.steps, inputs=("a different generator",))

    before = build.binary_artifact(tool, SAMPLE, asm, obj, "time")
    after = build.binary_artifact(tool, SAMPLE, other, obj, "time")

    assert before.key != after.key


def test_a_recipe_key_follows_the_commands(tool: build.Toolchain) -> None:
    asm = build.asm_artifact(tool, SAMPLE)
    tweaked = build.Artifact(asm.path, (build.Step("run", ("clang", "-O2")),))

    assert asm.key != tweaked.key
    assert asm.key == build.Artifact(asm.path, asm.steps, asm.inputs).key


def test_only_a_matching_key_on_an_existing_file_is_up_to_date(
    tmp_path: Path,
) -> None:
    artifact = build.Artifact(tmp_path / "out.S", (build.Step("run", ("true",)),))

    assert build.stale(artifact, {}, force=False)

    artifact.path.write_text("built")
    assert build.stale(artifact, {}, force=False)
    assert not build.stale(artifact, {str(artifact.path): artifact.key}, force=False)
    assert build.stale(artifact, {str(artifact.path): "something else"}, force=False)
    assert build.stale(artifact, {str(artifact.path): artifact.key}, force=True)


def test_an_identical_rebuild_leaves_the_old_binary_alone(tmp_path: Path) -> None:
    # `evaluate` re-measures any binary newer than its recorded time, and
    # measuring is the slow half of a dataset run.
    binary = tmp_path / "libxsmm.f64.time.o"
    binary.write_bytes(b"ELF")
    os.utime(binary, (10**9, 10**9))
    artifact = build.Artifact(binary, (), preserve=True)
    artifact.target.write_bytes(b"ELF")

    assert build.settle(artifact) is False
    assert binary.stat().st_mtime == 10**9
    assert not artifact.target.exists()


def test_a_changed_rebuild_replaces_the_binary(tmp_path: Path) -> None:
    binary = tmp_path / "libxsmm.f64.time.o"
    binary.write_bytes(b"ELF")
    artifact = build.Artifact(binary, (), preserve=True)
    artifact.target.write_bytes(b"ELF, but different")

    assert build.settle(artifact) is True
    assert binary.read_bytes() == b"ELF, but different"


def test_samples_of_one_shape_share_a_task() -> None:
    grouped = build.by_shape(
        [
            Sample(1, 2, 3, "libxsmm", "f64"),
            Sample(1, 2, 3, "aocl", "f64"),
            Sample(1, 2, 3, "aocl", "f32"),
        ]
    )

    assert [len(v) for v in grouped.values()] == [2, 1]


def test_a_generated_file_is_cleared_before_it_is_appended_to(
    tmp_path: Path,
) -> None:
    # The XSMM generators append, so a file left by an interrupted run would
    # otherwise end up with two kernels in it.
    generated = tmp_path / "libxsmm.f64.c"
    generated.write_text("stale\n")

    build.run_step(build.Step("remove", (str(generated),)))
    build.run_step(build.Step("append", (str(generated), "fresh\n")))

    assert generated.read_text() == "fresh\n"


def test_a_step_environment_is_put_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWAP_A_B", "0")

    with build.environment({"SWAP_A_B": "1"}):
        assert os.environ["SWAP_A_B"] == "1"

    assert os.environ["SWAP_A_B"] == "0"


def test_a_failing_step_is_reported_against_its_artifact(tmp_path: Path) -> None:
    job = build.ShapeJob(
        "1x1x1",
        (build.Artifact(tmp_path / "boom.S", (build.Step("run", ("false",)),)),),
    )

    result = build.build_shape(job)

    assert result.built == ()
    assert len(result.failures) == 1
    assert "false" in result.failures[0][1]


def test_a_failure_skips_what_depended_on_it(tmp_path: Path) -> None:
    asm = build.Artifact(tmp_path / "boom.S", (build.Step("run", ("false",)),))
    binary = build.Artifact(
        tmp_path / "boom.time.o",
        (build.Step("run", ("true",)),),
        needs=(asm.path,),
    )

    result = build.build_shape(build.ShapeJob("1x1x1", (asm, binary)))

    # One failure, and the link that could not have worked was never attempted.
    assert [path for path, _ in result.failures] == [str(asm.path)]
    assert result.built == ()


def test_an_unknown_variant_has_no_recipe(tool: build.Toolchain) -> None:
    with pytest.raises(build.BuildFailed, match="no recipe"):
        build.asm_artifact(tool, Sample(1, 1, 1, "tvm", "f64"))


@pytest.mark.skipif(shutil.which("clang") is None, reason="needs a C compiler")
def test_a_kernel_builds_and_then_stays_built(tmp_path: Path) -> None:
    sample = Sample(4, 4, 4, "naive_c", "f32")
    system_and_arch = (platform.system(), platform.machine())
    machine = {
        ("Darwin", "arm64"): "neon",
        ("Linux", "x86_64"): "ci",
    }.get(system_and_arch)
    if machine is None:
        pytest.skip(f"no machine configuration for {system_and_arch}")

    build.build([sample], machine, root=tmp_path, jobs=2, quiet=True)

    binary = tmp_path / Path(sample.path(machine, "time.o")).relative_to("build")
    assert binary.exists()
    stamped = binary.stat().st_mtime_ns

    # A second run has nothing to do, and above all does not disturb the binary.
    build.build([sample], machine, root=tmp_path, jobs=2, quiet=True)
    assert binary.stat().st_mtime_ns == stamped


def test_only_the_allocated_compxsmm_asks_the_generator_to_stand_down(
    tool: build.Toolchain,
) -> None:
    # `compxsmm` has xDSL allocate the registers, which only works if the
    # generator leaves them unassigned; `compxsmm_manual` keeps its own.
    def generate_args(variant: str) -> tuple[str, ...]:
        sample = Sample(3, 5, 7, variant, "f64")
        (step,) = [
            s
            for s in build.asm_artifact(tool, sample).steps
            if s.kind == "generate:compxsmm"
        ]
        return step.args

    assert "--disable-regalloc" in generate_args("compxsmm")
    assert "--disable-regalloc" not in generate_args("compxsmm_manual")


def test_the_two_compxsmm_variants_run_different_pipelines(
    tool: build.Toolchain,
) -> None:
    allocated = tool.pipelines["compxsmm"]
    manual = tool.pipelines["compxsmm_manual"]

    assert "x86-allocate-registers" in allocated
    assert "x86-allocate-registers" not in manual
    assert manual
