import os
import platform
import shutil
import subprocess
from dataclasses import replace
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


def test_a_generated_file_is_cleared_before_it_is_written(
    tmp_path: Path,
) -> None:
    # The XSMM generators append, so a file left by an interrupted run would
    # otherwise end up with two kernels in it.
    generated = tmp_path / "libxsmm.f64.c"
    generated.write_text("stale\n")

    build.run_step(build.Step("remove", (str(generated),)))

    assert not generated.exists()


def test_a_step_environment_is_put_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XTC_MLIR_PREFIX", "before")

    with build.environment({"XTC_MLIR_PREFIX": "during"}):
        assert os.environ["XTC_MLIR_PREFIX"] == "during"

    assert os.environ["XTC_MLIR_PREFIX"] == "before"


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


@pytest.mark.skipif(
    (platform.system(), platform.machine()) != ("Darwin", "arm64"),
    reason="native M4 toolchain needs the Apple SDK",
)
@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_m4_native_correctness_driver(tmp_path: Path, dtype: str) -> None:
    sample = Sample(5, 6, 7, "naive_c", dtype)
    build.build([sample], "m4", root=tmp_path, driver="test", jobs=1, quiet=True)
    binary = tmp_path / Path(sample.path("m4", "test.o")).relative_to("build")
    done = subprocess.run([binary], capture_output=True, text=True, check=True)
    assert "Test Passed" in done.stdout


def test_native_exp_builds_validates_and_pins_its_approximation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or "avx512f" not in Path("/proc/cpuinfo").read_text()
        or not shutil.which("clang")
        or not build.pkg_config("libxsmm")[0]
    ):
        pytest.skip("requires AVX-512 and the LIBXSMM development package")
    # Correctness does not need tower's PAPI timer or its Zen 5 compiler target.
    monkeypatch.setitem(
        build.MACHINES, "ci", replace(build.MACHINES["ci"], libxsmm_arch="skx")
    )
    samples = [
        Sample(128, 1, 1, "libxsmm", "f32", kernel="exp", exp_degree=degree)
        for degree in (3, 5)
    ]
    # Build together to catch shared paths or a driver compiled for the wrong
    # degree, as well as an ambient hint overriding the requested polynomial.
    build.build(samples, "ci", root=tmp_path, driver="test", jobs=1, quiet=True)
    for sample in samples:
        binary = tmp_path / Path(sample.path("ci", "test.o")).relative_to("build")
        outputs = []
        for hint in ("0", "1"):
            done = subprocess.run(
                [binary],
                env=os.environ
                | {"LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX": hint},
                capture_output=True,
                text=True,
                check=True,
            )
            assert f"Test Passed: degree-{sample.exp_degree} exp" in done.stdout
            outputs.append(done.stdout)
        assert outputs[0] == outputs[1]
        relative_error = float(outputs[0].splitlines()[0].split(": ")[1])
        if sample.exp_degree == 3:
            # The loose cubic tolerance alone would also accept degree five.
            assert 1e-5 < relative_error < 1e-3
        else:
            assert relative_error < 1e-6


@pytest.mark.parametrize("host", [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_m4_requires_native_apple_silicon(
    monkeypatch: pytest.MonkeyPatch, host: tuple[str, str]
) -> None:
    monkeypatch.setattr(platform, "system", lambda: host[0])
    monkeypatch.setattr(platform, "machine", lambda: host[1])
    with pytest.raises(build.BuildFailed, match="native Apple Silicon"):
        build.toolchain("m4")


def test_m4_does_not_use_the_legacy_cycle_estimate(tool: build.Toolchain) -> None:
    with pytest.raises(build.BuildFailed, match="timing is not enabled"):
        build.plan_shape(replace(tool, machine="m4"), [SAMPLE], "time")


@pytest.mark.skipif(shutil.which("clang") is None, reason="needs a C compiler")
def test_a_libxsmm_upgrade_through_a_symlink_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine = {
        ("Darwin", "arm64"): "neon",
        ("Linux", "x86_64"): "ci",
    }.get((platform.system(), platform.machine()))
    if machine is None:
        pytest.skip("no machine configuration for this host")

    # Docker keeps /opt/toolchain/bin fixed across image upgrades, but its
    # symlinks point into a different, immutable Nix package each time.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    generator = bin_dir / "libxsmm_gemm_generator"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    sample = Sample(1, 1, 1, "libxsmm", "f32")
    root = tmp_path / "build"
    binary = tmp_path / sample.path(machine, "test.o")
    generated = binary.with_name("libxsmm.f32.c")
    executables: list[bytes] = []

    for increment in (1, 2):
        implementation = tmp_path / f"generator-{increment}"
        source = f"void matmul(float *A, float *B, float *C) {{ C[0] += {increment}; }}"
        implementation.write_text(f"#!/bin/sh\nprintf '%s\\n' '{source}' > \"$2\"\n")
        implementation.chmod(0o755)
        generator.unlink(missing_ok=True)
        generator.symlink_to(implementation)

        build.build([sample], machine, root=root, driver="test", jobs=1, quiet=True)
        assert generated.read_text().strip() == source
        executables.append(binary.read_bytes())
        stamped = binary.stat().st_mtime_ns
        build.build([sample], machine, root=root, driver="test", jobs=1, quiet=True)
        assert binary.stat().st_mtime_ns == stamped

    assert executables[0] != executables[1]


def test_only_the_manual_compxsmm_keeps_the_generators_registers(
    tool: build.Toolchain,
) -> None:
    # The shared CompXSMM IR leaves registers for xDSL to allocate, while the
    # manual variant keeps the generator's assignment.
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


def test_compxsmm_schedules_use_the_same_generated_mlir(tool: build.Toolchain) -> None:
    variants = (
        "compxsmm",
        "compxsmm_plusnarrow",
        "libxsmm-skx-fsdbcst",
        "libxsmm-skx-nofsdbcst",
    )
    mlirs = [
        next(
            step.args[1]
            for step in build.asm_artifact(tool, Sample(3, 5, 7, variant, "f64")).steps
            if step.kind == "generate:compxsmm"
        )
        for variant in variants
    ]

    assert len(set(mlirs)) == 1


def test_pinned_compxsmm_schedules_run_distinct_pipelines(
    tool: build.Toolchain,
) -> None:
    for variant in ("libxsmm-skx-fsdbcst", "libxsmm-skx-nofsdbcst"):
        assert f"strategy={variant}" in tool.pipelines[variant]
        assert "{nanokernel}" not in tool.pipelines[variant]
        assert "disable-loop-construction=true" in tool.pipelines[variant]
        assert "x86-allocate-registers" in tool.pipelines[variant]


def test_the_two_compxsmm_variants_run_different_pipelines(
    tool: build.Toolchain,
) -> None:
    allocated = tool.pipelines["compxsmm"]
    manual = tool.pipelines["compxsmm_manual"]

    assert "x86-allocate-registers" in allocated
    assert "x86-allocate-registers" not in manual
    assert manual
