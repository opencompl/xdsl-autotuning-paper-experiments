import subprocess
import sys
from pathlib import Path

import pytest

from autotuner import validate
from autotuner.datasets import Sample

SAMPLE = Sample(3, 5, 7, "libxsmm", "f64")


def completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def executable(tmp_path: Path, script: str) -> Path:
    """A stand-in for a test binary, so `check` has something to run."""
    path = tmp_path / "libxsmm.f64.test.o"
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point `check` at a script in tmp_path instead of the build tree."""

    def place(script: str) -> Path:
        path = executable(tmp_path, script)
        monkeypatch.setattr(validate, "binary", lambda *_: path)
        return path

    return place


def test_the_binary_is_the_one_the_datasets_name() -> None:
    # The tests are linked by `autotuner.build`; spelling the path differently
    # here would run nothing and validate nothing.
    assert str(validate.binary(SAMPLE, "rapper")) == SAMPLE.path("rapper", "test.o")


def test_a_failing_kernel_is_reported_by_the_harness_verdict() -> None:
    done = completed(1, "A\n1 2\n\nTest Failed: The results do not match.\n")

    assert validate.why(done) == "Test Failed: The results do not match."


def test_a_kernel_that_never_reached_its_verdict_reports_how_it_died() -> None:
    # An illegal instruction, say: no verdict was printed, so the status is all
    # there is to go on.
    done = completed(-4, "A\n1 2\n", "Illegal instruction\n")

    assert validate.why(done) == "exited with -4: Illegal instruction"


def test_a_silent_crash_still_says_something() -> None:
    assert validate.why(completed(-11)) == "exited with -11"


def test_a_passing_kernel_keeps_its_output_beside_the_binary(binary) -> None:
    path = binary('echo "Test Passed: The results are equal!"')

    result = validate.check(SAMPLE, "rapper")

    assert result.passed
    assert result.reason == ""
    log = path.with_suffix(".log")
    assert log.name == "libxsmm.f64.test.log"
    assert log.read_text() == "Test Passed: The results are equal!\n"


def test_a_failing_kernel_keeps_its_output_too(binary) -> None:
    # The log is what a failure is diagnosed from, so it is written either way.
    path = binary('echo "Test Failed: The results do not match."; exit 1')

    result = validate.check(SAMPLE, "rapper")

    assert not result.passed
    assert result.reason == "Test Failed: The results do not match."
    assert "Test Failed" in path.with_suffix(".log").read_text()


def test_the_measured_environment_is_the_validated_one(binary) -> None:
    path = binary('echo "$OMP_NUM_THREADS $BLIS_NUM_THREADS $LIBPFM_FORCE_PMU"')

    validate.check(SAMPLE, "tower")

    assert path.with_suffix(".log").read_text().strip() == "1 1 amd64"


def test_a_missing_binary_is_a_failure_rather_than_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate, "binary", lambda *_: tmp_path / "never.built.test.o")

    result = validate.check(SAMPLE, "rapper")

    assert not result.passed
    assert "never.built.test.o" in result.reason


def test_the_tests_are_built_against_the_test_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not `time`: linking the timing binaries here would invalidate every
    # cached measurement.
    monkeypatch.setattr(
        validate,
        "selected_samples",
        lambda *_: [SAMPLE, Sample(1, 1, 1, "aocl", "f64")],
    )
    monkeypatch.setattr(validate, "run", lambda *_a, **_k: [])
    asked: list[tuple] = []
    monkeypatch.setattr(
        validate.builder,
        "build",
        lambda samples, machine, **kwargs: asked.append((samples, machine, kwargs)),
    )

    validate.validate("rapper")

    assert len(asked) == 1
    samples, machine, kwargs = asked[0]
    assert machine == "rapper"
    assert samples == [SAMPLE, Sample(1, 1, 1, "aocl", "f64")]
    assert kwargs["driver"] == "test"


def test_an_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        validate.validate("rapper", ["not.a.dataset"], build=False)


def test_every_sample_is_run_once(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [SAMPLE, Sample(1, 1, 1, "aocl", "f64")]
    monkeypatch.setattr(validate, "selected_samples", lambda *_: samples)
    ran: list[Sample] = []

    def record(sample: Sample, machine_name: str) -> validate.Result:
        ran.append(sample)
        return validate.Result(sample, True)

    monkeypatch.setattr(validate, "check", record)

    assert validate.validate("rapper", build=False) == []
    assert ran == samples


def test_a_failing_kernel_makes_the_run_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(sys, "argv", ["validate-dataset", "--machine", "rapper"])
    monkeypatch.setattr(validate, "selected_samples", lambda *_: [SAMPLE])
    monkeypatch.setattr(validate.builder, "build", lambda *_a, **_k: None)
    monkeypatch.setattr(
        validate,
        "run",
        lambda *_a, **_k: [validate.Result(SAMPLE, False, "Test Failed: nope")],
    )

    with pytest.raises(SystemExit) as exit:
        validate.main()

    assert exit.value.code == 1
    printed = capsys.readouterr().out
    assert "1 of 1 kernels failed" in printed
    # The shape and the variant say which kernel to go and look at.
    assert "3x5x7" in printed
    assert "libxsmm" in printed


def test_a_build_failure_exits_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(sys, "argv", ["validate-dataset", "--machine", "rapper"])

    def fails(*_args, **_kwargs):
        raise validate.BuildFailed("clang: error: no such file")

    monkeypatch.setattr(validate, "validate", fails)

    with pytest.raises(SystemExit) as exit:
        validate.main()

    assert exit.value.code == 1
    printed = capsys.readouterr().out
    assert "building the tests failed" in printed
    assert "no such file" in printed
