import json
import os
import sys
from pathlib import Path

import pytest

from autotuner import evaluate
from autotuner.datasets import Sample
from autotuner.machines import MACHINES

# A row exactly as the shell rule this replaced used to write it.
LINE = (
    '{"M":1,"N":1,"K":64,"peak":16.0,"flops":128,"time":457.750560,'
    '"variant":"aocl","machine":"rapper","family":"zen4","isa":"avx512",'
    '"compiler_march":"znver4","libxsmm_arch":"skx","dtype":"f64"}\n'
)


def test_a_row_serialises_the_way_the_shell_rule_did() -> None:
    assert evaluate.as_json_line(json.loads(LINE)) == LINE


def test_cycles_keep_the_harness_six_decimals() -> None:
    # Plain float repr would write 457.75056 and churn every committed line.
    assert '"time":457.750560' in evaluate.as_json_line({"time": 457.75056})


def test_a_row_is_built_from_the_sample_and_the_machine() -> None:
    row = evaluate.row(Sample(1, 1, 64, "aocl", "f64"), "rapper", "457.750560")

    assert evaluate.as_json_line(row) == LINE


def test_only_a_repeated_sample_records_which_pass_it_came_from() -> None:
    sample = Sample(1, 1, 64, "aocl", "f64")

    # A single-pass dataset keeps the schema it was committed with...
    assert "repeat" not in evaluate.row(sample, "rapper", "457.750560")
    # ...and the pass number goes last, so a repeated one's rows share that
    # committed prefix rather than shifting every field along.
    numbered = evaluate.row(sample, "rapper", "457.750560", repeat=2)
    assert evaluate.as_json_line(numbered) == LINE[:-2] + ',"repeat":2}\n'


def test_f64_peak_is_half_the_f32_peak() -> None:
    single = evaluate.row(Sample(1, 1, 1, "libxsmm", "f32"), "rapper", "1.0")["peak"]
    double = evaluate.row(Sample(1, 1, 1, "libxsmm", "f64"), "rapper", "1.0")["peak"]

    assert double == single / 2


def test_an_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        evaluate.evaluate("rapper", ["not.a.dataset"], build=False)


def measured_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A timing binary and the measurement cached beside it."""
    binary = tmp_path / "libxsmm.f64.time.o"
    binary.write_text("")
    return binary, tmp_path / "libxsmm.f64.time.txt"


def test_each_pass_caches_its_own_measurement_beside_the_binary(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "libxsmm.f64.time.o"

    # The first pass keeps the name the Snakefile's `time` rule writes.
    assert evaluate.measurement_path(binary).name == "libxsmm.f64.time.txt"
    assert evaluate.measurement_path(binary, 1).name == "libxsmm.f64.time.2.txt"
    assert evaluate.measurement_path(binary, 2).name == "libxsmm.f64.time.3.txt"


def test_a_later_pass_does_not_reuse_the_first_pass_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, measured = measured_pair(tmp_path)
    measured.write_text("1.0\n")
    monkeypatch.setattr(evaluate, "run_kernel", lambda *_: "2.0")

    assert evaluate.cycles(binary, MACHINES["rapper"], 1) == "2.0"
    # Each pass is a measurement of its own, cached without disturbing the
    # others, so an interrupted multi-pass run resumes where it stopped.
    assert (tmp_path / "libxsmm.f64.time.2.txt").read_text() == "2.0\n"
    assert measured.read_text() == "1.0\n"


def test_a_measurement_newer_than_its_binary_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, measured = measured_pair(tmp_path)
    measured.write_text("1.0\n")
    monkeypatch.setattr(
        evaluate, "run_kernel", lambda *_: pytest.fail("should not have re-run")
    )

    assert evaluate.cycles(binary, MACHINES["rapper"]) == "1.0"


def test_a_rebuilt_binary_is_measured_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, measured = measured_pair(tmp_path)
    measured.write_text("1.0\n")
    os.utime(measured, (0, 0))
    monkeypatch.setattr(evaluate, "run_kernel", lambda *_: "2.0")

    assert evaluate.cycles(binary, MACHINES["rapper"]) == "2.0"
    assert measured.read_text() == "2.0\n"


def test_an_empty_measurement_file_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An interrupted run leaves the redirect's file behind but empty.
    binary, measured = measured_pair(tmp_path)
    measured.write_text("")
    monkeypatch.setattr(evaluate, "run_kernel", lambda *_: "3.0")

    assert evaluate.cycles(binary, MACHINES["rapper"]) == "3.0"


def test_datasets_sharing_a_sample_measure_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = Sample(2, 2, 2, "libxsmm", "f64")
    extra = Sample(3, 3, 3, "libxsmm", "f64")
    monkeypatch.setattr(
        evaluate, "dataset_samples", lambda _: {"a": [shared], "b": [shared, extra]}
    )
    timed: list[Path] = []

    def record(binary: Path, _machine, _index: int = 0) -> str:
        timed.append(binary)
        return "1.0"

    monkeypatch.setattr(evaluate, "cycles", record)

    evaluate.evaluate("rapper", data_dir=tmp_path, build=False)

    # Two distinct samples, so two runs -- not three.
    assert len(timed) == 2
    written = {
        p.name: p.read_text().splitlines()
        for p in (tmp_path / "rapper").glob("*.jsonl")
    }
    assert set(written) == {"a.jsonl", "b.jsonl"}
    assert len(written["a.jsonl"]) == 1
    assert len(written["b.jsonl"]) == 2
    # The shared sample is recorded in both datasets.
    assert written["a.jsonl"][0] == written["b.jsonl"][0]


def test_a_repeated_dataset_is_swept_pass_by_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = Sample(2, 2, 2, "libxsmm", "f64")
    large = Sample(3, 3, 3, "libxsmm", "f64")
    monkeypatch.setattr(
        evaluate, "dataset_samples", lambda _machine: {"grid": [small, large]}
    )
    monkeypatch.setattr(evaluate, "dataset_repeats", lambda _name: 3)
    cycled = iter(["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"])
    order: list[tuple[str, int]] = []

    def record(binary: Path, _machine, index: int = 0) -> str:
        order.append((binary.parent.name, index))
        return next(cycled)

    monkeypatch.setattr(evaluate, "cycles", record)

    evaluate.evaluate("rapper", data_dir=tmp_path, build=False)

    # Whole passes over the samples, not each sample's three runs back to back.
    assert order == [
        ("2x2x2", 0),
        ("3x3x3", 0),
        ("2x2x2", 1),
        ("3x3x3", 1),
        ("2x2x2", 2),
        ("3x3x3", 2),
    ]
    written = (tmp_path / "rapper" / "grid.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in written]
    assert [(r["M"], r["repeat"], r["time"]) for r in rows] == [
        (2, 1, 1.0),
        (3, 1, 2.0),
        (2, 2, 3.0),
        (3, 2, 4.0),
        (2, 3, 5.0),
        (3, 3, 6.0),
    ]


def test_a_shared_sample_is_measured_as_often_as_the_greedier_dataset_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = Sample(2, 2, 2, "libxsmm", "f64")
    monkeypatch.setattr(
        evaluate,
        "dataset_samples",
        lambda _machine: {"once": [shared], "grid": [shared]},
    )
    monkeypatch.setattr(
        evaluate, "dataset_repeats", lambda name: 3 if name == "grid" else 1
    )
    cycled = iter(["1.0", "2.0", "3.0"])
    monkeypatch.setattr(evaluate, "cycles", lambda *_a: next(cycled))

    evaluate.evaluate("rapper", data_dir=tmp_path, build=False)

    written = {
        p.name: p.read_text().splitlines()
        for p in (tmp_path / "rapper").glob("*.jsonl")
    }
    # Three runs, and the dataset that wanted one keeps the first of them.
    assert len(written["grid.jsonl"]) == 3
    assert written["once.jsonl"] == [
        written["grid.jsonl"][0].replace(',"repeat":1', "")
    ]


def test_the_build_only_sees_each_shared_sample_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shared = Sample(1, 1, 1, "libxsmm", "f64")
    monkeypatch.setattr(
        evaluate,
        "dataset_samples",
        lambda _machine: {"a": [shared], "b": [shared, Sample(2, 2, 2, "aocl", "f64")]},
    )
    monkeypatch.setattr(evaluate, "cycles", lambda *_: "1.0")

    asked: list[list[Sample]] = []
    monkeypatch.setattr(
        evaluate.builder, "build", lambda samples, *_a, **_k: asked.append(samples)
    )

    evaluate.evaluate("rapper", data_dir=tmp_path)

    assert asked == [[shared, Sample(2, 2, 2, "aocl", "f64")]]


def test_a_build_failure_exits_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate", "--machine", "rapper"])

    def fails(*_args, **_kwargs):
        raise evaluate.BuildFailed("LockException: directory cannot be locked")

    monkeypatch.setattr(evaluate, "evaluate", fails)

    with pytest.raises(SystemExit) as exit:
        evaluate.main()

    assert exit.value.code == 1
    printed = capsys.readouterr().out
    assert "generating the kernels failed" in printed
    # The failing command and its error come through, not a traceback.
    assert "LockException" in printed
