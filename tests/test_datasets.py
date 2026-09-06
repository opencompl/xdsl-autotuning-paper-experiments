import json
from pathlib import Path

import pytest

from autotuner.datasets import Sample, dataset_samples, machine_file

# Datasets committed to the repo, which the sample order has to keep matching.
COMMITTED = [
    ("tower", "f64.small_matrices"),
    ("tower", "f64.ttile"),
    ("tower", "f32.ttile"),
    ("rapper", "f64.squares"),
]


@pytest.mark.parametrize(("machine", "dataset"), COMMITTED)
def test_sample_order_matches_the_committed_dataset(machine: str, dataset: str) -> None:
    path = Path("data") / machine / f"{dataset}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} is not checked out")

    recorded = [
        (row["M"], row["N"], row["K"], row["variant"], row["dtype"])
        for row in map(json.loads, path.read_text().splitlines())
    ]
    generated = [
        (s.m, s.n, s.k, s.variant, s.dtype) for s in dataset_samples(machine)[dataset]
    ]

    assert generated == recorded


def test_a_sample_knows_where_its_files_live() -> None:
    sample = Sample(3, 5, 7, "libxsmm", "f64")

    assert sample.path("rapper", "time.o") == (
        "build/rapper/matmul_rowmaj/3x5x7/libxsmm.f64.time.o"
    )


def test_the_path_helper_still_spells_out_wildcards() -> None:
    # The Snakefile uses the same helper to write rule inputs and outputs.
    assert (
        machine_file("S") == "build/{machine}/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.S"
    )


def test_a_machine_without_a_variant_list_yields_no_samples() -> None:
    assert dataset_samples("neon")["f64.squares"] == []


def test_the_square_sweep_keeps_every_dimension_equal() -> None:
    samples = dataset_samples("rapper")["f64.squares"]

    assert len(samples) == 64 * 4
    assert all(s.m == s.n == s.k for s in samples)
    assert {s.m for s in samples} == set(range(1, 65))
