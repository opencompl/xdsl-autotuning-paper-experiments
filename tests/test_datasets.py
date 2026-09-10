import json
from collections import defaultdict
from pathlib import Path

import pytest

from autotuner.datasets import (
    NANOKERNEL_GRID_K,
    NANOKERNEL_GRID_M,
    NANOKERNEL_GRID_N,
    NANOKERNEL_VARIANTS,
    Sample,
    dataset_samples,
    machine_file,
)

# Datasets committed to the repo, which the sample order has to keep matching.
COMMITTED = [
    (machine, dataset)
    for machine in ("rapper", "tower")
    for dataset in ("f32.ttile", "f64.ttile", "f64.small_matrices", "f64.squares")
    # Only rapper has run the grid, and the test skips a dataset that is absent.
] + [("rapper", "f64.nanokernel_grid")]


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

    # A dataset only has to hold samples the generator still asks for, in the
    # order it asks for them: then re-deriving the file leaves every committed
    # measurement where it is.  It may hold fewer -- widening a sweep leaves the
    # machines that have not re-run it since with a subset -- but never a sample
    # this machine no longer measures, and never in another order.
    assert recorded == [s for s in generated if s in set(recorded)]


def test_a_sample_knows_where_its_files_live() -> None:
    sample = Sample(3, 5, 7, "libxsmm", "f64")

    assert sample.path("rapper", "time.o") == (
        "build/rapper/matmul_colmaj/3x5x7/libxsmm.f64.time.o"
    )


def test_the_path_helper_still_spells_out_wildcards() -> None:
    # The Snakefile uses the same helper to write rule inputs and outputs.
    assert (
        machine_file("S") == "build/{machine}/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.S"
    )


def test_a_machine_without_a_variant_list_yields_no_samples() -> None:
    assert dataset_samples("neon")["f64.squares"] == []
    assert dataset_samples("neon")["f64.nanokernel_grid"] == []


def test_the_square_sweep_keeps_every_dimension_equal() -> None:
    samples = dataset_samples("rapper")["f64.squares"]

    assert len(samples) == 64 * 4
    assert all(s.m == s.n == s.k for s in samples)
    assert {s.m for s in samples} == set(range(1, 65))


def test_the_nanokernel_grid_only_measures_supported_tiles() -> None:
    samples = dataset_samples("rapper")["f64.nanokernel_grid"]
    swept = {(m, n) for m in NANOKERNEL_GRID_M for n in NANOKERNEL_GRID_N}
    k_by_tile: defaultdict[tuple[str, int, int], set[int]] = defaultdict(set)
    for s in samples:
        k_by_tile[s.variant, s.m, s.n].add(s.k)
    tiles = {
        variant: {(m, n) for v, m, n in k_by_tile if v == variant}
        for variant in NANOKERNEL_VARIANTS
    }

    # The tile's M is the matrix's M.  fsdbcst spans one f64 vector of it, so
    # it reaches only the top rows of the grid; nofsdbcst spans up to four,
    # which covers those rows too, so the two overlap there rather than
    # dividing the sweep between them.  The 28-column limit both share on the
    # tile's N never binds here: the grid sweeps N no further than 7.
    assert tiles["libxsmm-skx-fsdbcst"] == {
        (m, n) for m, n in swept if m <= 8 and n <= 28
    }
    # Once the tile's M takes four vectors, above 24, only six accumulator
    # columns are left, so the tallest tiles stop short of the last column.
    assert swept - tiles["libxsmm-skx-nofsdbcst"] == {
        (m, n) for m, n in swept if m > 24 and n > 6
    }

    # Every tile that is measured is measured over the whole of K.
    assert all(ks == set(NANOKERNEL_GRID_K) for ks in k_by_tile.values())
    assert len(samples) == len(k_by_tile) * len(NANOKERNEL_GRID_K)
