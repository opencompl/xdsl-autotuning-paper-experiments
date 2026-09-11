import json
from collections import defaultdict
from pathlib import Path

import pytest

from autotuner.datasets import (
    NANOKERNEL_GRID_K,
    NANOKERNEL_GRID_M,
    NANOKERNEL_GRID_N,
    NANOKERNEL_VARIANTS,
    VARIANTS,
    Sample,
    dataset_repeats,
    dataset_samples,
    default_variants,
    machine_file,
    variants_for,
)
from autotuner.machines import NEON, RAPPER, TOWER, Machine

# Datasets committed to the repo, which the sample order has to keep matching.
# The Grid'5000 clusters belong here as much as the machines named in
# `VARIANTS` do: what they measure is derived from rapper rather than written
# out (`datasets.AVX512_REFERENCE`), so this is what catches the derivation
# drifting away from what was actually measured.  `grvingt` is the one left
# out: its committed run predates that derivation and still holds a `naive_c`
# the definition no longer asks for, so it is a re-run away from belonging.
COMMITTED = [
    (machine, dataset)
    for machine in ("rapper", "tower", "chirop")
    # Not every machine has run every one of these, and the test skips a
    # dataset that is absent.
    for dataset in (
        "f32.ttile",
        "f64.ttile",
        "f64.small_matrices",
        "f32.squares",
        "f64.squares",
        "f64.nanokernel_grid",
    )
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

    # A dataset only has to hold samples the generator still asks for, in the
    # order it asks for them: then re-deriving the file leaves every committed
    # measurement where it is.  It may hold fewer -- widening a sweep leaves the
    # machines that have not re-run it since with a subset -- but never a sample
    # this machine no longer measures, and never in another order.
    measured = [s for s in generated if s in set(recorded)]

    # A repeated dataset is pass-major, so the file is that subset written out
    # once per pass.  At most the passes the dataset asks for and at least one:
    # a file collected before its repeat count went up holds fewer blocks.
    assert measured
    assert len(recorded) % len(measured) == 0
    passes = len(recorded) // len(measured)
    assert 1 <= passes <= dataset_repeats(dataset)
    assert measured * passes == recorded


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


def test_only_the_short_running_datasets_are_measured_more_than_once() -> None:
    assert dataset_repeats("f64.nanokernel_grid") == 3
    assert dataset_repeats("f32.squares") == 3
    assert dataset_repeats("f64.squares") == 3
    assert dataset_repeats("f32.ttile") == 1


def test_a_machine_without_a_variant_list_yields_no_samples() -> None:
    assert dataset_samples("neon")["f32.squares"] == []
    assert dataset_samples("neon")["f64.squares"] == []
    assert dataset_samples("neon")["f64.nanokernel_grid"] == []


@pytest.mark.parametrize(
    ("dataset", "variants"), [("f32.squares", 8), ("f64.squares", 8)]
)
def test_the_square_sweep_keeps_every_dimension_equal(
    dataset: str, variants: int
) -> None:
    samples = dataset_samples("rapper")[dataset]

    assert len(samples) == 64 * variants
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
    # narrow fsdbcst spans one M vector too, just in the narrowest bank that
    # covers it, so it draws the same top rows as fsdbcst -- which is what
    # makes the pair comparable panel by panel.  Its N reaches 30 rather than
    # 28, one accumulator per register left over, but the sweep stops at 7.
    assert tiles["llvm-skx-narrow-fsdbcst"] == tiles["libxsmm-skx-fsdbcst"]
    # Once the tile's M takes four vectors, above 24, only six accumulator
    # columns are left, so the tallest tiles stop short of the last column.
    assert swept - tiles["libxsmm-skx-nofsdbcst"] == {
        (m, n) for m, n in swept if m > 24 and n > 6
    }

    # Every tile that is measured is measured over the whole of K.
    assert all(ks == set(NANOKERNEL_GRID_K) for ks in k_by_tile.values())
    assert len(samples) == len(k_by_tile) * len(NANOKERNEL_GRID_K)


# --- machines that are detected rather than written out ---------------------


def test_a_reported_machine_is_never_derived() -> None:
    # `default_variants` is for machines nobody wrote an entry for; a published
    # machine must keep the frozen list even where the two would agree.
    for name in VARIANTS:
        assert variants_for(name) is VARIANTS[name]


def test_a_detected_avx512_machine_gets_the_full_comparison() -> None:
    # What a dataset compares follows from the hardware, so a detected avx512
    # machine reaches every variant the fullest machine does, whatever its own
    # vendor: the derived list is rapper's, tower's frozen one being a machine
    # whose committed measurements predate `compxsmm_plusnarrow`.
    assert default_variants(RAPPER) == VARIANTS["rapper"]
    assert default_variants(TOWER) == VARIANTS["rapper"]


def test_a_detected_machine_can_draw_the_squares_figure() -> None:
    # `plot_squares` draws its variant list unconditionally and raises if the
    # dataset lacks one, so what it draws has to be what a new machine measures.
    from autotuner.plot_squares import VARIANTS as DRAWN

    assert set(DRAWN) <= set(default_variants(RAPPER)["f64.squares"])


def test_a_detected_machine_without_avx512_is_baselines_only() -> None:
    # Neither our xdsl pipeline nor the pinned nano-kernels exist off avx512,
    # so there is nothing of ours to measure and the sweeps are baselines.
    arm = Machine(
        family="neoverse-v1",
        isa="neon",
        display_name="detected arm",
        target_triple="aarch64-unknown-linux-gnu",
        march="armv8.4-a",
        mtune="neoverse-v1",
        libxsmm_arch=None,
        freq=2.6,
        peak_f32=0,
        libs=(),
        linker_flag="",
        env={},
    )
    derived = default_variants(arm)

    assert derived["ttile"] == ["mkl", "aocl"]
    assert derived["f64.small_matrices"] == []
    assert derived["f32.squares"] == []
    assert derived["f64.squares"] == []
    assert derived["f64.nanokernel_grid"] == []


def test_every_dataset_is_covered_by_the_fallback() -> None:
    # A dataset added to `VARIANTS` but not to the baselines-only branch would
    # raise a KeyError deep inside `dataset_samples`, for a detected machine
    # off avx512 only.  The avx512 branch derives its keys, so it cannot drift.
    assert set(default_variants(RAPPER)) == set(VARIANTS["rapper"])
    assert set(default_variants(NEON)) == set(VARIANTS["neon"])


def test_a_detected_machine_does_not_share_the_reference_lists() -> None:
    # The derived dict is a copy: a caller sorting or appending to what it got
    # back must not edit the frozen table underneath it.
    derived = default_variants(TOWER)
    derived["ttile"].append("nonsense")
    assert "nonsense" not in VARIANTS["tower"]["ttile"]


def test_an_unknown_machine_says_how_to_detect_one() -> None:
    with pytest.raises(KeyError, match="machine-profile --name nowhere"):
        variants_for("nowhere")
