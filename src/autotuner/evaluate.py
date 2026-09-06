"""Build, measure and record the benchmark datasets.

    uv run evaluate --machine rapper
    uv run evaluate --machine rapper f64.mnk_grid

Three phases, because each wants something different from the machine:

1. generating and compiling the kernels is ordinary CPU work, so
   `autotuner.build` runs it across every core out of one process;
2. timing them has to be strictly serial, or the numbers measure contention;
3. writing the jsonl is bookkeeping.

The phases run over every requested dataset at once rather than one dataset at
a time, so a shape that two datasets share is compiled once and timed once.
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from autotuner import build as builder
from autotuner.build import BuildFailed
from autotuner.datasets import Sample, dataset_samples
from autotuner.machines import MACHINES, Machine

# Peak FLOP/cycle is quoted for f32; f64 halves the lanes.
DTYPE_PEAK_DIVISOR = {"f32": 1.0, "f64": 2.0}

console = Console()


# --- phase 1: generate and compile -----------------------------------------


def generate(samples: Sequence[Sample], machine: str, jobs: int | None = None) -> None:
    """Build every kernel these samples need, across every core."""
    builder.build(samples, machine, jobs=jobs)


# --- phase 2: measure, one kernel at a time --------------------------------


def run_kernel(binary: Path, machine: Machine) -> str:
    """Run one timing binary and return the line of cycles it prints."""
    env = (
        os.environ
        | {"OMP_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1"}
        | dict(machine.env)
    )
    try:
        done = subprocess.run(
            [str(binary)], env=env, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{binary} exited with {error.returncode}: {error.stderr.strip()}"
        ) from error

    line = done.stdout.strip()
    if not line:
        raise RuntimeError(f"{binary} printed no measurement: {done.stderr.strip()}")
    return line


def cycles(binary: Path, machine: Machine) -> str:
    """The measurement for ``binary``, running it only if it is out of date.

    The cache is the ``time.txt`` beside the binary, which is the same file the
    Snakefile's ``time`` rule writes for a one-off target.
    """
    measured = binary.with_suffix(".txt")
    if measured.exists() and measured.stat().st_mtime >= binary.stat().st_mtime:
        cached = measured.read_text().strip()
        # An interrupted run leaves the redirect's file behind but empty.
        if cached:
            return cached

    line = run_kernel(binary, machine)
    measured.write_text(line + "\n")
    return line


def measure(samples: Sequence[Sample], machine_name: str) -> dict[Sample, str]:
    """Time every sample in turn, showing how far along the run is."""
    machine = MACHINES[machine_name]
    measured: dict[Sample, str] = {}
    progress = Progress(
        TextColumn("[bold]measuring[/bold] {task.fields[shape]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("", total=len(samples), shape="")
        for one in samples:
            progress.update(task, shape=f"{one.m}x{one.n}x{one.k} {one.variant}")
            measured[one] = cycles(Path(one.path(machine_name, "time.o")), machine)
            progress.advance(task)
    return measured


# --- phase 3: write the datasets out ---------------------------------------


def row(one: Sample, machine_name: str, measured: str) -> dict:
    """One dataset row: the shape from the sample, the rest from the machine."""
    machine = MACHINES[machine_name]
    return {
        "M": one.m,
        "N": one.n,
        "K": one.k,
        "peak": machine.peak_f32 / DTYPE_PEAK_DIVISOR[one.dtype],
        # An FMA counts as two floating-point operations.
        "flops": 2 * one.m * one.n * one.k,
        "time": float(measured),
        "variant": one.variant,
        "machine": machine_name,
        "family": machine.family,
        "isa": machine.isa,
        "compiler_march": machine.march,
        "libxsmm_arch": machine.libxsmm_arch,
        "dtype": one.dtype,
    }


def as_json_line(fields: Mapping) -> str:
    """One row, formatted the way the shell rule this replaced wrote it.

    Cycles keep the six decimals the harness prints and the separators stay
    tight, so re-deriving a dataset whose measurements have not changed leaves
    the committed file byte-identical instead of reformatting every line.
    """
    body = ",".join(
        json.dumps(key) + ":" + (f"{value:.6f}" if key == "time" else json.dumps(value))
        for key, value in fields.items()
    )
    return "{" + body + "}\n"


def write(
    datasets: Mapping[str, Sequence[Sample]],
    machine: str,
    measured: Mapping[Sample, str],
    data_dir: Path,
) -> None:
    """Write each dataset's jsonl from the shared pool of measurements."""
    for name, samples in datasets.items():
        path = data_dir / machine / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(as_json_line(row(s, machine, measured[s])) for s in samples)
        )
        console.print(f"[bold]wrote[/bold] {path} ({len(samples)} samples)")


# --- driver ----------------------------------------------------------------


def evaluate(
    machine: str,
    names: Sequence[str] | None = None,
    *,
    data_dir: Path = Path("data"),
    build: bool = True,
    jobs: int | None = None,
) -> None:
    """Run all three phases over the named datasets."""
    defined = dataset_samples(machine)
    unknown = set(names or ()) - set(defined)
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(sorted(unknown))}")

    datasets = {
        name: samples
        for name, samples in defined.items()
        if samples and (names is None or name in names)
    }
    if not datasets:
        print(f"{machine} defines no samples for these datasets", file=sys.stderr)
        return

    total = sum(len(s) for s in datasets.values())

    # dict, not set: one entry per distinct sample, in first-requested order, so
    # a shape two datasets share is built and measured once.
    shared = dict.fromkeys(s for samples in datasets.values() for s in samples)
    if build:
        console.print(f"[bold]generating[/bold] code for {', '.join(datasets)}")
        generate(list(shared), machine, jobs)

    console.print(
        f"[bold]measuring[/bold] {len(shared)} kernels"
        + (f" ({total} samples, deduplicated)" if len(shared) != total else "")
    )

    write(datasets, machine, measure(list(shared), machine), data_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Build, measure and record the benchmark datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="datasets to evaluate (default: every one this machine defines)",
    )
    parser.add_argument(
        "--machine",
        default=os.environ.get("MACHINE"),
        help="machine to evaluate for (default: $MACHINE)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="where to write the jsonl"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip phase 1 and measure what is already built",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="how many kernels to compile at once (default: every core)",
    )
    args = parser.parse_args()

    if not args.machine:
        parser.error("no machine given; pass --machine or set MACHINE")

    try:
        evaluate(
            args.machine,
            args.datasets or None,
            data_dir=args.data_dir,
            build=not args.no_build,
            jobs=args.jobs,
        )
    except BuildFailed as failure:
        console.print("[bold red]generating the kernels failed[/bold red]")
        if failure.args and failure.args[0]:
            console.print(f"{failure.args[0]}", highlight=False)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
