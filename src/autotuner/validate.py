"""Build and run each dataset's correctness test.

    uv run validate-dataset --machine rapper
    uv run validate-dataset --machine rapper f64.squares

Two phases, the way `autotuner.evaluate` has three:

1. `autotuner.build` compiles the same kernels the datasets measure, linked
   against `test.c` rather than `time.c`, across every core;
2. every test is run, also across every core -- unlike a measurement, a
   comparison against the reference does not care what else the machine is
   doing.

The datasets are validated together, so a shape two of them share is compiled
and checked once.  Each test's output is kept in the `test.log` beside its
binary, which is where the Snakefile's `validation` rule used to write it.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
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
from rich.table import Table

from autotuner import build as builder
from autotuner.build import BuildFailed, selected_samples, workers
from autotuner.datasets import Sample
from autotuner.machines import MACHINES, Machine

# How many failing kernels are named before the rest are counted.
FAILURES_SHOWN = 20

console = Console()


@dataclass(frozen=True)
class Result:
    """What one test binary did."""

    sample: Sample
    passed: bool
    # Why it counts as a failure, in one line; empty when it passed.
    reason: str = ""


def binary(sample: Sample, machine: str) -> Path:
    return Path(sample.path(machine, "test.o"))


def label(sample: Sample) -> str:
    return f"{sample.m}x{sample.n}x{sample.k} {sample.variant}"


def why(done: subprocess.CompletedProcess) -> str:
    """One line saying how this run failed.

    The harness ends with its own verdict when it ran to completion, so that
    line is the useful one; a kernel that trapped instead has only its exit
    status and whatever it managed to print on stderr.
    """
    printed = done.stdout.strip().splitlines()
    if printed and printed[-1].startswith("Test Failed"):
        return printed[-1]
    status = f"exited with {done.returncode}"
    errors = done.stderr.strip().splitlines()
    return f"{status}: {errors[-1]}" if errors else status


def check(sample: Sample, machine_name: str) -> Result:
    """Run one test binary, keeping its output in the log beside it."""
    machine: Machine = MACHINES[machine_name]
    # The same single-threaded environment the measurements run under, so a
    # vendor kernel is validated in the configuration it is timed in.
    env = (
        os.environ
        | {"OMP_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1"}
        | dict(machine.env)
    )
    executable = binary(sample, machine_name)
    try:
        done = subprocess.run(
            [str(executable)], env=env, capture_output=True, text=True, check=False
        )
    except OSError as error:
        return Result(sample, False, f"{error}")

    executable.with_suffix(".log").write_text(done.stdout)
    if done.returncode:
        return Result(sample, False, why(done))
    return Result(sample, True)


def run(
    samples: Sequence[Sample], machine: str, jobs: int | None = None
) -> list[Result]:
    """Run every sample's test, showing how far along the run is."""
    failures: list[Result] = []
    progress = Progress(
        TextColumn("[bold]validating[/bold] {task.fields[shape]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress, ThreadPoolExecutor(max_workers=workers(jobs)) as pool:
        task = progress.add_task("", total=len(samples), shape="")
        for one, result in zip(
            samples, pool.map(partial(check, machine_name=machine), samples)
        ):
            progress.update(task, shape=label(one), advance=1)
            if not result.passed:
                failures.append(result)
    return failures


def report(failures: Sequence[Result], total: int) -> None:
    """Say what failed, or that nothing did."""
    if not failures:
        console.print(f"[bold green]validated[/bold green] {total} kernels")
        return

    table = Table(header_style="bold", show_edge=False)
    table.add_column("shape")
    table.add_column("variant")
    table.add_column("dtype")
    table.add_column("why", overflow="fold")
    for failed in failures[:FAILURES_SHOWN]:
        one = failed.sample
        table.add_row(f"{one.m}x{one.n}x{one.k}", one.variant, one.dtype, failed.reason)

    console.print(f"[bold red]{len(failures)} of {total} kernels failed[/bold red]")
    console.print(table)
    if len(failures) > FAILURES_SHOWN:
        console.print(f"... and {len(failures) - FAILURES_SHOWN} more")
    console.print("each kernel's output is in the test.log beside its binary")


def validate(
    machine: str,
    names: Sequence[str] | None = None,
    *,
    build: bool = True,
    jobs: int | None = None,
) -> list[Result]:
    """Build and run both phases over the named datasets."""
    samples = selected_samples(machine, names)
    if not samples:
        print(f"{machine} defines no samples for these datasets", file=sys.stderr)
        return []

    if build:
        console.print(f"[bold]building[/bold] tests for {len(samples)} kernels")
        builder.build(samples, machine, jobs=jobs, driver="test")

    failures = run(samples, machine, jobs)
    report(failures, len(samples))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run the correctness test for every dataset sample."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="datasets to validate (default: every one this machine defines)",
    )
    parser.add_argument(
        "--machine",
        default=os.environ.get("MACHINE"),
        help="machine to validate for (default: $MACHINE)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="skip phase 1 and run what is already built",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="how many kernels to build and run at once (default: every core)",
    )
    args = parser.parse_args()

    if not args.machine:
        parser.error("no machine given; pass --machine or set MACHINE")

    try:
        failures = validate(
            args.machine,
            args.datasets or None,
            build=not args.no_build,
            jobs=args.jobs,
        )
    except BuildFailed as failure:
        console.print("[bold red]building the tests failed[/bold red]")
        console.print(f"{failure}", highlight=False)
        raise SystemExit(1) from None

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
