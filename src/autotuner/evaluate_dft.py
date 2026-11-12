"""Build, check, measure and record the DFT datasets.

    uv run evaluate-dft --machine rapper

The DFT counterpart of `autotuner.evaluate`, in the same three phases.  There
are only a couple of hundred kernels, few enough that Snakemake can schedule
them without the startup cost `autotuner.build` exists to avoid, so phase 1
asks Snakemake for every timing binary together with the validation log of the
same kernel: nothing is measured unless it computes the DFT correctly.  Timing
and the jsonl format are `autotuner.evaluate`'s.

Throughput follows FFTW's convention (benchFFT's "mflops"): a complex DFT of
N points counts as 5 N log2 N floating-point operations, whatever the algorithm
actually executes, so the number compares implementations rather than counting
their work.
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from autotuner.datasets import DFT_SIZES, DFT_VARIANTS, dataset_repeats, dft_file
from autotuner.evaluate import DTYPE_PEAK_DIVISOR, as_json_line, console, cycles
from autotuner.machines import MACHINES

DTYPES = ("f32", "f64")

# One kernel: (N, variant, dtype), in the order its dataset records it.
Kernel = tuple[int, str, str]


def kernels(dtype: str) -> list[Kernel]:
    return [(n, variant, dtype) for n in DFT_SIZES for variant in DFT_VARIANTS]


def build(machine: str, needed: list[Kernel], jobs: int | None) -> None:
    """Build and validate every kernel, across every core."""
    targets = [
        dft_file(ext, machine=machine, n=n, variant=variant, dtype=dtype)
        for n, variant, dtype in needed
        for ext in ("test.log", "time.o")
    ]
    subprocess.run(
        [
            *(sys.executable, "-m", "snakemake"),
            *("--cores", str(jobs) if jobs else "all"),
            *("--config", f"machine={machine}"),
            *("--quiet", "all"),
            "--",
            *targets,
        ],
        check=True,
    )


def row(kernel: Kernel, machine_name: str, measured: str, repeat: int) -> dict:
    n, variant, dtype = kernel
    machine = MACHINES[machine_name]
    return {
        "N": n,
        "peak": machine.peak_f32 / DTYPE_PEAK_DIVISOR[dtype],
        "flops": 5 * n * math.log2(n),
        "time": float(measured),
        "variant": variant,
        "machine": machine_name,
        "family": machine.family,
        "isa": machine.isa,
        "compiler_march": machine.march,
        "dtype": dtype,
        "repeat": repeat,
    }


def evaluate(
    machine_name: str,
    dtypes: tuple[str, ...] = DTYPES,
    *,
    data_dir: Path = Path("data"),
    build_first: bool = True,
    jobs: int | None = None,
) -> None:
    datasets = {f"{dtype}.dft": kernels(dtype) for dtype in dtypes}
    if build_first:
        console.print(f"[bold]generating[/bold] code for {', '.join(datasets)}")
        build(machine_name, [k for ks in datasets.values() for k in ks], jobs)

    machine = MACHINES[machine_name]
    passes = max(dataset_repeats(name) for name in datasets)
    total = sum(len(ks) * dataset_repeats(name) for name, ks in datasets.items())
    rows: dict[str, list[str]] = {name: [] for name in datasets}

    # Whole passes over every kernel rather than each kernel's repeats back to
    # back, for the reason `evaluate.measure` gives.
    progress = Progress(
        TextColumn("[bold]measuring[/bold] {task.fields[what]}"),
        BarColumn(),
        MofNCompleteColumn(),
    )
    with progress:
        task = progress.add_task("", total=total, what="")
        for index in range(passes):
            for name, ks in datasets.items():
                if index >= dataset_repeats(name):
                    continue
                for kernel in ks:
                    n, variant, dtype = kernel
                    progress.update(
                        task, what=f"{variant} {dtype} N={n} pass {index + 1}"
                    )
                    binary = Path(
                        dft_file(
                            "time.o",
                            machine=machine_name,
                            n=n,
                            variant=variant,
                            dtype=dtype,
                        )
                    )
                    measured = cycles(binary, machine, index)
                    rows[name].append(
                        as_json_line(row(kernel, machine_name, measured, index + 1))
                    )
                    progress.advance(task)

    for name, lines in rows.items():
        path = data_dir / machine_name / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))
        console.print(f"[bold]wrote[/bold] {path} ({len(lines)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Build, check, measure and record the DFT datasets."
    )
    parser.add_argument(
        "dtypes",
        nargs="*",
        choices=DTYPES,
        help="data types to evaluate (default: both)",
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

    evaluate(
        args.machine,
        tuple(args.dtypes) or DTYPES,
        data_dir=args.data_dir,
        build_first=not args.no_build,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
