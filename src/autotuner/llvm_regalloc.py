"""Count what LLVM's register allocator inserts into CompXSMM's kernels.

`autotuner.x86_mir` prints a kernel's register-unallocated IR as LLVM Machine
IR; llc allocates it with `LLC_FLAGS`, and the allocator's own optimisation
remarks report the spills, reloads, and copies it inserted.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import yaml
from xdsl.context import Context

from autotuner.x86_mir import LLC_FLAGS, print_mir


@dataclass(frozen=True)
class RegallocStats:
    """What LLVM's register allocator inserted.

    Spills and reloads are LLVM's own `N-byte Spill`/`Reload` annotations of
    stack-slot accesses in the emitted assembly, which every allocator gets;
    a folded reload is a spill slot read directly as a memory operand.
    `copies` are the register copies only the greedy allocator reports.
    """

    spills: int = 0
    folded_spills: int = 0
    reloads: int = 0
    folded_reloads: int = 0
    copies: int = 0

    @property
    def spilled(self) -> bool:
        return any((self.spills, self.folded_spills, self.reloads, self.folded_reloads))


SPILL_ANNOTATION = re.compile(r"# \d+-byte (Folded )?(Spill|Reload)$", re.MULTILINE)


class RemarkLoader(yaml.SafeLoader):
    """Reads LLVM's remark documents, whose tags (`!Missed` ...) carry no data."""


RemarkLoader.add_multi_constructor(
    "!", lambda loader, _, node: loader.construct_mapping(node, deep=True)
)


def parse_output(asm: str, remarks: str) -> RegallocStats:
    counts = {"copies": 0}
    for folded, kind in SPILL_ANNOTATION.findall(asm):
        name = f"{'folded_' if folded else ''}{kind.lower()}s"
        counts[name] = counts.get(name, 0) + 1
    for remark in yaml.load_all(remarks, Loader=RemarkLoader):
        args = {k: v for arg in remark.get("Args", ()) for k, v in arg.items()}
        if remark["Name"] == "SpillReloadCopies":
            counts["copies"] += int(args.get("NumVRCopies", 0))
    return RegallocStats(**counts)


def llvm_allocate(
    mir: str, mcpu: str, *, regalloc: str = "greedy", asm: Path | None = None
) -> RegallocStats:
    """Allocate `mir` with llc and report what the allocator inserted."""
    with tempfile.TemporaryDirectory() as tmp:
        source, remarks = Path(tmp, "kernel.mir"), Path(tmp, "remarks.yaml")
        asm = asm or Path(tmp, "kernel.s")
        source.write_text(mir)
        subprocess.run(
            [
                "llc",
                *LLC_FLAGS,
                "-mtriple=x86_64-unknown-linux-gnu",
                f"-mcpu={mcpu}",
                "-verify-machineinstrs",
                f"-regalloc={regalloc}",
                f"-pass-remarks-output={remarks}",
                "-pass-remarks-filter=regalloc",
                "-o",
                str(asm),
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_output(asm.read_text(), remarks.read_text())


# Command line


def kernel_mir(path: Path, passes: str) -> str:
    from io import StringIO

    from xdsl.parser import Parser
    from xdsl.passes import PassPipeline
    from xdsl.universe import Universe

    multiverse = Universe.get_multiverse()
    ctx = Context()
    for name, dialect in multiverse.all_dialects.items():
        ctx.register_dialect(name, dialect)
    module = Parser(ctx, path.read_text(), str(path)).parse_module()
    PassPipeline.parse_spec(multiverse.all_passes, passes).apply(ctx, module)
    out = StringIO()
    print_mir(module, out)
    return out.getvalue()


@dataclass(frozen=True)
class Job:
    path: Path
    passes: str
    mcpu: str
    regalloc: str
    keep: Path | None


def run_job(job: Job) -> tuple[Path, RegallocStats | str]:
    try:
        mir = kernel_mir(job.path, job.passes)
        asm = None
        if job.keep is not None:
            stem = job.keep / job.path.parent.name / job.path.name.split(".")[1]
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.with_suffix(".mir").write_text(mir)
            asm = stem.with_suffix(".s")
        return job.path, llvm_allocate(mir, job.mcpu, regalloc=job.regalloc, asm=asm)
    except (subprocess.CalledProcessError, NotImplementedError, KeyError) as e:
        detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else e
        return job.path, f"{type(e).__name__}: {detail}".strip()


def kernel_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("**/compxsmm.*.compxsmm.mlir"))
        else:
            yield path


def main(argv: Sequence[str] | None = None) -> None:
    from autotuner.machines import MACHINES

    parser = argparse.ArgumentParser(
        description="Allocate CompXSMM kernels' registers with LLVM and report spills."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="compxsmm.*.compxsmm.mlir generator outputs, or directories of them",
    )
    parser.add_argument("--machine", default=os.environ.get("MACHINE"))
    parser.add_argument("--config", type=Path, default=Path("default.yaml"))
    parser.add_argument(
        "--pipeline",
        default="compxsmm-llvm-gemm-passes",
        help="the pipeline in --config that leaves registers to LLVM",
    )
    parser.add_argument(
        "--regalloc", default="greedy", choices=["greedy", "basic", "pbqp", "fast"]
    )
    parser.add_argument(
        "--keep", type=Path, help="write each kernel's .mir and allocated .s here"
    )
    parser.add_argument("--csv", type=Path, help="write per-kernel results here")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)

    if not args.machine:
        parser.error("no machine given; pass --machine or set MACHINE")
    machine = MACHINES[args.machine]
    pipelines = yaml.safe_load(args.config.read_text())
    passes = ",".join(pipelines[args.pipeline][machine.isa])
    jobs = [
        Job(path, passes, machine.march, args.regalloc, args.keep)
        for path in kernel_files(args.inputs)
    ]

    names = [f.name for f in fields(RegallocStats)]
    rows: list[tuple[dict[str, object], RegallocStats]] = []
    failures = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for path, result in pool.map(run_job, jobs, chunksize=4):
            kernel = f"{path.parent.name}/{path.name.split('.')[1]}"
            if isinstance(result, str):
                failures += 1
                print(f"{kernel}: {result}", file=sys.stderr)
                continue
            row = {"kernel": kernel, **{n: getattr(result, n) for n in names}}
            rows.append((row, result))

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, ["kernel", *names])
            writer.writeheader()
            writer.writerows(row for row, _ in rows)

    spilled = [row for row, stats in rows if stats.spilled]
    for row in spilled:
        detail = ", ".join(f"{n}={row[n]}" for n in names if row[n])
        print(f"{row['kernel']}: {detail}")
    print(
        f"{len(spilled)} of {len(rows)} kernels spill under LLVM's {args.regalloc} "
        f"allocator ({failures} failed to convert)"
    )
    if failures:
        raise SystemExit(1)
