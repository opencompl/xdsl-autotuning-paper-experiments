"""Collect LLVM's static register-allocation counts for the XTC datasets.

    uv run collect-spills --machine tower f64.squares f64.small_matrices

This compiles kernels, but never executes or times them. Function totals include
folded accesses; loop remarks are overlapping breakdowns, not additional spills.
"""

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import yaml

from autotuner import build
from autotuner.datasets import dataset_samples
from autotuner.machines import MACHINES


class RemarkLoader(yaml.SafeLoader):
    """LLVM tags its register-allocation remarks with ``!Missed``."""


RemarkLoader.add_constructor("!Missed", RemarkLoader.construct_mapping)


def read_spills(path: Path) -> dict[str, int]:
    """Read the function total, not the overlapping loop or helper totals.

    LLVM omits zero-valued fields, and can emit an empty file for a spill-free
    kernel. Only a successfully built, existing report can represent zero:
    missing files and malformed YAML are errors, never zero measurements.
    """
    records = [
        record
        for record in yaml.load_all(path.read_text(), Loader=RemarkLoader)
        if record
        and record.get("Pass") == "regalloc"
        and record.get("Name") == "SpillReloadCopies"
        and record.get("Function") == "matmul"
    ]
    if len(records) > 1:
        raise ValueError(f"{path}: multiple register-allocation totals for matmul")
    args = {
        key: value
        for arg in (records[0]["Args"] if records else [])
        for key, value in arg.items()
    }
    folded_spills = int(args.get("NumFoldedSpills", 0))
    folded_reloads = int(args.get("NumFoldedReloads", 0))
    return {
        "spills": int(args.get("NumSpills", 0)) + folded_spills,
        "reloads": int(args.get("NumReloads", 0)) + folded_reloads,
        "folded_spills": folded_spills,
        "folded_reloads": folded_reloads,
    }


def collect(
    machine: str,
    names: Sequence[str] | None = None,
    *,
    root: Path = Path("build"),
    data_dir: Path = Path("data"),
    jobs: int | None = None,
    force: bool = False,
) -> None:
    """Build XTC reports and write one compiler-metrics row per sample."""
    available = {
        name: [s for s in samples if s.variant == "libxtcmm"]
        for name, samples in dataset_samples(machine).items()
    }
    available = {name: samples for name, samples in available.items() if samples}
    if names is not None and (unknown := set(names) - available.keys()):
        raise ValueError(f"no XTC samples for dataset(s): {', '.join(sorted(unknown))}")
    chosen = {
        name: samples
        for name, samples in available.items()
        if names is None or name in names
    }
    if not chosen:
        raise ValueError(f"{machine} defines no XTC datasets")
    samples = list(dict.fromkeys(s for group in chosen.values() for s in group))
    build.build(samples, machine, root=root, jobs=jobs, force=force, driver="spills")

    tool = build.toolchain(machine, root=root)
    llc = Path(tool.xtc_llvm_bin) / "bin" / "llc"
    version = subprocess.check_output([str(llc), "--version"], text=True)
    version = version.split("LLVM version ", 1)[1].splitlines()[0].strip()
    manifest = build.load_manifest(tool)
    spec = MACHINES[machine]
    for name, group in chosen.items():
        rows = []
        for sample in group:
            report = root / Path(sample.path(machine, "spills.yaml")).relative_to(
                "build"
            )
            rows.append(
                {
                    "M": sample.m,
                    "N": sample.n,
                    "K": sample.k,
                    "variant": sample.variant,
                    "dtype": sample.dtype,
                    "machine": machine,
                    "family": spec.family,
                    "isa": spec.isa,
                    "compiler_march": spec.march,
                    "libxsmm_arch": spec.libxsmm_arch,
                    "llvm_version": version,
                    "method": "llvm-regalloc-static",
                    "function": "matmul",
                    "build_key": manifest[str(report)],
                    **read_spills(report),
                }
            )
        output = data_dir / machine / f"{name}.spills.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        scratch = output.with_suffix(".tmp")
        scratch.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        )
        scratch.replace(output)
        build.console.print(f"[bold]wrote[/bold] {output} ({len(rows)} kernels)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*", help="datasets (default: all XTC datasets)"
    )
    parser.add_argument(
        "--machine", choices=MACHINES, default=os.environ.get("MACHINE")
    )
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("build"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.machine:
        parser.error("no machine given; pass --machine or set MACHINE")
    try:
        collect(
            args.machine,
            args.datasets or None,
            root=args.root,
            data_dir=args.data_dir,
            jobs=args.jobs,
            force=args.force,
        )
    except (build.BuildFailed, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
