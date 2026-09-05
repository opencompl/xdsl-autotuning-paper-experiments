#!/usr/bin/env python3
"""Compile a linalg payload to an object file with a Lighthouse pipeline.

This is the code generation step of the ``lighthouse`` variant. It takes the
same tensor-level ``linalg.matmul`` payload that the MLIR variants in this
repository start from, hands it to the pipeline descriptor that Lighthouse
selects for the host, and writes the resulting object file.

The pipeline descriptor and every transform schedule it includes come from the
installed ``lighthouse`` package (see ``nix/lighthouse.nix``), so the numbers we
report are produced by the upstream schedules rather than by a copy of them.

This script runs under the Lighthouse interpreter provided by the flake
(``lighthouse-python``), *not* under the project's uv environment, and therefore
must not import anything from ``autotuner``.
"""

import argparse
import sys
from pathlib import Path

from mlir import ir

import lighthouse.dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.execution.target import TargetInfo
from lighthouse.pipeline import find_pipeline_file
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import PipelineDriver
from lighthouse.schedule.func import convert_function_results
from lighthouse.utils.importer import import_mlir_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="tensor-level MLIR payload")
    parser.add_argument(
        "--output", type=Path, required=True, help="object file to write"
    )
    parser.add_argument(
        "--entry", default="matmul", help="name of the payload function to compile"
    )
    parser.add_argument(
        "--symbol",
        help="rename the entry point to this symbol; use it to keep the emitted "
        "function from colliding with the `matmul` the C shim exports",
    )
    parser.add_argument(
        "--pipeline",
        default="matmul",
        help="name of the Lighthouse pipeline to look up for the host target",
    )
    parser.add_argument("--dtype", default="f32", help="pipeline data type")
    parser.add_argument(
        "-O", type=int, default=3, help="optimization level for the object file"
    )
    parser.add_argument(
        "--print-mlir-after-all",
        action="store_true",
        help="print the module after every pipeline stage",
    )
    return parser.parse_args()


def rename_entry_point(module: ir.Module, entry: str, symbol: str) -> None:
    """Rename the `entry` function to `symbol`.

    The payload's entry point is not called from anywhere inside the module, so
    renaming the symbol is enough; there are no call sites to update.
    """
    for op in module.body.operations:
        if "sym_name" in op.attributes and op.attributes["sym_name"].value == entry:
            op.attributes["sym_name"] = ir.StringAttr.get(symbol)
            return
    raise SystemExit(f"error: no function named '{entry}' in the payload")


def main() -> int:
    args = parse_args()

    target = TargetInfo.host()
    pipeline_file, feature = find_pipeline_file(target, args.pipeline, args.dtype)
    if pipeline_file is None:
        raise SystemExit(
            f"error: lighthouse ships no '{args.pipeline}' pipeline for "
            f"{args.dtype} on {target.arch}"
        )
    print(
        f"lighthouse: {target.arch} {args.pipeline}/{args.dtype}"
        f"{f' [{feature}]' if feature else ''}: {pipeline_file}",
        file=sys.stderr,
    )

    context = ir.Context()
    with context:
        lh_dialects.register_and_load()
    module = import_mlir_module(str(args.payload), context)

    entry = args.entry
    if args.symbol and args.symbol != entry:
        with context:
            rename_entry_point(module, entry, args.symbol)
        entry = args.symbol

    # Emit the `_mlir_ciface_<entry>` wrapper the C shim calls. Has to happen
    # while the entry point is still a `func.func`.
    Runner.make_function_callable(module, entry)

    driver = PipelineDriver(context)
    # Turn the returned tensor into a leading destination argument, so the shim
    # hands the kernel the caller's buffer instead of getting a freshly
    # allocated one back on every call. Same reason lighthouse's own
    # `kernel-bench` runs this transform ahead of the pipeline.
    with context:
        driver.add_transform(convert_function_results(entry))
    driver.add_stage(Descriptor(pipeline_file))
    optimized = driver.apply(module, print_after_all=args.print_mlir_after_all)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Runner(optimized, opt_level=args.O, target=target).dump_object_file(
        str(args.output)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
