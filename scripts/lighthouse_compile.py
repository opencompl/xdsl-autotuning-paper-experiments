"""Run a Lighthouse pipeline YAML on a tensor-form MLIR payload.

This is the single deterministic entry point used by Snakemake for the
Lighthouse lowering path. It mirrors `lighthouse.pipeline.driver.CompilerDriver`
with one project-specific tweak: it sets `llvm.emit_c_interface` on a
named function before LLVM lowering, so the resulting LLVM IR exposes a
C-callable symbol with the same name as the linalg payload.
"""

# `lighthouse.pipeline.stage` references `importlib.util` without importing it.
# Importing it here ensures the attribute is bound on the `importlib` module
# before any Lighthouse stage tries to load a Python schedule.
import importlib.util  # noqa: F401

import argparse
from pathlib import Path

from lighthouse.pipeline.driver import CompilerDriver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the tensor-form MLIR payload.",
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        type=Path,
        help="Path to the Lighthouse pipeline YAML descriptor.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the lowered MLIR module.",
    )
    parser.add_argument(
        "--func",
        default="matmul",
        help="Name of the function to mark as C-callable (default: matmul).",
    )
    args = parser.parse_args()

    pipeline_abs = args.pipeline.resolve()
    if not pipeline_abs.is_file():
        raise FileNotFoundError(f"Pipeline descriptor not found: {pipeline_abs}")

    driver = CompilerDriver(str(args.input), stages=[str(pipeline_abs)])
    driver.make_function_callable(args.func)
    lowered = driver.run(print_after_all=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(str(lowered))


if __name__ == "__main__":
    main()
