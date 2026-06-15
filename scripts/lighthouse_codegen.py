"""Compile a parameterized matmul kernel via Lighthouse torch ingress.

Codegen-only entry point for Snakemake: imports an inline torch matmul model
(equivalent to KernelBench level1/2), runs the vendored x86_64/matmul/f32.yaml
pipeline, and writes a linkable object file via ``Runner.dump_object_file``.
"""

# `lighthouse.pipeline.stage` references `importlib.util` without importing it.
import importlib.util  # noqa: F401

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse import ingress as lh_ingress
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.descriptor import Descriptor
from lighthouse.pipeline.driver import PipelineDriver, make_function_callable
from lighthouse.schedule import convert_function_results

ENTRY_POINT = "main"


class MatmulModel(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


def define_compiler_pipeline(module: ir.Module, pipeline_yaml: Path) -> PipelineDriver:
    driver = PipelineDriver(module.context)
    with module.context:
        lh_dialects.register_and_load()
        driver.add_transform(convert_function_results(ENTRY_POINT))
        make_function_callable(module, ENTRY_POINT)
        driver.add_stage(Descriptor(str(pipeline_yaml.resolve())))
    return driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-O", type=int, default=3)
    args = parser.parse_args()

    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("m, n, and k must be positive")

    pipeline_abs = args.pipeline.resolve()
    if not pipeline_abs.is_file():
        raise FileNotFoundError(f"Pipeline descriptor not found: {pipeline_abs}")

    sample_a = torch.randn(args.m, args.k, dtype=torch.float32)
    sample_b = torch.randn(args.k, args.n, dtype=torch.float32)

    mlir_module = lh_ingress.torch.import_from_model(
        model=MatmulModel(),
        sample_args=(sample_a, sample_b),
        ir_context=ir.Context(),
    )

    driver = define_compiler_pipeline(mlir_module, pipeline_abs)
    optimized = driver.apply(mlir_module, print_after_all=False)

    runner = Runner(optimized, opt_level=args.O)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner.dump_object_file(str(args.output))


if __name__ == "__main__":
    main()
