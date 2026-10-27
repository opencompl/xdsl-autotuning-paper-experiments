"""Standalone unary EXP entry point, parallel to generator_gemm.py.

Only the initial experiment's packed FP32 column vectors on SKX are ported.
The polynomial degree is explicit rather than an ambient LIBXSMM hint.
"""

from pathlib import Path

from xdsl.builder import Builder
from xdsl.dialects import x86
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86.registers import RDI
from xdsl.dialects.x86_func import FuncOp, RetOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from xdsl.rewriter import InsertPoint

from autotuner.libxsmm_gemm.generator_mateltwise_unary_binary_avx_avx512 import (
    libxsmm_generator_unary_binary_avx512_kernel,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_generator import GeneratedCode
from autotuner.libxsmm_gemm.libxsmm_main import MeltwDescriptor
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def libxsmm_generator_mateltwise_kernel(
    func: FuncOp, arch: Arch, desc: MeltwDescriptor, degree: int
) -> None:
    if arch != Arch.LIBXSMM_X86_AVX512_SKX:
        raise ValueError("EXP currently supports only skx")
    if degree not in (3, 5):
        raise ValueError("EXP polynomial degree must be 3 or 5")
    if not (
        desc.datatype == Datatype.F32
        and desc.m > 0
        and desc.m % 16 == 0
        and desc.n == 1
        and desc.ldi == desc.ldo == desc.m
    ):
        raise ValueError(
            "EXP requires packed FP32 column vectors, M > 0 divisible by 16"
        )
    code = GeneratedCode(Builder(InsertPoint.at_end(func.body.block)), arch)
    libxsmm_generator_unary_binary_avx512_kernel(code, desc, degree)
    code.insert(RetOp())


def libxsmm_generator_mateltwise_directasm(
    file_out: Path, routine_name: str, desc: MeltwDescriptor, arch: Arch, degree: int
) -> None:
    # Like matmul, emit typed, allocated x86 IR; xdsl-opt prints assembly.
    func = FuncOp(routine_name, Region(Block(arg_types=(RDI,))), ((RDI,), ()), "public")
    # The timing harness copies this self-contained kernel (including its
    # inline constants) to the same executable buffer as the native JIT code.
    module = ModuleOp(
        [
            func,
            x86.ops.DirectiveOp(".globl", f"{routine_name}_end"),
            x86.ops.LabelOp(f"{routine_name}_end"),
        ]
    )
    libxsmm_generator_mateltwise_kernel(func, arch, desc, degree)
    module.verify()
    with file_out.open("w") as f:
        Printer(stream=f).print_op(module)
