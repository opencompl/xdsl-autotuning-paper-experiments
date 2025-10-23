from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.x86_func import FuncOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from autotuner.libxsmm_gemm.generator_common import libxsmm_mmfunction_signature
from autotuner.libxsmm_gemm.libxsmm_cpuid import Arch
from autotuner.libxsmm_gemm.libxsmm_main import GEMMDescriptor


def libxsmm_generator_gemm_directasm(
    file_out: Path, routine_name: str, desc: GEMMDescriptor, arch: Arch
):
    module_op = ModuleOp(Region(Block()))

    # Instead of function signature, generate `FuncOp`.
    func_op = libxsmm_mmfunction_signature(module_op, routine_name)

    # Generate the actual kernel code for current description depending on the
    # architecture.
    libxsmm_generator_gemm_kernel(func_op, arch, desc)

    # Append code to source file
    with open(file_out, "w") as f:
        Printer(stream=f).print_op(func_op)


def libxsmm_generator_gemm_kernel(
    func_op: FuncOp, arch: Arch, desc: GEMMDescriptor
): ...
