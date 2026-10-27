"""Emit a schedule-neutral exp operation using LIBXSMM's unary ABI."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from xdsl.dialects import builtin, x86, x86_func
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from autotuner.dialects.xsmm import ExpOp


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=Path)
    parser.add_argument("routine_name")
    parser.add_argument("length", type=int)
    parser.add_argument("--degree", type=int, choices=(3, 5), required=True)
    parser.add_argument("--arch", choices=("skx",), default="skx")
    args = parser.parse_args(argv)
    if args.length <= 0 or args.length % 16:
        parser.error("length must be positive and divisible by 16")
    block = Block(arg_types=(x86.registers.RDI,))
    input = x86.ops.DM_MovOp(block.args[0], 32, destination=x86.registers.R8)
    output = x86.ops.DM_MovOp(block.args[0], 80, destination=x86.registers.R9)
    block.add_ops(
        (
            input,
            output,
            ExpOp(input.destination, output.destination, args.length, args.degree),
            x86_func.RetOp(),
        )
    )
    module = builtin.ModuleOp(
        [
            x86_func.FuncOp(
                args.routine_name, Region(block), ((x86.registers.RDI,), ()), "public"
            ),
            x86.ops.DirectiveOp(".globl", f"{args.routine_name}_end"),
            x86.ops.LabelOp(f"{args.routine_name}_end"),
        ]
    )
    module.verify()
    with args.filename.open("w") as stream:
        Printer(stream=stream).print_op(module)


if __name__ == "__main__":
    main()
