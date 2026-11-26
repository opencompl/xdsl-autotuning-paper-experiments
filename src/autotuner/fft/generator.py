"""Emit the ``fft.dft`` operation for a 1-D complex DFT as an x86 function.

The function is ``void dft(const T *in, T *out)`` over interleaved complex
arrays of ``n`` points: forward (sign -1), unnormalised, out of place.
"""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from xdsl.dialects import x86, x86_func
from xdsl.dialects.builtin import Float32Type, Float64Type, ModuleOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from autotuner.dialects.fft import DftOp


def dft_module(
    n: int, datatype: Float32Type | Float64Type, *, sign: int = -1, name: str = "dft"
) -> ModuleOp:
    block = Block(arg_types=(x86.registers.RDI, x86.registers.RSI))
    block.add_ops(
        [
            DftOp(
                block.args[0],
                block.args[1],
                n=n,
                sign=sign,
                istride=1,
                ostride=1,
                datatype=datatype,
            ),
            x86_func.RetOp(),
        ]
    )
    func = x86_func.FuncOp(
        name,
        Region(block),
        ((x86.registers.RDI, x86.registers.RSI), ()),
        visibility="public",
    )
    return ModuleOp([func])


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Emit an fft.dft kernel as MLIR.")
    parser.add_argument("output", type=Path)
    parser.add_argument("n", type=int, help="Number of complex points")
    parser.add_argument("precision", choices=["SP", "DP"])
    parser.add_argument("--sign", type=int, choices=[-1, 1], default=-1)
    args = parser.parse_args(argv)
    datatype = Float32Type() if args.precision == "SP" else Float64Type()
    with args.output.open("w") as stream:
        Printer(stream=stream).print_op(dft_module(args.n, datatype, sign=args.sign))
        stream.write("\n")


if __name__ == "__main__":
    main()
