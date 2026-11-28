from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from autotuner.libxsmm_gemm.generator_mateltwise import (
    libxsmm_generator_mateltwise_directasm,
)
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE
from autotuner.libxsmm_gemm.libxsmm_main import MeltwDescriptor
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(
        description="Literal SKX FP32 EXP port: emit allocated x86 IR."
    )
    parser.add_argument("filename", type=Path)
    parser.add_argument("routine_name")
    parser.add_argument("length", type=int)
    parser.add_argument("--degree", type=int, choices=(3, 5), required=True)
    parser.add_argument("--arch", choices=("skx",), default="skx")
    args = parser.parse_args(argv)
    if args.length <= 0 or args.length % 16:
        parser.error("length must be positive and divisible by 16")
    desc = MeltwDescriptor(args.length, 1, args.length, args.length, Datatype.F32)
    libxsmm_generator_mateltwise_directasm(
        args.filename, args.routine_name, desc, ARCH_BY_CODE[args.arch], args.degree
    )


if __name__ == "__main__":
    main()
