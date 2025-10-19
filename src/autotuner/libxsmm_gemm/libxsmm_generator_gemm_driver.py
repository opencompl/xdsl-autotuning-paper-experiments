from pathlib import Path
from autotuner.libxsmm_gemm.generator_gemm import libxsmm_generator_gemm_directasm
from autotuner.libxsmm_gemm.libxsmm_main import DescDataType, GEMMDescriptor, GemmFlag
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_typedefs import DataType


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Python reimplementation of libxsmm_gemm_generator (dense only)."
    )
    parser.add_argument(
        "density",
        choices=["dense", "dense_asm", "sparse", "sparse_csr"],
        help="Matrix multiplication density; only ''dense_asm' (dense*dense) is supported.",
    )
    parser.add_argument("filename", type=Path, help="Filename to append")
    parser.add_argument("routine_name", help="Routine name")
    parser.add_argument("m", type=int, help="M dimension")
    parser.add_argument("n", type=int, help="N dimension")
    parser.add_argument("k", type=int, help="K dimension")
    parser.add_argument("lda", type=int, help="Leading dimension A (LDA)")
    parser.add_argument("ldb", type=int, help="Leading dimension B (LDB)")
    parser.add_argument("ldc", type=int, help="Leading dimension C (LDC)")
    parser.add_argument("alpha", type=int, help="Alpha (must be -1 or 1)")
    parser.add_argument("beta", type=int, help="Beta (0 or 1)")
    parser.add_argument("align_a", type=int, help="0: unaligned A, otherwise aligned")
    parser.add_argument("align_c", type=int, help="0: unaligned C, otherwise aligned")
    parser.add_argument("arch", choices=ARCH_BY_CODE, help="Target architecture")
    parser.add_argument("prefetch", choices=["nopf", "AL2"], help="Prefetch strategy")
    parser.add_argument(
        "precision",
        choices=["SP", "DP"],
        help="Precision: SP or DP",
    )

    args = parser.parse_args()

    if args.density not in ["dense", "dense_asm"]:
        if args.density in ["sparse", "sparse_csr"]:
            parser.error(
                "Sparse mode (sparse/sparse_csr) is not implemented in this tool."
            )
        else:
            parser.error(f"Unrecognized density argument: {args.density}")

    # parse DataType
    match args.precision:
        case "F32" | "F64":
            dt = DataType.F32 if args.precision == "SP" else DataType.F64
            desc_datatype = DescDataType(dt, dt, dt, dt)
        case "I16":
            desc_datatype = DescDataType(
                DataType.I16, DataType.I16, DataType.I32, DataType.I32
            )
        case _:
            parser.error(f"Unsupported precision: {args.precision}")

    # parse Arch
    try:
        arch = ARCH_BY_CODE[args.arch]
    except KeyError:
        parser.error(f"Unsupported architecture: {args.arch}")

    descriptor = GEMMDescriptor(
        m=args.m,
        n=args.n,
        k=args.k,
        lda=args.lda,
        ldb=args.ldb,
        ldc=args.ldc,
        datatype=desc_datatype,
        flags=GemmFlag(0),
    )

    assert args.density == "dense_asm", f"Only dense_asm supported, got {args.density}"
    assert arch == Arch.LIBXSMM_X86_AVX512_SKX, f"Only `skx` arch supported, got {arch}"

    libxsmm_generator_gemm_directasm(args.filename, args.routine_name, descriptor, arch)
