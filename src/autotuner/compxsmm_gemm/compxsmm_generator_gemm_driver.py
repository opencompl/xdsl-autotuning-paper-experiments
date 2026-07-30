from pathlib import Path
from autotuner.compxsmm_gemm.generator_gemm import compxsmm_generator_gemm_directasm
from autotuner.libxsmm_gemm.libxsmm_macros import gemm_flags
from autotuner.libxsmm_gemm.libxsmm_main import DescDatatype, GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="CompXSMM reimplementation of libxsmm_gemm_generator (dense only)."
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
    parser.add_argument("beta", type=int, choices=[0, 1], help="Beta (0 or 1)")
    parser.add_argument(
        "align_a", type=int, choices=[0, 1], help="0: unaligned A, otherwise aligned"
    )
    parser.add_argument(
        "align_c", type=int, choices=[0, 1], help="0: unaligned C, otherwise aligned"
    )
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

    # parse Datatype
    match args.precision:
        case "SP" | "DP":
            dt = Datatype.F32 if args.precision == "SP" else Datatype.F64
            desc_datatype = DescDatatype(dt, dt, dt, dt)
        case "I16":
            desc_datatype = DescDatatype(
                Datatype.I16, Datatype.I16, Datatype.I32, Datatype.I32
            )
        case _:
            parser.error(f"Unsupported precision: {args.precision}")

    # parse Arch
    try:
        arch = ARCH_BY_CODE[args.arch]
    except KeyError:
        parser.error(f"Unsupported architecture: {args.arch}")

    flags = gemm_flags("N", "N")

    if args.align_a:
        flags |= GEMMFlag.ALIGN_A

    if args.align_c:
        flags |= GEMMFlag.ALIGN_C

    if args.beta == 0:
        flags |= GEMMFlag.BETA_0

    descriptor = GEMMDescriptor(
        m=args.m,
        n=args.n,
        k=args.k,
        lda=args.lda,
        ldb=args.ldb,
        ldc=args.ldc,
        datatype=desc_datatype,
        flags=flags,
        prefetch=args.prefetch,
    )

    assert args.density == "dense", f"Only dense supported, got {args.density}"
    assert arch == Arch.LIBXSMM_X86_AVX512_SKX, f"Only `skx` arch supported, got {arch}"

    compxsmm_generator_gemm_directasm(
        args.filename, args.routine_name, descriptor, arch
    )
