"""``libxtcmm-gemm`` CLI: same arguments as ``compxsmm-gemm`` / ``libxsmm-gemm``,
but applies the corresponding LIBXSMM schedule to a matmul through the XTC
scheduler and writes the resulting XTC MLIR to ``filename``.

Reuses LIBXSMM's argument parsing / descriptor construction and its real
scheduling-decision functions (via :func:`compute_plan`); see the package
docstrings for the dimension mapping and the register-resident scope.
"""

import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from autotuner.libxsmm_gemm.libxsmm_cpuid import ARCH_BY_CODE, Arch
from autotuner.libxsmm_gemm.libxsmm_macros import gemm_flags
from autotuner.libxsmm_gemm.libxsmm_main import DescDatatype, GEMMDescriptor, GEMMFlag
from autotuner.libxsmm_gemm.libxsmm_typedefs import Datatype
from autotuner.libxtcmm_gemm.plan import compute_plan
from autotuner.libxtcmm_gemm.schedule import emit_mlir


def main(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(
        description=(
            "Apply the LIBXSMM GEMM schedule to a matmul via the XTC scheduler "
            "and emit the resulting XTC MLIR (dense only)."
        )
    )
    parser.add_argument(
        "density",
        choices=["dense", "dense_asm", "sparse", "sparse_csr"],
        help="Matrix multiplication density; only 'dense'/'dense_asm' is supported.",
    )
    parser.add_argument("filename", type=Path, help="Output file for the XTC MLIR")
    parser.add_argument("routine_name", help="Routine name")
    parser.add_argument("m", type=int, help="M dimension")
    parser.add_argument("n", type=int, help="N dimension")
    parser.add_argument("k", type=int, help="K dimension")
    parser.add_argument("lda", type=int, help="Leading dimension A (LDA)")
    parser.add_argument("ldb", type=int, help="Leading dimension B (LDB)")
    parser.add_argument("ldc", type=int, help="Leading dimension C (LDC)")
    parser.add_argument(
        "alpha", type=int, choices=[1], help="Alpha (only 1 is supported)"
    )
    parser.add_argument(
        "beta", type=int, choices=[1], help="Beta (only 1 is supported)"
    )
    parser.add_argument(
        "align_a", type=int, choices=[0, 1], help="0: unaligned A, otherwise aligned"
    )
    parser.add_argument(
        "align_c", type=int, choices=[0, 1], help="0: unaligned C, otherwise aligned"
    )
    parser.add_argument("arch", choices=ARCH_BY_CODE, help="Target architecture")
    parser.add_argument("prefetch", choices=["nopf", "AL2"], help="Prefetch strategy")
    parser.add_argument("precision", choices=["SP", "DP"], help="Precision: SP or DP")
    parser.add_argument(
        "--disable-regalloc",
        dest="disable_regalloc",
        action="store_true",
        help=(
            "Accepted for CLI compatibility with compxsmm/libxsmm; ignored "
            "(XTC does its own register allocation)."
        ),
    )
    parser.add_argument(
        "--mask-tail",
        dest="mask_tail",
        action="store_true",
        help=(
            "Vectorize a sub-vector-length tail as a masked full-width vector "
            "instead of the default narrow unmasked vector."
        ),
    )

    args = parser.parse_args(argv)

    # `precision` and `arch` are already constrained by argparse `choices`.
    dt = Datatype.F32 if args.precision == "SP" else Datatype.F64
    desc_datatype = DescDatatype(dt, dt, dt, dt)
    arch = ARCH_BY_CODE[args.arch]

    flags = gemm_flags("N", "N")
    if args.align_a:
        flags |= GEMMFlag.ALIGN_A
    if args.align_c:
        flags |= GEMMFlag.ALIGN_C
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

    assert args.density in ("dense", "dense_asm"), (
        f"Only dense supported, got {args.density}"
    )
    assert arch == Arch.LIBXSMM_X86_AVX512_SKX, f"Only `skx` arch supported, got {arch}"

    if args.disable_regalloc:
        print(
            "note: --disable-regalloc is ignored; XTC does its own register allocation",
            file=sys.stderr,
        )

    plan = compute_plan(descriptor, arch)
    emit_mlir(
        plan, descriptor, args.filename, args.routine_name, mask_tail=args.mask_tail
    )


if __name__ == "__main__":
    main()
