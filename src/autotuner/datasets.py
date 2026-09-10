"""What each dataset measures, and where a sample's files live.

Imported by both the Snakefile, which generates and compiles the code for these
samples, and `autotuner.evaluate`, which times them, so the two cannot drift.
"""

from dataclasses import dataclass
from functools import cache

# Every dataset measures the column-major matmul, so M is the contiguous
# dimension -- the one a kernel vectorizes -- and N is the one it blocks.
KERNEL = "matmul_colmaj"

# Sizes swept by the two square sweeps, one per data type, which set
# M = N = K to each of them.
SQUARE_RANGE = range(1, 65)

# Sizes swept by the two tile sweeps, which set N to each of them and hold
# M = K at the tile in the dataset's name.  Every N up to the tile edge, so the
# curve is sampled at the same density everywhere rather than dense at the small
# end and coarse at the large one.  The steps of 2 and 3 these sweeps were first
# measured at are subsets of this, so every committed measurement is reused.
TTILE_F32_RANGE = range(1, 49)
TTILE_F64_RANGE = range(1, 61)

# Sizes swept by the nano-kernel grid.
#
# A nano-kernel tile's M is the matrix's M, the contiguous dimension, and it is
# that M which spans the vector registers -- one for fsdbcst, two to four for
# nofsdbcst -- so M is the dimension swept out to four f64 vectors, in twos to
# keep the figure sixteen rows tall rather than thirty-two, and N is kept
# inside the tile-N limit every nano-kernel shares.
NANOKERNEL_GRID_M = range(2, 33, 2)
NANOKERNEL_GRID_N = range(1, 8)
NANOKERNEL_GRID_K = range(1, 17)

# The pinned variants use the names that `xsmm-apply-schedule`'s `strategy`
# option takes, so no second variant-to-strategy mapping is needed.  A new one
# is appended rather than interleaved: `dataset_samples` writes the jsonl
# variant-major, so its rows land after the ones already committed and leave
# those untouched.
NANOKERNEL_VARIANTS = (
    "libxsmm-skx-fsdbcst",
    "libxsmm-skx-nofsdbcst",
    "llvm-skx-narrow-fsdbcst",
)

# Which implementations each machine has to compare, per dataset.
VARIANTS = {
    "neon": {
        "ttile": [],
        "f64.small_matrices": [],
        "f32.squares": [],
        "f64.squares": [],
        "f64.nanokernel_grid": [],
    },
    "tower": {
        "ttile": [
            "libxsmm",
            "mkl",
            "aocl",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
        ],
        "f64.small_matrices": [
            "libxsmm",
            "aocl",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
        ],
        # The baselines figure draws the vendor libraries and XTC as well, so
        # the square sweep measures every implementation the tile sweep does.
        # New variants go last: appending leaves the committed measurements
        # where they are.
        "f32.squares": [
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
            "mkl",
            "aocl",
        ],
        # f64 also feeds the squares figure, which prices xDSL's register
        # allocator, so it adds the hand-assigned CompXSMM that figure
        # compares against.
        "f64.squares": [
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "compxsmm_manual",
            "libxtcmm",
            "mkl",
            "aocl",
        ],
        "f64.nanokernel_grid": list(NANOKERNEL_VARIANTS),
    },
    "pinocchio": {
        "ttile": ["libxsmm", "mkl", "aocl"],
        "f64.small_matrices": ["llvm_intrinsics", "libxsmm", "mkl", "aocl"],
        # Neither of ours is generated for this target, so there is no
        # register allocation to price here, and no nano-kernels to pin.
        "f32.squares": [],
        "f64.squares": [],
        "f64.nanokernel_grid": [],
    },
    "rapper": {
        "ttile": [
            "libxsmm",
            "mkl",
            "aocl",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
        ],
        "f64.small_matrices": [
            "libxsmm",
            "aocl",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
        ],
        # The baselines figure draws the vendor libraries and XTC as well, so
        # the square sweep measures every implementation the tile sweep does.
        # New variants go last: appending leaves the committed measurements
        # where they are.
        "f32.squares": [
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
            "mkl",
            "aocl",
        ],
        # f64 also feeds the squares figure, which prices xDSL's register
        # allocator, so it adds the hand-assigned CompXSMM that figure
        # compares against.
        "f64.squares": [
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "compxsmm_manual",
            "libxtcmm",
            "mkl",
            "aocl",
        ],
        "f64.nanokernel_grid": list(NANOKERNEL_VARIANTS),
    },
    "ci": {
        "ttile": [],
        "f64.small_matrices": [],
        "f32.squares": [],
        "f64.squares": [],
        "f64.nanokernel_grid": [],
    },
}


# Path management.  The defaults are Snakemake wildcards, so the Snakefile can
# use these to spell out a rule's inputs and outputs as well as a real path.


# A shape component is an int for a real path, a wildcard string in a rule.
Size = int | str


def machine_base(
    machine: str = "{machine}",
    kernel: str = "{kernel}",
    m: Size = "{m}",
    n: Size = "{n}",
    k: Size = "{k}",
) -> str:
    return f"build/{machine}/{kernel}/{m}x{n}x{k}"


def variant_filename(
    ext: str, variant: str = "{variant}", dtype: str = "{dtype}"
) -> str:
    return f"{variant}.{dtype}.{ext}"


def machine_file(
    ext: str,
    machine: str = "{machine}",
    kernel: str = "{kernel}",
    m: Size = "{m}",
    n: Size = "{n}",
    k: Size = "{k}",
    variant: str = "{variant}",
    dtype: str = "{dtype}",
) -> str:
    base = machine_base(machine=machine, kernel=kernel, m=m, n=n, k=k)
    var = variant_filename(variant=variant, dtype=dtype, ext=ext)
    return f"{base}/{var}"


@dataclass(frozen=True)
class Sample:
    """One shape measured with one implementation."""

    m: int
    n: int
    k: int
    variant: str
    dtype: str
    kernel: str = KERNEL

    def path(self, machine: str, ext: str) -> str:
        """Where this sample's ``ext`` file lives, e.g. ``time.o``."""
        return machine_file(
            ext,
            machine=machine,
            kernel=self.kernel,
            m=self.m,
            n=self.n,
            k=self.k,
            variant=self.variant,
            dtype=self.dtype,
        )


@cache
def nanokernel_grid_shapes(variant: str) -> tuple[tuple[int, int, int], ...]:
    """The (M, N, K) the nano-kernel grid measures ``variant`` at.

    Only the M-by-N tiles the pinned nano-kernel actually supports: every point
    of this figure is meant to be one nano-kernel invocation, so a shape the
    kernel could reach only by looping smaller tiles is not its to draw.  The
    xdsl imports are deferred because the Snakefile imports this module for its
    path helpers alone.
    """
    from xdsl.dialects import builtin

    from autotuner.nano_kernel import SupportedTile
    from autotuner.skx_nano_kernel import AVX512Info, get_skx_nano_kernel

    nano_kernel = get_skx_nano_kernel(variant)
    supported = nano_kernel.supported_tile_sizes(builtin.f64, AVX512Info())
    return tuple(
        (m, n, k)
        for m in NANOKERNEL_GRID_M
        for n in NANOKERNEL_GRID_N
        if SupportedTile(m, n) in supported
        for k in NANOKERNEL_GRID_K
    )


def dataset_samples(machine: str) -> dict[str, list[Sample]]:
    """The samples each dataset measures, in the order its jsonl records them.

    Whether variant or shape varies fastest is not a style choice: it is the
    order each committed dataset was first written in, so keeping it means
    re-deriving a file from unchanged measurements leaves it untouched.
    """
    variants = VARIANTS[machine]

    def by_variant(dtype, shapes, key):
        return [
            Sample(m, n, k, variant, dtype)
            for variant in variants[key]
            for m, n, k in shapes
        ]

    def by_shape(dtype, shapes, key):
        return [
            Sample(m, n, k, variant, dtype)
            for m, n, k in shapes
            for variant in variants[key]
        ]

    # The tile sweeps vary N, the blocked dimension, and hold the contiguous M
    # fixed: M is what a column-major kernel vectorizes, so it is the register
    # block size rather than the trip count those figures are about.  The
    # square sweeps vary all three dimensions together.
    return {
        "f32.ttile": by_variant(
            "f32", [(128, n, 128) for n in TTILE_F32_RANGE], "ttile"
        ),
        "f64.ttile": by_variant("f64", [(64, n, 64) for n in TTILE_F64_RANGE], "ttile"),
        "f64.small_matrices": by_shape(
            "f64",
            [(m, n, 64) for n in range(1, 17) for m in range(1, 17)],
            "f64.small_matrices",
        ),
        "f32.squares": by_shape(
            "f32", [(s, s, s) for s in SQUARE_RANGE], "f32.squares"
        ),
        "f64.squares": by_shape(
            "f64", [(s, s, s) for s in SQUARE_RANGE], "f64.squares"
        ),
        # Not `by_shape`: which shapes are measured depends on the variant, so
        # the variants cannot share one shape list.
        "f64.nanokernel_grid": [
            Sample(m, n, k, variant, "f64")
            for variant in variants["f64.nanokernel_grid"]
            for m, n, k in nanokernel_grid_shapes(variant)
        ],
    }
