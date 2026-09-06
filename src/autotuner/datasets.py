"""What each dataset measures, and where a sample's files live.

Imported by both the Snakefile, which generates and compiles the code for these
samples, and `autotuner.evaluate`, which times them, so the two cannot drift.
"""

from dataclasses import dataclass

# Every dataset measures the row-major matmul.
KERNEL = "matmul_rowmaj"

# Sizes swept by the square dataset, which sets M = N = K to each of them.
SQUARE_RANGE = range(1, 65)

# Which implementations each machine has to compare, per dataset.
VARIANTS = {
    "neon": {
        "ttile": ["naive_c"],
        "f64.small_matrices": [],
        "f64.squares": [],
    },
    "tower": {
        "ttile": [
            "naive_c",
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
        "f64.squares": [
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "compxsmm_manual",
        ],
    },
    "pinocchio": {
        "ttile": ["naive_c", "libxsmm", "mkl", "aocl"],
        "f64.small_matrices": ["llvm_intrinsics", "libxsmm", "mkl", "aocl"],
        # Neither of ours is generated for this target, so there is no
        # register allocation to price here.
        "f64.squares": [],
    },
    "rapper": {
        "ttile": [
            "naive_c",
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
        "f64.squares": [
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "compxsmm_manual",
        ],
    },
    "ci": {
        "ttile": ["naive_c"],
        "f64.small_matrices": [],
        "f64.squares": [],
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

    return {
        "f32.ttile": by_variant(
            "f32", [(m, 128, 128) for m in range(8, 50, 2)], "ttile"
        ),
        "f64.ttile": by_variant("f64", [(m, 64, 64) for m in range(9, 63, 3)], "ttile"),
        "f64.small_matrices": by_shape(
            "f64",
            [(m, n, 64) for m in range(1, 17) for n in range(1, 17)],
            "f64.small_matrices",
        ),
        "f64.squares": by_shape(
            "f64", [(s, s, s) for s in SQUARE_RANGE], "f64.squares"
        ),
    }
