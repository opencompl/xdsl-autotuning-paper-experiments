"""Generate and compile every kernel the datasets need, in one process.

    uv run build-dataset --machine rapper
    uv run build-dataset --machine rapper f64.nanokernel_grid

Snakemake ran this as one job per file, which cost more than the work it
scheduled: the full rapper sweep is ~37k jobs, and Snakemake dispatches around
120 of them a second, so a 128-core machine sat about half idle.  Worse, most
of a job *was* startup -- `compxsmm-gemm` spends 0.46s importing and 0.005s
generating, `xdsl-opt` 0.64s importing and 0.09s compiling.

So this builds the same files with the same commands, but from a single
process: one task per shape, a pool as wide as the machine, and the code
generators called in-process, so a worker imports them once instead of once
per shape.  The plan is decided here and shipped to the workers as data --
see `Step` -- which keeps the recipe and the up-to-date check in one place.

Staleness is content-keyed, not mtime-keyed.  Every artifact records the hash
of the commands that made it plus the sources those commands read, in
`build/<machine>/.build-manifest.json`.  A timing binary that comes out
byte-identical is left alone rather than rewritten, because `autotuner.evaluate`
treats a binary newer than its `.txt` as needing a fresh measurement, and
measuring is the slow, strictly serial half of `make dataset`.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from autotuner.datasets import NANOKERNEL_VARIANTS, Sample, dataset_samples
from autotuner.machines import MACHINES

# The C spelling of each dtype, and how the XSMM generators name it.
C_TYPE = {"f32": "float", "f64": "double"}
XSMM_PRECISION = {"f32": "SP", "f64": "DP"}

# Which sources decide that generated code is out of date.  Directories are
# expanded to the Python files under them; `pyproject.toml` pins xdsl itself.
GENERATOR_SOURCES: Mapping[str, tuple[str, ...]] = {
    "libxsmm": ("src/autotuner/libxsmm_gemm",),
    "compxsmm": ("src/autotuner/libxsmm_gemm", "src/autotuner/compxsmm_gemm"),
    "libxtcmm": ("src/autotuner/libxsmm_gemm", "src/autotuner/libxtcmm_gemm"),
    # Every xdsl-opt pipeline runs project passes out of these two packages.
    "xdsl": ("src/autotuner/passes", "src/autotuner/dialects", "pyproject.toml"),
}

# The generator each "generate:*" step calls, imported once per worker.
GENERATOR_MODULES = {
    "generate:libxsmm": "autotuner.libxsmm_gemm.libxsmm_generator_gemm_driver",
    "generate:compxsmm": "autotuner.compxsmm_gemm.compxsmm_generator_gemm_driver",
    "generate:libxtcmm": "autotuner.libxtcmm_gemm.libxtcmm_generator_gemm_driver",
}

# These all lower the same schedule-neutral CompXSMM IR. The ordinary variant
# lets the SKX heuristic choose a nano-kernel; the others pin one by name.
COMPXSMM_SHARED_VARIANTS = frozenset(("compxsmm", *NANOKERNEL_VARIANTS))
COMPXSMM_VARIANTS = COMPXSMM_SHARED_VARIANTS | {"compxsmm_manual"}

MANIFEST = ".build-manifest.json"

console = Console()


def scratch_name(path: Path) -> Path:
    """Where a rebuild lands before it is compared with what is already there."""
    return path.with_name(path.name + ".tmp")


class BuildFailed(RuntimeError):
    """One or more kernels could not be generated or compiled."""


class StepFailed(RuntimeError):
    """A single command failed, with the command and its stderr attached."""


# --- the plan, as data ------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One command in a recipe.

    `kind` picks the executor: "run" shells out, the "generate:*" kinds call a
    generator's `main` in this process, "xdsl-opt" drives xDSLOptMain, and
    "remove" clears an output the XSMM generators would otherwise append to.
    Everything is plain strings so a plan pickles cheaply into a worker -- and
    so the same tuple can be hashed into the artifact's key.
    """

    kind: str
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Artifact:
    """A file, the steps that produce it, and what makes it stale."""

    path: Path
    steps: tuple[Step, ...]
    # Digests of the sources the steps read, plus the keys of upstream
    # artifacts, so a changed generator invalidates everything downstream.
    inputs: tuple[str, ...] = ()
    # Artifacts that must succeed first; identified by path.
    needs: tuple[Path, ...] = ()
    # Keep the existing file, and its mtime, when the rebuild matches it byte
    # for byte.  Set for timing binaries: an untouched binary keeps its
    # measurement valid.
    preserve: bool = False

    @property
    def target(self) -> Path:
        """Where the last step writes: a temp file when the old one matters."""
        return scratch_name(self.path) if self.preserve else self.path

    @property
    def key(self) -> str:
        """What this artifact is the product of."""
        return sha256_of(
            json.dumps(
                [[s.kind, list(s.args), list(s.env)] for s in self.steps]
                + [list(self.inputs)],
                sort_keys=True,
            )
        )


@dataclass(frozen=True)
class ShapeJob:
    """Everything one worker task builds: one shape, every variant of it."""

    label: str
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True)
class ShapeResult:
    built: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    failures: tuple[tuple[str, str], ...] = ()


# --- what the commands need to know ----------------------------------------


@dataclass(frozen=True)
class Toolchain:
    """The machine-wide settings every recipe reads."""

    machine: str
    isa: str
    triple: str
    march: str
    mtune: str
    freq: float
    libs: tuple[str, ...]
    linker_flag: str
    libxsmm_arch: str | None
    cc: str
    cc_asm: str
    libxsmm_generator: str
    mkl: tuple[tuple[str, ...], tuple[str, ...]]
    aocl: tuple[tuple[str, ...], tuple[str, ...]]
    pipelines: Mapping[str, str]
    xtc_mlir_bin: str
    xtc_llvm_bin: str
    root: Path = Path("build")
    digests: Mapping[str, str] = field(default_factory=dict)

    def shape_dir(self, kernel: str, m: int, n: int, k: int) -> Path:
        return self.root / self.machine / kernel / f"{m}x{n}x{k}"

    @property
    def use_papi(self) -> tuple[str, ...]:
        return ("-DUSE_PAPI",) if "papi" in self.libs else ()


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def digest(paths: Iterable[Path]) -> str:
    """A digest of these files' names and contents."""
    running = hashlib.sha256()
    for path in sorted(paths):
        running.update(str(path).encode())
        running.update(b"\0")
        running.update(path.read_bytes())
        running.update(b"\0")
    return running.hexdigest()


def source_digests(repo: Path = Path(".")) -> dict[str, str]:
    """One digest per generator, over the sources it is compiled from."""
    digests = {}
    for name, roots in GENERATOR_SOURCES.items():
        files: list[Path] = []
        for spelling in roots:
            where = repo / spelling
            files.extend(sorted(where.rglob("*.py")) if where.is_dir() else [where])
        digests[name] = digest(f for f in files if f.exists())
    return digests


def pkg_config(module: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`module`'s cflags and libs, or empty tuples when it is not installed."""
    if not shutil.which("pkg-config"):
        return ((), ())
    if subprocess.run(["pkg-config", "--exists", module], check=False).returncode:
        return ((), ())

    def ask(what: str) -> tuple[str, ...]:
        out = subprocess.run(
            ["pkg-config", what, module], capture_output=True, text=True, check=True
        )
        return tuple(shlex.split(out.stdout.strip()))

    return (ask("--cflags"), ask("--libs"))


def tool_path(name: str) -> str:
    """Resolve a tool to its full path, so a toolchain bump rebuilds."""
    found = shutil.which(name)
    if found is None:
        raise BuildFailed(f"{name} is not on PATH")
    return found


def toolchain(
    machine: str,
    *,
    root: Path = Path("build"),
    config: Path = Path("default.yaml"),
    repo: Path = Path("."),
) -> Toolchain:
    """Read the machine and the config the Snakefile used to read."""
    settings = yaml.safe_load(config.read_text())
    spec = MACHINES[machine]

    def per_isa(setting: str) -> str:
        by_isa = settings.get(setting, {})
        return ",".join(by_isa[spec.isa]) if spec.isa in by_isa else ""

    nanokernel_pipeline = per_isa("compxsmm-nanokernel-passes")

    # Keyed by variant, not by generator: CompXSMM variants share generated IR
    # and differ in the pass pipeline that lowers it.
    pipelines = {
        "xdsl_libxsmm": ",".join(settings["libxsmm-gemm-passes"]),
        "compxsmm": per_isa("compxsmm-gemm-passes"),
        "compxsmm_manual": per_isa("compxsmm-manual-gemm-passes"),
        **{
            variant: nanokernel_pipeline.replace("{nanokernel}", variant)
            for variant in NANOKERNEL_VARIANTS
        },
        "libxtcmm": per_isa("libxtcmm-gemm-passes"),
    }

    return Toolchain(
        machine=machine,
        isa=spec.isa,
        triple=spec.target_triple,
        march=spec.march,
        mtune=spec.mtune,
        freq=spec.freq,
        libs=tuple(spec.libs),
        linker_flag=spec.linker_flag,
        libxsmm_arch=spec.libxsmm_arch,
        cc=tool_path(settings["cc"]),
        cc_asm=tool_path(settings.get("cc_asm", settings["cc"])),
        libxsmm_generator=shutil.which("libxsmm_gemm_generator") or "",
        mkl=pkg_config("mkl-dynamic-ilp64-seq"),
        aocl=pkg_config("blis"),
        pipelines=pipelines,
        xtc_mlir_bin=os.environ.get("XTC_MLIR_PREFIX", ""),
        xtc_llvm_bin=os.environ.get("XTC_LLVM_PREFIX", ""),
        root=root,
        digests=source_digests(repo),
    )


# --- recipes: one per variant, mirroring the Snakefile ----------------------


def kernel_source(kernel: str, name: str) -> Path:
    return Path("kernels") / kernel / name


# What every variant's `matmul` is aligned to, as a power of two.  The entry
# point has to land on the same boundary for all of them or the comparison
# measures code layout: at M = N = K = 14 the four variants emit an identical
# vector instruction stream, yet LIBXSMM's -- inline asm inside a C function,
# which clang aligns to 16 -- came out 10% slower than the same body emitted as
# an aligned bare function, and padding CompXSMM's entry by 40 bytes moved it
# 5% the other way.  A cache line settles it for all of them.
ASM_ALIGNMENT = 6


def asm_artifact(tool: Toolchain, sample: Sample) -> Artifact:
    """This sample's assembly, with its entry point aligned like every other's.

    The alignment is a step of its own rather than a flag on each recipe:
    the variants come out of clang, of xDSL and of llc, which agree on neither
    the directive nor whether they emit one at all.
    """
    recipe = asm_recipe(tool, sample)
    align = Step("align", (str(recipe.path), str(ASM_ALIGNMENT)))
    return replace(recipe, steps=recipe.steps + (align,))


def asm_recipe(tool: Toolchain, sample: Sample) -> Artifact:
    """How this sample's assembly is produced."""
    m, n, k, dtype = sample.m, sample.n, sample.k, sample.dtype
    here = tool.shape_dir(sample.kernel, m, n, k)
    out = here / f"{sample.variant}.{dtype}.S"
    ctype = C_TYPE[dtype]
    target = ("-target", tool.triple, f"-march={tool.march}")

    match sample.variant:
        case "naive_c":
            source = kernel_source(sample.kernel, "naive_c.c")
            steps = (
                Step(
                    "run",
                    (
                        tool.cc_asm,
                        "-O3",
                        f"-DCROWS={m}",
                        f"-DCCOLS={n}",
                        f"-DINNER={k}",
                        f"-DDTYPE={ctype}",
                        "-S",
                        *target,
                        "-o",
                        str(out),
                        str(source),
                    ),
                ),
            )
            return Artifact(out, steps, (digest([source]),))

        case "llvm_intrinsics":
            source = kernel_source(sample.kernel, "llvm_intrinsics.c")
            steps = (
                Step(
                    "run",
                    (
                        tool.cc_asm,
                        "-O3",
                        "-c",
                        str(source),
                        f"-DM={m}",
                        f"-DN={n}",
                        f"-DK={k}",
                        f"-DDTYPE={ctype}",
                        "-S",
                        "-fenable-matrix",
                        *target,
                        f"-mtune={tool.mtune}",
                        "-o",
                        str(out),
                        "-ffp-contract=fast",
                        "-ffast-math",
                        "-mprefer-vector-width=512",
                    ),
                ),
            )
            return Artifact(out, steps, (digest([source]),))

        case "mkl" | "aocl":
            vendor = sample.variant
            source = kernel_source(sample.kernel, f"{vendor}.c")
            cflags, _ = tool.mkl if vendor == "mkl" else tool.aocl
            prefix = vendor.upper()
            is_float = f"-D{prefix}_DTYPE_IS_FLOAT=1"
            is_double = f"-D{prefix}_DTYPE_IS_DOUBLE=1"
            steps = (
                Step(
                    "run",
                    (
                        tool.cc,
                        "-O3",
                        str(source),
                        *cflags,
                        f"-D{prefix}_M={m}",
                        f"-D{prefix}_N={n}",
                        f"-D{prefix}_K={k}",
                        is_float if dtype == "f32" else is_double,
                        "-S",
                        *target,
                        "-o",
                        str(out),
                    ),
                ),
            )
            return Artifact(out, steps, (digest([source]),))

        case "libxsmm":
            # A = M*K, B = K*N, C = M*N column-major, so the leading dimensions
            # are M, K and M and libxsmm's own ABI is the one we benchmark: it
            # emits `void matmul(A, B, C)` and needs no wrapper.
            generated = here / f"libxsmm.{dtype}.c"
            steps = (
                Step("remove", (str(generated),)),
                Step(
                    "run",
                    (
                        tool.libxsmm_generator,
                        "dense",
                        str(generated),
                        "matmul",
                        str(m),
                        str(n),
                        str(k),
                        str(m),
                        str(k),
                        str(m),
                        "1",
                        "1",
                        "1",
                        "1",
                        str(tool.libxsmm_arch),
                        "nopf",
                        XSMM_PRECISION[dtype],
                    ),
                ),
                Step(
                    "run",
                    (
                        tool.cc,
                        "-O3",
                        "-DNDEBUG",
                        str(generated),
                        "-S",
                        *target,
                        "-o",
                        str(out),
                    ),
                ),
                # The C is only a carrier for one asm block; keep the kernel
                # and drop the frame clang had to build around it, so this
                # variant is the same shape of function as the xDSL ones.
                Step("bare", (str(out),)),
            )
            return Artifact(out, steps)

        case variant if variant == "xdsl_libxsmm" or variant in COMPXSMM_VARIANTS:
            generator = "libxsmm" if variant == "xdsl_libxsmm" else "compxsmm"
            mlir_variant = (
                "compxsmm" if variant in COMPXSMM_SHARED_VARIANTS else variant
            )
            mlir = here / f"{mlir_variant}.{dtype}.{generator}.mlir"
            extra = (
                ("--disable-regalloc",) if variant in COMPXSMM_SHARED_VARIANTS else ()
            )
            steps = (
                Step("remove", (str(mlir),)),
                Step(
                    f"generate:{generator}",
                    (
                        "dense",
                        str(mlir),
                        "matmul",
                        str(m),
                        str(n),
                        str(k),
                        str(m),
                        str(k),
                        str(m),
                        "1",
                        "1",
                        "1",
                        "1",
                        str(tool.libxsmm_arch),
                        "nopf",
                        XSMM_PRECISION[dtype],
                        *extra,
                    ),
                ),
                Step(
                    "xdsl-opt",
                    (
                        str(mlir),
                        "-p",
                        tool.pipelines[variant],
                        "-t",
                        "x86-asm",
                        "-o",
                        str(out),
                    ),
                ),
            )
            return Artifact(out, steps, (tool.digests[generator], tool.digests["xdsl"]))

        case "libxtcmm":
            mlir = here / f"libxtcmm.{dtype}.xtcmm.mlir"
            pipeline = f"builtin.module({tool.pipelines['libxtcmm']})"
            mlir_bin = Path(tool.xtc_mlir_bin) / "bin"
            llvm_bin = Path(tool.xtc_llvm_bin) / "bin"
            # Replay XTC's own lowering, the way the Snakefile did.
            steps = (
                Step("remove", (str(mlir),)),
                Step(
                    "generate:libxtcmm",
                    (
                        "dense",
                        str(mlir),
                        "matmul",
                        str(m),
                        str(n),
                        str(k),
                        str(m),
                        str(k),
                        str(m),
                        "1",
                        "1",
                        "1",
                        "1",
                        str(tool.libxsmm_arch),
                        "nopf",
                        XSMM_PRECISION[dtype],
                    ),
                ),
                Step(
                    "run",
                    (
                        str(mlir_bin / "mlir-opt"),
                        str(mlir),
                        f"-pass-pipeline={pipeline}",
                        "-o",
                        f"{out}.llvm.mlir",
                    ),
                ),
                Step(
                    "run",
                    (
                        str(mlir_bin / "mlir-translate"),
                        "--mlir-to-llvmir",
                        f"{out}.llvm.mlir",
                        "-o",
                        f"{out}.ll",
                    ),
                ),
                Step(
                    "run",
                    (
                        str(llvm_bin / "opt"),
                        "-O2",
                        "--fp-contract=fast",
                        f"-mtriple={tool.triple}",
                        f"-mcpu={tool.march}",
                        f"{out}.ll",
                        "-o",
                        f"{out}.bc",
                    ),
                ),
                Step(
                    "run",
                    (
                        str(llvm_bin / "llc"),
                        "-O2",
                        "-filetype=asm",
                        f"-mtriple={tool.triple}",
                        f"-mcpu={tool.march}",
                        f"{out}.bc",
                        "-o",
                        str(out),
                    ),
                ),
            )
            return Artifact(
                out, steps, (tool.digests["libxtcmm"], tool.digests["xdsl"])
            )

    raise BuildFailed(f"no recipe for variant {sample.variant!r}")


def driver_artifact(
    tool: Toolchain, kernel: str, m: int, n: int, k: int, dtype: str, driver: str
) -> Artifact:
    """The harness object, compiled once per shape rather than per variant.

    `time.c` reads the shape and the dtype off the command line and nothing
    else, so the seven variants of one shape were compiling the same object
    seven times under Snakemake -- 9,450 redundant compiles for the rapper
    sweep.
    """
    source = kernel_source(kernel, f"{driver}.c")
    out = tool.shape_dir(kernel, m, n, k) / f"{driver}.{dtype}.o"
    steps = (
        Step(
            "run",
            (
                tool.cc,
                f"-DCROWS={m}",
                f"-DCCOLS={n}",
                f"-DINNER={k}",
                f"-DDTYPE={C_TYPE[dtype]}",
                f"-DFREQ={tool.freq}",
                *tool.use_papi,
                "-target",
                tool.triple,
                f"-march={tool.march}",
                "-c",
                str(source),
                "-o",
                str(out),
            ),
        ),
    )
    return Artifact(out, steps, (digest([source, *headers()]),))


def headers() -> list[Path]:
    """The headers every driver includes."""
    return sorted(Path("headers").glob("*.h"))


def binary_artifact(
    tool: Toolchain, sample: Sample, asm: Artifact, obj: Artifact, driver: str
) -> Artifact:
    """The linked benchmark: the shared driver object plus this variant's asm."""
    out = tool.shape_dir(sample.kernel, sample.m, sample.n, sample.k) / (
        f"{sample.variant}.{sample.dtype}.{driver}.o"
    )
    vendor_libs: tuple[str, ...] = ()
    if sample.variant == "mkl":
        vendor_libs = tool.mkl[1]
    elif sample.variant == "aocl":
        vendor_libs = tool.aocl[1]

    link = (
        tool.cc,
        "-target",
        tool.triple,
        f"-march={tool.march}",
        "-o",
        str(scratch_name(out)),
        str(obj.path),
        str(asm.path),
        *(f"-l{lib}" for lib in tool.libs),
        *vendor_libs,
        *((tool.linker_flag,) if tool.linker_flag else ()),
    )
    return Artifact(
        out,
        (Step("run", link),),
        inputs=(asm.key, obj.key),
        needs=(asm.path, obj.path),
        preserve=True,
    )


def plan_shape(
    tool: Toolchain, samples: Sequence[Sample], driver: str
) -> tuple[Artifact, ...]:
    """Every artifact for one shape, in the order a worker must build them."""
    first = samples[0]
    obj = driver_artifact(
        tool, first.kernel, first.m, first.n, first.k, first.dtype, driver
    )
    artifacts: list[Artifact] = [obj]
    for sample in samples:
        asm = asm_artifact(tool, sample)
        artifacts.append(asm)
        artifacts.append(binary_artifact(tool, sample, asm, obj, driver))
    return tuple(artifacts)


# --- running the plan -------------------------------------------------------


def run_command(args: Sequence[str], env: Mapping[str, str] | None = None) -> None:
    done = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )
    if done.returncode:
        raise StepFailed(f"{shlex.join(args)}\n{done.stderr.strip()}")


# The registers a function must give back to its caller unchanged, with the
# narrower names that alias them, so that scanning a kernel body for them is
# enough to know what it has to save.
CALLEE_SAVED = {
    "rbx": ("rbx", "ebx", "bx", "bl"),
    "rbp": ("rbp", "ebp", "bp", "bpl"),
    "r12": ("r12", "r12d", "r12w", "r12b"),
    "r13": ("r13", "r13d", "r13w", "r13b"),
    "r14": ("r14", "r14d", "r14w", "r14b"),
    "r15": ("r15", "r15d", "r15w", "r15b"),
}

# Where the System V ABI leaves `matmul(A, B, C)`, which is where LIBXSMM's
# kernel wants them: the reload the wrapper does is a round trip to nowhere.
ARGUMENT_REGISTERS = ("%rdi", "%rsi", "%rdx")

# The frame LIBXSMM's own kernel opens -- 192 bytes of prefetch/scratch/eltwise
# slots aligned to 64 -- and the teardown that matches it.  1.17 emitted
# neither; the revision pinned in the flake emits both.  It is the generator's
# frame, not clang's, so it is code under test and stays.
OWN_FRAME_OPEN = re.compile(r"\A\tpushq\t%rbp\n\tmovq\t%rsp, %rbp\n")
OWN_FRAME_CLOSE = re.compile(r"\tmovq\t%rbp, %rsp\n\tpopq\t%rbp\Z")


def strip_asm_wrapper(path: Path) -> None:
    """Re-emit a `matmul` that is one inline-asm block as a bare function.

    LIBXSMM's generator hands us C whose entire body is a single `__asm__`
    block with `"m"` operands and a clobber list naming every general-purpose
    register there is.  Clang has to believe it: it opens a frame, spills the
    three pointers to the stack for the asm to load straight back, and saves
    five callee-saved registers the kernel never touches.  Called back to back
    the way `time.c` calls it, that wrapper is not free -- each restore queues
    behind the body's masked AVX-512 stores -- and at M = N = K = 14 it cost 52
    of the 493 cycles measured, which was the whole of the gap to the variants
    xDSL emits as bare functions.  So this drops the wrapper and puts back only
    what the ABI actually asks of this body: whichever callee-saved registers
    it mentions.

    Only for a body that is one asm block.  A variant that is really compiled
    -- `naive_c`, the vendor libraries, `libxtcmm` -- keeps its prologue,
    because there the prologue is part of the code under test.
    """
    text = path.read_text()
    if text.count("#APP") != 1 or text.count("#NO_APP") != 1:
        raise StepFailed(f"{path} is not a single inline-asm body")

    head, rest = text.split("#APP", 1)
    body, _ = rest.split("#NO_APP", 1)

    # Which stack slot clang put each argument in, so a reload of that slot can
    # be answered from the register the ABI already has it in.
    spills = {
        match.group(2): match.group(1)
        for match in re.finditer(r"movq\s+(%r[a-z0-9]+), (-?\d+\(%rbp\))", head)
        if match.group(1) in ARGUMENT_REGISTERS
    }

    def reload(match: re.Match[str]) -> str:
        source = spills.get(match.group(1))
        if source is None:
            raise StepFailed(
                f"{path}: asm reads {match.group(1)}, which is not an argument slot"
            )
        return "" if source == match.group(2) else f"\tmovq\t{source}, {match.group(2)}"

    body = re.sub(r"\tmovq\t(-?\d+\(%rbp\)), (%r[a-z0-9]+)(?=\n)", reload, body)

    # A body that opens its own frame keeps `%rbp`, and preserves it itself, so
    # asking for it back here would only pay for the save twice.  The frame has
    # to be the first thing left in the body: a reload of one of clang's slots
    # that the rewrite above did not answer would sit ahead of it, and then the
    # mention of `%rbp` is the leftover this has always refused to emit.
    body = body.lstrip("\n").rstrip()
    own_frame = bool(OWN_FRAME_OPEN.match(body) and OWN_FRAME_CLOSE.search(body))
    if not own_frame and "%rbp" in body:
        raise StepFailed(f"{path}: asm body still reaches for the frame pointer")

    mentioned = [
        name
        for name, aliases in CALLEE_SAVED.items()
        if not (own_frame and name == "rbp")
        and any(re.search(rf"%{alias}\b", body) for alias in aliases)
    ]
    save = "".join(f"\tpushq\t%{name}\n" for name in mentioned)
    restore = "".join(f"\tpopq\t%{name}\n" for name in reversed(mentioned))

    path.write_text(
        "\t.text\n"
        "\t.globl\tmatmul\n"
        "\t.type\tmatmul,@function\n"
        "matmul:\n"
        f"{save}"
        f"{body}\n"
        f"{restore}"
        "\tretq\n"
        ".Lmatmul_end:\n"
        "\t.size\tmatmul, .Lmatmul_end-matmul\n"
        '\t.section\t".note.GNU-stack","",@progbits\n'
    )


def align_entry(path: Path, alignment: int) -> None:
    """Align an assembly file's `matmul` label to 2**`alignment` bytes.

    Whatever the producer emitted for that label is dropped first, so the
    directive left behind is the only one and every variant's entry point lands
    on the same boundary -- clang writes `.p2align 4`, xDSL writes nothing.
    """
    lines = path.read_text().splitlines(keepends=True)
    label = next(
        (i for i, line in enumerate(lines) if line.split("#")[0].strip() == "matmul:"),
        None,
    )
    if label is None:
        raise StepFailed(f"{path} defines no `matmul` label to align")

    # The producer's own directive sits between `.globl matmul` and the label,
    # among the `.type` and comment lines it also puts there.
    start = next(
        (
            i
            for i in range(label - 1, -1, -1)
            if ".globl" in lines[i] and "matmul" in lines[i]
        ),
        label,
    )
    aligners = (".p2align", ".align", ".balign")
    kept = [
        line
        for line in lines[start:label]
        if not line.split("#")[0].strip().startswith(aligners)
    ]
    directive = f"    .p2align {alignment}\n"
    path.write_text("".join(lines[:start] + kept + [directive] + lines[label:]))


def run_step(step: Step) -> None:
    """Execute one step, in this process wherever that is possible."""
    env = dict(step.env)
    match step.kind:
        case "run":
            run_command(step.args, env)

        case "remove":
            # The XSMM generators append to their output; a stale file left by
            # an interrupted run would otherwise be built on top of.
            Path(step.args[0]).unlink(missing_ok=True)

        case "generate:libxsmm" | "generate:compxsmm" | "generate:libxtcmm":
            module = import_module(GENERATOR_MODULES[step.kind])
            with environment(env):
                module.main(list(step.args))

        case "xdsl-opt":
            from xdsl.xdsl_opt_main import xDSLOptMain

            xDSLOptMain(args=list(step.args)).run()

        case "align":
            path, alignment = step.args
            align_entry(Path(path), int(alignment))

        case "bare":
            strip_asm_wrapper(Path(step.args[0]))

        case _:
            raise StepFailed(f"unknown step {step.kind!r}")


class environment:
    """Set these variables for the duration of a call, then put them back."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = values
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for name, value in self.values.items():
            self.saved[name] = os.environ.get(name)
            os.environ[name] = value

    def __exit__(self, *_: object) -> None:
        for name, old in self.saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def settle(artifact: Artifact) -> bool:
    """Put the rebuilt file in place; True when the bytes actually changed.

    A `preserve` artifact that came out identical is dropped rather than moved,
    so the file on disk -- and the measurement keyed to its mtime -- survives.
    """
    if not artifact.preserve:
        return True

    fresh = artifact.target
    if artifact.path.exists() and fresh.read_bytes() == artifact.path.read_bytes():
        fresh.unlink()
        return False
    os.replace(fresh, artifact.path)
    return True


def build_shape(job: ShapeJob) -> ShapeResult:
    """Build one shape's artifacts in order, skipping anything already broken."""
    built: list[str] = []
    unchanged: list[str] = []
    failures: list[tuple[str, str]] = []
    broken: set[Path] = set()

    for artifact in job.artifacts:
        if any(need in broken for need in artifact.needs):
            continue
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            for step in artifact.steps:
                run_step(step)
            changed = settle(artifact)
        except (Exception, SystemExit) as failure:  # one bad kernel is not fatal
            broken.add(artifact.path)
            artifact.target.unlink(missing_ok=True)
            failures.append((str(artifact.path), f"{failure}"))
            continue
        built.append(str(artifact.path))
        if not changed:
            unchanged.append(str(artifact.path))

    return ShapeResult(tuple(built), tuple(unchanged), tuple(failures))


# --- the manifest -----------------------------------------------------------


def manifest_path(tool: Toolchain) -> Path:
    return tool.root / tool.machine / MANIFEST


def load_manifest(tool: Toolchain) -> dict[str, str]:
    path = manifest_path(tool)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A manifest lost to an interrupted write costs a rebuild, not a run.
        return {}


def save_manifest(tool: Toolchain, manifest: Mapping[str, str]) -> None:
    path = manifest_path(tool)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(manifest, indent=0, sort_keys=True))
    os.replace(scratch, path)


def stale(artifact: Artifact, manifest: Mapping[str, str], force: bool) -> bool:
    if force or not artifact.path.exists():
        return True
    return manifest.get(str(artifact.path)) != artifact.key


# --- driver -----------------------------------------------------------------


def by_shape(samples: Iterable[Sample]) -> dict[tuple, list[Sample]]:
    """Group samples so one task builds every variant of a shape."""
    shapes: dict[tuple, list[Sample]] = {}
    for sample in samples:
        shapes.setdefault(
            (sample.kernel, sample.m, sample.n, sample.k, sample.dtype), []
        ).append(sample)
    return shapes


def workers(requested: int | None) -> int:
    if requested:
        return requested
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def build(
    samples: Sequence[Sample],
    machine: str,
    *,
    root: Path = Path("build"),
    jobs: int | None = None,
    force: bool = False,
    driver: str = "time",
    quiet: bool = False,
) -> None:
    """Generate and compile everything `samples` needs."""
    tool = toolchain(machine, root=root)
    manifest = load_manifest(tool)

    keys: dict[str, str] = {}
    pending: list[ShapeJob] = []
    total = 0
    for (kernel, m, n, k, dtype), group in by_shape(samples).items():
        artifacts = plan_shape(tool, group, driver)
        total += len(artifacts)
        for artifact in artifacts:
            keys[str(artifact.path)] = artifact.key
        # A binary is only stale if its own recipe changed, so a shape whose
        # sources are untouched costs nothing at all here.
        outstanding = tuple(a for a in artifacts if stale(a, manifest, force))
        if outstanding:
            pending.append(ShapeJob(f"{m}x{n}x{k} {dtype} {kernel}", outstanding))

    if not pending:
        if not quiet:
            console.print(f"[bold]up to date[/bold] ({total} artifacts)")
        return

    failures: list[tuple[str, str]] = []
    rebuilt = 0
    identical = 0
    pool = ProcessPoolExecutor(max_workers=workers(jobs))
    progress = Progress(
        TextColumn("[bold]building[/bold] {task.fields[shape]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=quiet,
    )
    try:
        with pool, progress:
            task = progress.add_task("", total=len(pending), shape="")
            for job, result in zip(
                pending, pool.map(build_shape, pending, chunksize=1)
            ):
                progress.update(task, shape=job.label, advance=1)
                for path in result.built:
                    manifest[path] = keys[path]
                rebuilt += len(result.built)
                identical += len(result.unchanged)
                failures.extend(result.failures)
    finally:
        # An interrupted run keeps what it finished, rather than rebuilding it.
        save_manifest(tool, manifest)

    if not quiet:
        console.print(
            f"[bold]built[/bold] {rebuilt} artifacts across {len(pending)} shapes"
            + (f", {identical} unchanged" if identical else "")
        )
    if failures:
        raise BuildFailed(
            "\n\n".join(f"{path}\n{message}" for path, message in failures[:10])
            + (f"\n\n... and {len(failures) - 10} more" if len(failures) > 10 else "")
        )


def selected_samples(machine: str, names: Sequence[str] | None) -> list[Sample]:
    """Every distinct sample the named datasets measure, in request order."""
    defined = dataset_samples(machine)
    unknown = set(names or ()) - set(defined)
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    chosen = {
        name: group
        for name, group in defined.items()
        if group and (names is None or name in names)
    }
    return list(dict.fromkeys(s for group in chosen.values() for s in group))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and compile the kernels the datasets measure."
    )
    parser.add_argument("datasets", nargs="*", help="datasets to build (default: all)")
    parser.add_argument("--machine", default=os.environ.get("MACHINE"))
    parser.add_argument("--jobs", type=int, default=None, help="workers (default: all)")
    parser.add_argument("--root", type=Path, default=Path("build"))
    parser.add_argument("--force", action="store_true", help="rebuild everything")
    args = parser.parse_args()

    if not args.machine:
        parser.error("no machine given; pass --machine or set MACHINE")

    samples = selected_samples(args.machine, args.datasets or None)
    if not samples:
        print(f"{args.machine} defines no samples for these datasets", file=sys.stderr)
        return

    try:
        build(
            samples,
            args.machine,
            root=args.root,
            jobs=args.jobs,
            force=args.force,
        )
    except BuildFailed as failure:
        console.print("[bold red]generating the kernels failed[/bold red]")
        console.print(f"{failure}", highlight=False)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
