configfile: "default.yaml"

import glob
import os
import shutil

from autotuner.datasets import (
    dataset_samples,
    machine_base,
    machine_file,
    variant_filename,
)

########################################################################################
# Build
########################################################################################

# MKL paths (discovered via pkg-config when available)

TVM_FUNC_NAME = "tvm_matmul"

if shutil.which("pkg-config") and os.system("pkg-config --exists mkl-dynamic-ilp64-seq") == 0:
    MKL_CFLAGS = shell("pkg-config --cflags mkl-dynamic-ilp64-seq", read=True).strip()
    MKL_LIBS   = shell("pkg-config --libs mkl-dynamic-ilp64-seq", read=True).strip()
else:
    MKL_CFLAGS = ""
    MKL_LIBS = ""

# AOCL-BLAS paths. The Nix package installs AMD's BLIS fork as the `blis`
# pkg-config module.

if shutil.which("pkg-config") and os.system("pkg-config --exists blis") == 0:
    AOCL_CFLAGS = shell("pkg-config --cflags blis", read=True).strip()
    AOCL_LIBS   = shell("pkg-config --libs blis", read=True).strip()
else:
    AOCL_CFLAGS = ""
    AOCL_LIBS = ""

# Machine-specific parameters

CC_ASM = config.get("cc_asm", config["cc"])

# MLIR/LLVM tools used to lower libxtcmm's transform-dialect IR. XTC resolves
# these from XTC_MLIR_PREFIX/XTC_LLVM_PREFIX, set by the Nix and Docker environments.
from xtc.utils.tools import get_mlir_prefix, get_llvm_prefix


def xtc_mlir_bin(_wildcards):
    return get_mlir_prefix(None) / "bin"


def xtc_llvm_bin(_wildcards):
    return get_llvm_prefix(None) / "bin"

from autotuner.machines import MACHINES

def machine_config(wildcards):
    return MACHINES[wildcards.machine]

def target_triple(wildcards):
    return machine_config(wildcards).target_triple

def compiler_march(wildcards):
    return machine_config(wildcards).march

def compiler_mtune(wildcards):
    return machine_config(wildcards).mtune

def machine_isa(wildcards):
    return machine_config(wildcards).isa

def machine_linker_flag(wildcards):
    return machine_config(wildcards).linker_flag

def machine_freq(wildcards):
    return machine_config(wildcards).freq

def machine_libs(wildcards):
    libs = machine_config(wildcards).libs
    if os.environ.get("USE_PAPI") == "1" and wildcards.machine == "ci":
        return (*libs, "papi")
    return libs

def machine_use_papi(wildcards):
    if 'papi' in machine_libs(wildcards):
        return '-DUSE_PAPI'
    return ''

def machine_env(wildcards):
    return ' '.join([f"{k}={v}" for k,v in machine_config(wildcards).env.items()])

def libxsmm_arch(wildcards):
    arch = machine_config(wildcards).libxsmm_arch
    if arch is None:
        raise ValueError(
            f"machine '{wildcards.machine}' does not support a libxsmm architecture"
        )
    return arch

def machine_libs_opts(wildcards):
    return " ".join(f"-l{x}" for x in machine_libs(wildcards))

LIBXSMM_GEMM_SOURCES = sorted(
    glob.glob("src/autotuner/libxsmm_gemm/**/*.py", recursive=True)
)

# CompXSMM currently reuses the generator modules from libxsmm_gemm, so depend on both
COMPXSMM_GEMM_SOURCES = LIBXSMM_GEMM_SOURCES + sorted(
    glob.glob("src/autotuner/compxsmm_gemm/**/*.py", recursive=True)
)

# libxtcmm reuses libxsmm_gemm's scheduling decisions plus its own emitter
LIBXTCMM_GEMM_SOURCES = LIBXSMM_GEMM_SOURCES + sorted(
    glob.glob("src/autotuner/libxtcmm_gemm/**/*.py", recursive=True)
)

# Rules

wildcard_constraints:
    dtype = "f32|f64",
    kernel="matmul_(rowmaj|colmaj)",
    executable="time|test",
    machine="|".join(MACHINES),
    variant="naive_c|naive_mlir|vector_intrinsic|transform_mlir|transform_xdsl|libxsmm|mkl|aocl|llvm_intrinsics|tvm|xdsl_libxsmm|compxsmm|compxsmm_manual|libxtcmm"

VARIANTS_ARITH = "naive_mlir|vector_intrinsic|transform_mlir"

rule templated_tensor:
    input: "kernels/{kernel}/mlir.mlir"
    output: machine_file(variant='tensor',ext='mlir')
    template_engine:
        "jinja2"

rule templated_vector_intrinsic:
    input: "kernels/{kernel}/vector_intrinsic.mlir"
    output: machine_file(variant='vector_intrinsic',ext='arith.mlir')
    template_engine:
        "jinja2"

rule merge_transform:
    input:
        matmul=machine_file(variant='memref',ext='mlir'),
        transform="kernels/{variant}.transform.mlir"
    output: machine_file(ext='transform.mlir')
    shell:
        './src/autotuner/merge_transform.awk {input.matmul} {input.transform} > {output}'

rule execute_transform:
    input: machine_file(ext='transform.mlir')
    output: machine_file(ext='transformed.mlir')
    shell:
        """mlir-opt {input} \
            --transform-interpreter \
            --mlir-print-op-generic \
        | xdsl-opt \
            -p test-transform-dialect-erase-schedule \
            --allow-unregistered-dialect \
            -o {output}"""

rule vector_to_arith:
    input: machine_file(variant='transform_mlir',ext='transformed.mlir')
    output: machine_file(variant='transform_mlir',ext='arith.mlir')
    shell:
        """mlir-opt {input} \
            --canonicalize \
            --lower-affine \
            --convert-vector-to-scf \
            --convert-scf-to-cf \
            -o {output}"""

rule transform_xdsl:
    input:
        "pyproject.toml",
        "src/autotuner/passes/vectorize_libxsmm.py",
        program = machine_file(variant='transform_xdsl',ext='transformed.mlir'),
    output: machine_file(variant='transform_xdsl',ext='vector.mlir')
    shell:
        """xdsl-opt -p vectorize-libxsmm{{vector-size=8}} {input.program} -o {output}"""

rule backend_xdsl:
    input:
        "pyproject.toml",
        program=machine_file(variant='transform_xdsl',ext='vector.mlir'),
    output: machine_file(variant='transform_xdsl',ext='S')
    params:
        passes=lambda wc: ",".join(config["xdsl-opt-backend-passes"][machine_isa(wc)])
    shell:
        """xdsl-opt -p '{params.passes}' -t x86-asm {input.program} -o {output}"""

rule memref_mlir:
    input: machine_file(variant='tensor',ext='mlir')
    output: machine_file(variant='memref',ext='mlir')
    shell:
        """mlir-opt {input} \
            --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
            -o {output}"""

rule naive_mlir:
    input: machine_file(variant='memref',ext='mlir')
    output: machine_file(variant='naive_mlir',ext='arith.mlir')
    shell:
        """mlir-opt {input} \
            --convert-linalg-to-loops \
            --convert-scf-to-cf \
            --buffer-results-to-out-params \
            -o {output}"""

rule arith_to_llvm:
    wildcard_constraints:
        variant = VARIANTS_ARITH
    input: machine_file(ext='arith.mlir')
    output: machine_file(ext='llvm.mlir')
    shell:
        """mlir-opt {input} \
            --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
            --finalize-memref-to-llvm \
            --canonicalize --cse --sccp \
            --convert-vector-to-llvm=enable-x86vector \
            --convert-index-to-llvm \
            --convert-arith-to-llvm \
            --convert-ub-to-llvm \
            --convert-cf-to-llvm \
            --canonicalize --cse --sccp \
            -o {output}"""

rule mlir_to_ll:
    wildcard_constraints:
        variant = VARIANTS_ARITH
    input: machine_file(ext='llvm.mlir')
    output: machine_file(ext='ll')
    shell:
        "mlir-translate --mlir-to-llvmir {input} -o {output}"

rule asm_ll:
    wildcard_constraints:
        variant = VARIANTS_ARITH
    input: machine_file(ext='ll')
    output: machine_file(ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        cc=CC_ASM,
    shell:
        "{params.cc} -O3 -S -fenable-matrix -Wno-override-module -target {params.target_triple} -march={params.compiler_march} -o {output} {input}"

rule asm_c:
    input: "kernels/{kernel}/naive_c.c"
    output: machine_file(variant='naive_c',ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        cc=CC_ASM,
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -O3 -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -S -target {params.target_triple} -march={params.compiler_march} -o {output} {input}"

rule libxsmm_colmaj_c:
    output: machine_file(kernel='matmul_colmaj',variant='libxsmm',ext='c')
    params:
        libxsmm_arch=libxsmm_arch,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        libxsmm_gemm_generator dense {output} matmul_colmaj \
            {wildcards.m} {wildcards.n} {wildcards.k} \
            {wildcards.m} {wildcards.k} {wildcards.m} \
            1 1 0 0 {params.libxsmm_arch} nopf {params.dtype}
        """

rule libxsmm_rowmaj_c:
    output: machine_file(kernel='matmul_rowmaj',variant='libxsmm',ext='c')
    params:
        libxsmm_arch=libxsmm_arch,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
        c_dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        libxsmm_gemm_generator dense {output} matmul_bac \
            {wildcards.n} {wildcards.m} {wildcards.k} \
            {wildcards.n} {wildcards.k} {wildcards.n} \
            1 1 \
            1 1 \
            {params.libxsmm_arch} \
            nopf \
            {params.dtype} && \
        echo 'void matmul({params.c_dtype} *A, {params.c_dtype} *B, {params.c_dtype} *C) {{matmul_bac(B, A, C);}}' >> {output}
        """


rule xdsl_libxsmm_rowmaj_mlir:
    input: ["pyproject.toml"] + LIBXSMM_GEMM_SOURCES
    output: machine_file(kernel='matmul_rowmaj',variant='xdsl_libxsmm',ext='libxsmm.mlir')
    params:
        libxsmm_arch=libxsmm_arch,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
        c_dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        SWAP_A_B=1 libxsmm-gemm dense {output} matmul \
            {wildcards.n} {wildcards.m} {wildcards.k} \
            {wildcards.n} {wildcards.k} {wildcards.n} \
            1 1 \
            1 1 \
            {params.libxsmm_arch} \
            nopf \
            {params.dtype}
        """

rule xdsl_libxsmm_s:
    input:
        mlir=machine_file(variant='xdsl_libxsmm',ext='libxsmm.mlir'),
        sources=["pyproject.toml"] + LIBXSMM_GEMM_SOURCES,
    output: machine_file(variant='xdsl_libxsmm',ext='S')
    params:
        passes=",".join(config["libxsmm-gemm-passes"])
    shell:
        """
        xdsl-opt {input.mlir} -p '{params.passes}' -t x86-asm -o {output}
        """

rule compxsmm_rowmaj_mlir:
    input: ["pyproject.toml"] + COMPXSMM_GEMM_SOURCES
    output: machine_file(kernel='matmul_rowmaj',variant='compxsmm',ext='compxsmm.mlir')
    params:
        libxsmm_arch=libxsmm_arch,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
        c_dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        SWAP_A_B=1 compxsmm-gemm dense {output} matmul \
            {wildcards.n} {wildcards.m} {wildcards.k} \
            {wildcards.n} {wildcards.k} {wildcards.n} \
            1 1 \
            1 1 \
            {params.libxsmm_arch} \
            nopf \
            {params.dtype} \
            --disable-regalloc
        """

rule compxsmm_s:
    input:
        mlir=machine_file(variant='compxsmm',ext='compxsmm.mlir'),
        sources=["pyproject.toml"] + COMPXSMM_GEMM_SOURCES,
    output: machine_file(variant='compxsmm',ext='S')
    params:
        passes=lambda wc: ",".join(config["compxsmm-gemm-passes"][machine_isa(wc)])
    shell:
        """
        xdsl-opt {input.mlir} -p '{params.passes}' -t x86-asm -o {output}
        """

# The same kernel with the registers assigned by the generator and the schedule
# rather than by xDSL, so a figure can put the two side by side.

rule compxsmm_manual_rowmaj_mlir:
    input: ["pyproject.toml"] + COMPXSMM_GEMM_SOURCES
    output: machine_file(kernel='matmul_rowmaj',variant='compxsmm_manual',ext='compxsmm.mlir')
    params:
        libxsmm_arch=libxsmm_arch,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        SWAP_A_B=1 compxsmm-gemm dense {output} matmul \
            {wildcards.n} {wildcards.m} {wildcards.k} \
            {wildcards.n} {wildcards.k} {wildcards.n} \
            1 1 \
            1 1 \
            {params.libxsmm_arch} \
            nopf \
            {params.dtype}
        """

rule compxsmm_manual_s:
    input:
        mlir=machine_file(variant='compxsmm_manual',ext='compxsmm.mlir'),
        sources=["pyproject.toml"] + COMPXSMM_GEMM_SOURCES,
    output: machine_file(variant='compxsmm_manual',ext='S')
    params:
        passes=lambda wc: ",".join(config["compxsmm-manual-gemm-passes"][machine_isa(wc)])
    shell:
        """
        xdsl-opt {input.mlir} -p '{params.passes}' -t x86-asm -o {output}
        """

rule libxtcmm_rowmaj_mlir:
    input: ["pyproject.toml"] + LIBXTCMM_GEMM_SOURCES
    output: machine_file(kernel='matmul_rowmaj',variant='libxtcmm',ext='xtcmm.mlir')
    params:
        libxsmm_arch=libxsmm_arch,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        # Emits `void matmul(A, B, C)` with the direct row-major ABI, so it links
        # into the timing/validation drivers exactly like the other variants.
        libxtcmm-gemm dense {output} matmul \
            {wildcards.n} {wildcards.m} {wildcards.k} \
            {wildcards.n} {wildcards.k} {wildcards.n} \
            1 1 \
            1 1 \
            {params.libxsmm_arch} \
            nopf \
            {params.dtype}
        """

rule libxtcmm_s:
    input:
        mlir=machine_file(variant='libxtcmm',ext='xtcmm.mlir'),
        sources=["pyproject.toml"] + LIBXTCMM_GEMM_SOURCES,
    output: machine_file(variant='libxtcmm',ext='S')
    params:
        passes=lambda wc: ",".join(config["libxtcmm-gemm-passes"][machine_isa(wc)]),
        triple=target_triple,
        march=compiler_march,
        mlir_bin=xtc_mlir_bin,
        llvm_bin=xtc_llvm_bin,
    shell:
        """
        # libxtcmm lowers with the Nix/Docker mlir-opt, mlir-translate, opt and
        # llc. Their paths are resolved through XTC's prefix helpers only when
        # this rule is selected, which gives a clear error if they are unavailable.
        # Replay XTC's own lowering here: apply the transform + lower to the llvm
        # dialect (mlir-opt), then mlir-translate/opt/llc with XTC's options.
        {params.mlir_bin}/mlir-opt {input.mlir} \
            -pass-pipeline="builtin.module({params.passes})" -o {output}.llvm.mlir
        {params.mlir_bin}/mlir-translate --mlir-to-llvmir {output}.llvm.mlir -o {output}.ll
        {params.llvm_bin}/opt -O2 --fp-contract=fast \
            -mtriple={params.triple} -mcpu={params.march} {output}.ll -o {output}.bc
        {params.llvm_bin}/llc -O2 -filetype=asm \
            -mtriple={params.triple} -mcpu={params.march} {output}.bc -o {output}
        """

rule mkl_rowmaj_s:
    output: machine_file(kernel='matmul_rowmaj',variant='mkl',ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        cc=config["cc"],
        dtype_flag=lambda w: "-DMKL_DTYPE_IS_FLOAT=1" if w.dtype=="f32" else "-DMKL_DTYPE_IS_DOUBLE=1",
    shell:
        "{params.cc} -O3 kernels/matmul_rowmaj/mkl.c {MKL_CFLAGS} -DMKL_M={wildcards.m} -DMKL_N={wildcards.n} -DMKL_K={wildcards.k} {params.dtype_flag} -S -target {params.target_triple} -march={params.compiler_march} -o {output}"

rule aocl_rowmaj_s:
    output: machine_file(kernel='matmul_rowmaj',variant='aocl',ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        cc=config["cc"],
        dtype_flag=lambda w: "-DAOCL_DTYPE_IS_FLOAT=1" if w.dtype=="f32" else "-DAOCL_DTYPE_IS_DOUBLE=1",
    shell:
        "{params.cc} -O3 kernels/matmul_rowmaj/aocl.c {AOCL_CFLAGS} -DAOCL_M={wildcards.m} -DAOCL_N={wildcards.n} -DAOCL_K={wildcards.k} {params.dtype_flag} -S -target {params.target_triple} -march={params.compiler_march} -o {output}"

rule llvm_intrinsics_rowmaj_s:
    output: machine_file(kernel='matmul_rowmaj',variant='llvm_intrinsics',ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        compiler_mtune=compiler_mtune,
        cc=CC_ASM,
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -O3 -c kernels/matmul_rowmaj/llvm_intrinsics.c -DM={wildcards.m} -DN={wildcards.n} -DK={wildcards.k} -DDTYPE={params.dtype} -S -fenable-matrix -target {params.target_triple} -march={params.compiler_march} -mtune={params.compiler_mtune} -o {output} -ffp-contract=fast -ffast-math -mprefer-vector-width=512"

rule tvm_rowmaj_c:
    output: machine_file(kernel='matmul_rowmaj',variant='tvm',ext='c')
    params:
        compiler_march=compiler_march,
        dtype=lambda wildcards: {"f32": "float32", "f64": "float64"}[wildcards.dtype],
    shell:
        """
        {{
            TVM_NUM_THREADS=1 python3.12 kernels/matmul_rowmaj/tvm_matmul_row_major.py \
                --M {wildcards.m} --N {wildcards.n} --K {wildcards.k} \
                --dtype {params.dtype} --symbol {TVM_FUNC_NAME} --cpu {params.compiler_march}
            cat kernels/matmul_rowmaj/tvm_matmul_wrapper.c
        }} > {output}
        """

rule tvm_rowmaj_s:
    input: machine_file(kernel='matmul_rowmaj',variant='tvm',ext='c')
    output: machine_file(kernel='matmul_rowmaj',variant='tvm',ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "MM_DTYPE_float", "f64": "MM_DTYPE_double"}[wildcards.dtype],
    shell:
                "{params.cc} -O3 -c {input} -DKERNEL_FUNC=matmul -DPACKED_FUNC={TVM_FUNC_NAME} -DMM_I={wildcards.m} -DMM_J={wildcards.n} -DMM_K={wildcards.k} -DMM_DTYPE={params.dtype} -S -target {params.target_triple} -march={params.compiler_march} -o {output}"

rule libxsmm_s:
    input: machine_file(variant='libxsmm',ext='c')
    output: machine_file(variant='libxsmm',ext='S')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        cc=config["cc"],
    shell:
        "{params.cc} -O3 -DNDEBUG {input} -S -target {params.target_triple} -march={params.compiler_march} -o {output}"

rule executable:
    input: machine_file(ext='S')
    output: machine_file(ext='{executable}.o')
    params:
        target_triple=target_triple,
        compiler_march=compiler_march,
        machine_libs_opts=machine_libs_opts,
        machine_freq=machine_freq,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
        use_papi=machine_use_papi,
        mkl_libs=lambda wc: MKL_LIBS if wc.variant == "mkl" else "",
        aocl_libs=lambda wc: AOCL_LIBS if wc.variant == "aocl" else "",
        linker_flag=machine_linker_flag,
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -DFREQ={params.machine_freq} {params.use_papi} -target {params.target_triple} -march={params.compiler_march} -o {output} kernels/{wildcards.kernel}/{wildcards.executable}.c {input} {params.machine_libs_opts} {params.mkl_libs} {params.aocl_libs} {params.linker_flag}"

rule validation:
    input:  machine_file(ext='test.o')
    log:    machine_file(ext='test.log')
    shell:  '{input} > {log}'

########################################################################################
# Time
########################################################################################

rule time:
    input: machine_file(ext="time.o")
    output: machine_file(ext="time.txt")
    params: machine_env=machine_env,
    shell: 'OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1 {params.machine_env} {input} > {output}'

########################################################################################
# Dataset
########################################################################################

# Select the machine by passing `--config machine=NAME` to Snakemake, setting
# MACHINE=NAME for Make, or adding MACHINE=NAME to .env.
THIS_MACHINE = config["machine"]

# What each dataset measures lives in autotuner.datasets, which the evaluate
# script shares; `uv run evaluate` builds these, times them and writes the
# jsonl.  The build itself is `autotuner.build` rather than a rule here: at
# ~37k sub-second jobs Snakemake's dispatch loop, not the machine, was the
# limit.  What is left is validation, which is a handful of jobs per sample.
DATASETS = dataset_samples(THIS_MACHINE)

# `--config datasets=a,b` narrows the build to the datasets being evaluated.
SELECTED = (
    config["datasets"].split(",") if config.get("datasets") else list(DATASETS)
)

def dataset_files(ext):
    """Every distinct file of this kind the selected datasets need."""
    return list(dict.fromkeys(
        sample.path(THIS_MACHINE, ext)
        for name in SELECTED
        for sample in DATASETS[name]
    ))

# The timing kernels are built by `uv run build-dataset` (see `make
# dataset_code`); building them here as well would relink every binary and so
# invalidate every cached measurement.

rule dataset_validate:
    input: dataset_files("test.log")

########################################################################################
# CI
########################################################################################

from typing import NamedTuple

class Kernel3D(NamedTuple):
    kernel: str
    m: int
    n: int
    k: int

KERNELS_CI = [
    Kernel3D("matmul_rowmaj", 4, 4, 4),
    Kernel3D("matmul_rowmaj", 5, 6, 7),
    Kernel3D("matmul_colmaj", 4, 4, 4),
    Kernel3D("matmul_colmaj", 5, 6, 7),
]

VARIANT_CI = [
    "naive_c",
    "naive_mlir",
    "vector_intrinsic",
]

# Kernel matrix sizes shared by CI test lists.
def testset_ci(machine: str, ext: str):
    return expand(
        "build/" + machine + "/{k.kernel}/{k.m}x{k.n}x{k.k}/{variant}.{dtype}." + ext,
        k=KERNELS_CI,
        variant=VARIANT_CI,
        dtype=["f32", "f64"],
    )


# Exercise XTC's F32 emitter and the complete MLIR/LLVM lowering without
# requiring the CI host to execute AVX-512 instructions.
LIBXTCMM_COMPILE_SMOKE = machine_file(
    kernel="matmul_rowmaj", m="29", n="16", k="25",
    variant="libxtcmm", dtype="f32", machine="tower", ext="S"
)

TESTSET_MAC = [
    # Validate CI test set neon executables
    *testset_ci(machine="neon", ext="test.log"),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", machine="neon", ext="test.log"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", machine="neon", ext="time.txt"
    ),
    # Generate CI test set x86 assembly
    *testset_ci(machine="ci", ext="S"),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", machine="ci", ext="S"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="vector_intrinsic", dtype="f32", machine="ci", ext="S"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="3", n="16", k="5",
        variant="transform_xdsl", dtype="f64", machine="tower", ext="S"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="6", n="32", k="5",
        variant="transform_xdsl", dtype="f64", machine="tower", ext="S"
    ),
    LIBXTCMM_COMPILE_SMOKE,
]

TESTSET_CI = [
    # Generate CI test set neon assembly
    *testset_ci(machine="neon", ext="S"),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", machine="neon", ext="S"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="vector_intrinsic", dtype="f32", machine="neon", ext="S"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="3", n="16", k="5",
        variant="transform_xdsl", dtype="f64", machine="tower", ext="S"
    ),
    LIBXTCMM_COMPILE_SMOKE,
    # Validate CI test set x86 executables
    *testset_ci(machine="ci", ext="test.log"),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", machine="ci", ext="test.log"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", machine="ci", ext="time.txt"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="vector_intrinsic", dtype="f32", machine="ci", ext="time.txt"
    ),
    machine_file(
        kernel="matmul_rowmaj", m="5", n="6", k="7",
        variant="transform_mlir", dtype="f32", machine="ci", ext="time.txt"
    ),
]

# Functional coverage for the Python XSMM generators.
KERNELS_XSMM_AVX512 = [
    Kernel3D("matmul_rowmaj", 5, 34, 16),  # fully unrolled K, multiple M tiles
    Kernel3D("matmul_rowmaj", 29, 16, 16),  # fully unrolled K, multiple N blocks
    Kernel3D("matmul_rowmaj", 5, 34, 24),  # exact tiled K loop
    Kernel3D("matmul_rowmaj", 5, 34, 25),  # tiled K loop plus remainder
]

# For machines that can execute the AVX-512 ISA
TESTSET_AVX512 = [
    *expand(
        machine_file(
            kernel="matmul_rowmaj", m="3", n="16", k="5", dtype="f64",
            machine=THIS_MACHINE,
            ext="test.log",
        ),
        variant=[
            "transform_xdsl",
            "llvm_intrinsics",
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "libxtcmm",
            "mkl",
            "aocl",
        ],
    ),
    machine_file(
        kernel="matmul_rowmaj", m="29", n="16", k="25", dtype="f64",
        machine=THIS_MACHINE, variant="libxtcmm", ext="test.log",
    ),
    # Exercise the Python generators across M/N blocking and all K-loop strategies.
    *expand(
        "build/"
        + THIS_MACHINE
        + "/{case.kernel}/{case.m}x{case.n}x{case.k}/{variant}.f64.test.log",
        case=KERNELS_XSMM_AVX512,
        variant=["xdsl_libxsmm", "compxsmm", "compxsmm_manual"],
    ),
]

TESTSET_BASE = TESTSET_MAC if THIS_MACHINE == "neon" else TESTSET_CI
TESTSET = TESTSET_BASE + (
    TESTSET_AVX512 if MACHINES[THIS_MACHINE].isa == "avx512" else []
)

rule tests:
    input: TESTSET
