configfile: "default.yaml"

import glob
import os
import shutil

from xdsl.dialects import builtin

from autotuner.skx_nano_kernel import (
    SKX_NANO_KERNELS,
    SkxTargetInfo,
    get_skx_nano_kernel,
)

# Use the steady-state K tile from larger GEMMs for the 2D nano-kernel plots.
# This is a benchmarking convention, not a nano-kernel correctness limit.
NANO_KERNEL_BENCHMARK_K = 4

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

# Target-specific parameters

CC_ASM = config.get("cc_asm", config["cc"])

T = config["targets"]
if os.environ.get("USE_PAPI") == "1":
    T['ci']['libs'].append('papi')

def target_triple(wildcards):
    return T[wildcards.target]['triple']

def target_freq(wildcards):
    return T[wildcards.target]['freq']

def target_arch(wildcards):
    return T[wildcards.target]['arch']

def target_linker_flag(wildcards):
    return T[wildcards.target]['linker_flag']

def target_peak_flops(wildcards):
    match wildcards.dtype:
        case 'f32':
            factor = 1.0
        case 'f64':
            factor = 2.0
        case _:
            assert False
    flops_per_cycle = T[wildcards.target]['peak_f32'] / factor
    return flops_per_cycle

def target_use_papi(wildcards):
    if 'papi' in T[wildcards.target]['libs']:
        return '-DUSE_PAPI'
    return ''

def target_env(wildcards):
    return ' '.join([f"{k}={v}" for k,v in T[wildcards.target]['env'].items()])

def target_xsmm(wildcards):
    arch = target_arch(wildcards)
    return config['xsmm_map'][arch]

def target_libs_opts(wildcards):
    return " ".join(f"-l{x}" for x in T[wildcards.target]['libs'])

# Path management

def target_base(
        target = "{target}",
        kernel = "{kernel}",
        m = "{m}",
        n = "{n}",
        k = "{k}"
):
    return f"build/{target}/{kernel}/{m}x{n}x{k}"

def target_variant(
        ext,
        variant = "{variant}",
        dtype = "{dtype}"
):
    return f"{variant}.{dtype}.{ext}"

def target_file(
    ext,
    target = "{target}",
    kernel = "{kernel}",
    m = "{m}",
    n = "{n}",
    k = "{k}",
    variant = "{variant}",
    dtype = "{dtype}",
):
    base = target_base(target=target, kernel=kernel, m=m, n=n, k=k)
    var = target_variant(variant=variant, dtype=dtype, ext=ext)
    return f"{base}/{var}"

def target_ll_file(
    ext,
    kernel = "{kernel}",
    m = "{m}",
    n = "{n}",
    k = "{k}",
    variant = "{variant}",
    dtype = "{dtype}",
    target = "{target}"
):
    base = target_base(target=target, kernel=kernel, m=m, n=n, k=k)
    var = target_variant(variant=variant, dtype=dtype, ext=ext)
    return f"{base}/{var}"

LIBXSMM_GEMM_SOURCES = sorted(
    glob.glob("src/autotuner/libxsmm_gemm/**/*.py", recursive=True)
)

# CompXSMM currently reuses the generator modules from libxsmm_gemm, so depend on both
COMPXSMM_GEMM_SOURCES = LIBXSMM_GEMM_SOURCES + sorted(
    glob.glob("src/autotuner/compxsmm_gemm/**/*.py", recursive=True)
)

NANO_KERNEL_SOURCES = sorted(
    glob.glob("src/autotuner/**/*.py", recursive=True)
)
NANO_KERNEL_NAMES = tuple(sorted(SKX_NANO_KERNELS))
NANO_KERNEL_PATTERN = "|".join(NANO_KERNEL_NAMES)
VARIANT_PATTERN = (
    "naive_c|naive_mlir|vector_intrinsic|transform_mlir|transform_xdsl|"
    "libxsmm|mkl|llvm_intrinsics|tvm|xdsl_libxsmm|compxsmm|"
    f"compxsmm-({NANO_KERNEL_PATTERN})"
)

# Rules

wildcard_constraints:
    dtype = "f32|f64",
    kernel="matmul_(rowmaj|colmaj)",
    executable="time|test",
    target="neon|ci|tower|pinocchio",
    variant=VARIANT_PATTERN,
    nano_kernel=NANO_KERNEL_PATTERN

VARIANTS_ARITH = "naive_mlir|vector_intrinsic|transform_mlir"

rule templated_tensor:
    input: "kernels/{kernel}/mlir.mlir"
    output: target_file(variant='tensor',ext='mlir')
    template_engine:
        "jinja2"

rule templated_vector_intrinsic:
    input: "kernels/{kernel}/vector_intrinsic.mlir"
    output: target_file(variant='vector_intrinsic',ext='arith.mlir')
    template_engine:
        "jinja2"

rule merge_transform:
    input:
        matmul=target_file(variant='memref',ext='mlir'),
        transform="kernels/{variant}.transform.mlir"
    output: target_file(ext='transform.mlir')
    shell:
        './src/autotuner/merge_transform.awk {input.matmul} {input.transform} > {output}'

rule execute_transform:
    input: target_file(ext='transform.mlir')
    output: target_file(ext='transformed.mlir')
    shell:
        """mlir-opt {input} \
            --transform-interpreter \
            --mlir-print-op-generic \
        | xdsl-opt \
            -p test-transform-dialect-erase-schedule \
            --allow-unregistered-dialect \
            -o {output}"""

rule vector_to_arith:
    input: target_file(variant='transform_mlir',ext='transformed.mlir')
    output: target_file(variant='transform_mlir',ext='arith.mlir')
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
        program = target_file(variant='transform_xdsl',ext='transformed.mlir'),
    output: target_file(variant='transform_xdsl',ext='vector.mlir')
    shell:
        """xdsl-opt -p vectorize-libxsmm{{vector-size=8}} {input.program} -o {output}"""

rule backend_xdsl:
    input:
        "pyproject.toml",
        program=target_file(variant='transform_xdsl',ext='vector.mlir'),
    output: target_ll_file(variant='transform_xdsl',ext='S')
    params:
        passes = ",".join(config["xdsl-opt-backend-passes"])
    shell:
        """xdsl-opt -p {params.passes} -t x86-asm {input.program} -o {output}"""

rule memref_mlir:
    input: target_file(variant='tensor',ext='mlir')
    output: target_file(variant='memref',ext='mlir')
    shell:
        """mlir-opt {input} \
            --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
            -o {output}"""

rule naive_mlir:
    input: target_file(variant='memref',ext='mlir')
    output: target_file(variant='naive_mlir',ext='arith.mlir')
    shell:
        """mlir-opt {input} \
            --convert-linalg-to-loops \
            --convert-scf-to-cf \
            --buffer-results-to-out-params \
            -o {output}"""

rule arith_to_llvm:
    wildcard_constraints:
        variant = VARIANTS_ARITH
    input: target_file(ext='arith.mlir')
    output: target_file(ext='llvm.mlir')
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
    input: target_file(ext='llvm.mlir')
    output: target_file(ext='ll')
    shell:
        "mlir-translate --mlir-to-llvmir {input} -o {output}"

rule asm_ll:
    wildcard_constraints:
        variant = VARIANTS_ARITH
    input: target_file(ext='ll')
    output: target_ll_file(ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=CC_ASM,
    shell:
        "{params.cc} -O3 -S -fenable-matrix -Wno-override-module -target {params.target_triple} -march={params.target_arch} -o {output} {input}"

rule asm_c:
    input: "kernels/{kernel}/naive_c.c"
    output: target_ll_file(variant='naive_c',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=CC_ASM,
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -O3 -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -S -target {params.target_triple} -march={params.target_arch} -o {output} {input}"

rule libxsmm_colmaj_c:
    output: target_ll_file(kernel='matmul_colmaj',variant='libxsmm',ext='c')
    params:
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        libxsmm_gemm_generator dense {output} matmul_colmaj \
            {wildcards.m} {wildcards.n} {wildcards.k} \
            {wildcards.m} {wildcards.k} {wildcards.m} \
            1 1 0 0 hsw nopf {params.dtype}
        """

rule libxsmm_rowmaj_c:
    output: target_ll_file(kernel='matmul_rowmaj',variant='libxsmm',ext='c')
    params:
        target_xsmm=target_xsmm,
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
            {params.target_xsmm} \
            nopf \
            {params.dtype} && \
        echo 'void matmul({params.c_dtype} *A, {params.c_dtype} *B, {params.c_dtype} *C) {{matmul_bac(B, A, C);}}' >> {output}
        """


rule xdsl_libxsmm_rowmaj_mlir:
    input: ["pyproject.toml"] + LIBXSMM_GEMM_SOURCES
    output: target_ll_file(kernel='matmul_rowmaj',variant='xdsl_libxsmm',ext='libxsmm.mlir')
    params:
        target_xsmm=target_xsmm,
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
            {params.target_xsmm} \
            nopf \
            {params.dtype}
        """

rule xdsl_libxsmm_s:
    input:
        mlir=target_ll_file(variant='xdsl_libxsmm',ext='libxsmm.mlir'),
        sources=["pyproject.toml"] + LIBXSMM_GEMM_SOURCES,
    output: target_ll_file(variant='xdsl_libxsmm',ext='S')
    params:
        passes=",".join(config["libxsmm-gemm-passes"])
    shell:
        """
        xdsl-opt {input.mlir} -p {params.passes} -t x86-asm -o {output}
        """

rule compxsmm_rowmaj_mlir:
    input: ["pyproject.toml"] + COMPXSMM_GEMM_SOURCES
    output: target_ll_file(kernel='matmul_rowmaj',variant='compxsmm',ext='compxsmm.mlir')
    params:
        target_xsmm=target_xsmm,
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
            {params.target_xsmm} \
            nopf \
            {params.dtype} \
            --disable-regalloc
        """

rule compxsmm_s:
    input:
        mlir=target_ll_file(variant='compxsmm',ext='compxsmm.mlir'),
        sources=["pyproject.toml"] + COMPXSMM_GEMM_SOURCES,
    output: target_ll_file(variant='compxsmm',ext='S')
    params:
        passes=",".join(config["compxsmm-gemm-passes"])
    shell:
        """
        xdsl-opt {input.mlir} -p {params.passes} -t x86-asm -o {output}
        """

# Expand exactly one requested nano-kernel shape. Unlike the normal CompXSMM
# pipeline, this deliberately does not tile N or K: K=4 represents the
# steady-state K tile used inside larger GEMMs, and is a benchmark convention
# rather than a correctness limit of the nano-kernel implementation.
rule compxsmm_nano_kernel_s:
    wildcard_constraints:
        k=str(NANO_KERNEL_BENCHMARK_K)
    input:
        mlir=target_ll_file(kernel='matmul_rowmaj',variant='compxsmm',ext='compxsmm.mlir'),
        sources=["pyproject.toml"] + NANO_KERNEL_SOURCES,
    output: target_ll_file(kernel='matmul_rowmaj',variant='compxsmm-{nano_kernel}',ext='S')
    params:
        passes=lambda wildcards: ",".join((
            "xsmm-matmul-n-to-m",
            "xsmm-tile-m{disable-regalloc=true}",
            "xsmm-matmul-m-to-k",
            "convert-xsmm-to-x86{"
            f"nano-kernel={wildcards.nano_kernel} disable-regalloc=true"
            "}",
            "x86-regalloc-verify-liveness",
            "x86-allocate-registers",
            "convert-x86-scf-to-x86",
            "x86-prologue-epilogue-insertion",
        ))
    shell:
        """
        xdsl-opt {input.mlir} -p '{params.passes}' -t x86-asm -o {output}
        """

rule mkl_rowmaj_s:
    output: target_ll_file(kernel='matmul_rowmaj',variant='mkl',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
        dtype_flag=lambda w: "-DMKL_DTYPE_IS_FLOAT=1" if w.dtype=="f32" else "-DMKL_DTYPE_IS_DOUBLE=1",
    shell:
        "{params.cc} -O3 kernels/matmul_rowmaj/mkl.c {MKL_CFLAGS} -DMKL_M={wildcards.m} -DMKL_N={wildcards.n} -DMKL_K={wildcards.k} {params.dtype_flag} -S -target {params.target_triple} -march={params.target_arch} -o {output}"

rule llvm_intrinsics_rowmaj_s:
    output: target_ll_file(kernel='matmul_rowmaj',variant='llvm_intrinsics',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=CC_ASM,
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -O3 -c kernels/matmul_rowmaj/llvm_intrinsics.c -DM={wildcards.m} -DN={wildcards.n} -DK={wildcards.k} -DDTYPE={params.dtype} -S -fenable-matrix -target {params.target_triple} -march={params.target_arch} -mtune={params.target_arch} -o {output} -ffp-contract=fast -ffast-math -mprefer-vector-width=512"

rule tvm_rowmaj_c:
    output: target_ll_file(kernel='matmul_rowmaj',variant='tvm',ext='c')
    params:
        target_arch=target_arch,
        dtype=lambda wildcards: {"f32": "float32", "f64": "float64"}[wildcards.dtype],
    shell:
        """
        {{
            TVM_NUM_THREADS=1 python3.12 kernels/matmul_rowmaj/tvm_matmul_row_major.py \
                --M {wildcards.m} --N {wildcards.n} --K {wildcards.k} \
                --dtype {params.dtype} --symbol {TVM_FUNC_NAME} --cpu {params.target_arch}
            cat kernels/matmul_rowmaj/tvm_matmul_wrapper.c
        }} > {output}
        """

rule tvm_rowmaj_s:
    input: target_ll_file(kernel='matmul_rowmaj',variant='tvm',ext='c')
    output: target_ll_file(kernel='matmul_rowmaj',variant='tvm',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "MM_DTYPE_float", "f64": "MM_DTYPE_double"}[wildcards.dtype],
    shell:
                "{params.cc} -O3 -c {input} -DKERNEL_FUNC=matmul -DPACKED_FUNC={TVM_FUNC_NAME} -DMM_I={wildcards.m} -DMM_J={wildcards.n} -DMM_K={wildcards.k} -DMM_DTYPE={params.dtype} -S -target {params.target_triple} -march={params.target_arch} -o {output}"

rule libxsmm_s:
    input: target_ll_file(variant='libxsmm',ext='c')
    output: target_ll_file(variant='libxsmm',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
    shell:
        "{params.cc} -O3 -DNDEBUG {input} -S -target {params.target_triple} -march={params.target_arch} -o {output}"

rule executable:
    input: target_ll_file(ext='S')
    output: target_ll_file(ext='{executable}.o')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        target_libs_opts=target_libs_opts,
        target_freq=target_freq,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
        use_papi=target_use_papi,
        mkl_libs=lambda wc: MKL_LIBS if wc.variant == "mkl" else "",
        linker_flag=target_linker_flag,
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -DFREQ={params.target_freq} {params.use_papi} -target {params.target_triple} -march={params.target_arch} -o {output} kernels/{wildcards.kernel}/{wildcards.executable}.c {input} {params.target_libs_opts} {params.mkl_libs} {params.linker_flag}"

rule validation:
    input:  target_ll_file(ext='test.o')
    log:    target_ll_file(ext='test.log')
    shell:  '{input} > {log}'

########################################################################################
# Time
########################################################################################

rule time:
    input: target_ll_file(ext="time.o")
    output: target_ll_file(ext="time.txt")
    params: target_env=target_env,
    shell: 'OMP_NUM_THREADS=1  {params.target_env} {input} > {output}'

rule flops:
    input: "kernels/{kernel}/flops.sh"
    output: "build/{target}/{kernel}/{m}x{n}x{k}/flops.txt"
    shell: "./{input} {wildcards.m} {wildcards.n} {wildcards.k} > {output}"

rule json:
    input:
        time_txt=target_ll_file(ext="time.txt"),
        flops_txt="build/{target}/{kernel}/{m}x{n}x{k}/flops.txt"
    output:
        json=target_ll_file(ext="json")
    params:
        target_peak_flops=target_peak_flops,
    shell:
        """
        M={wildcards.m}
        N={wildcards.n}
        K={wildcards.k}
        PEAK="{params.target_peak_flops}"
        FLOPS=$(head -n 1 {input.flops_txt} | tr -d '[:space:]')
        TIME=$(head -n 1 {input.time_txt} | tr -d '[:space:]')
        VARIANT="{wildcards.variant}"
        TARGET="{wildcards.target}"
        DTYPE="{wildcards.dtype}"
        echo '{{"M":'${{M}}',"N":'${{N}}',"K":'${{K}}',"peak":'${{PEAK}}',"flops":'${{FLOPS}}',"time":'${{TIME}}',"variant":"'${{VARIANT}}'","target":"'${{TARGET}}'","dtype":"'${{DTYPE}}'"}}' > {output.json}
        """

########################################################################################
# Dataset
########################################################################################

# Set the target by:
# passing `--target=THIS_TARGET` when running snakemake
# passing TARGET="THIS_TARGET" when running make
# adding `TARGET="THIS_TARGET"` in .env, which will be read by make automatically
THIS_TARGET = config["target"]

DATASET_VARIANTS = {
    "neon": {
        "ttile": ["naive_c"],
        "f64.small_matrices": [],
    },
    "tower": {
        "ttile": ["naive_c", "libxsmm", "mkl", "xdsl_libxsmm", "compxsmm"],
        "f64.small_matrices": ["libxsmm", "xdsl_libxsmm", "compxsmm"],
    },
    "pinocchio": {
        "ttile": ["naive_c", "libxsmm", "mkl"],
        "f64.small_matrices": ["llvm_intrinsics", "libxsmm","mkl"],
    },
    "ci": {
        "ttile": ["naive_c"],
        "f64.small_matrices": [],
    },
}[THIS_TARGET]

# Values are missing the extension
# The extension is added in the dataset rules below
DATASET_BASES = {
    "f32.ttile": expand(
        target_file(
            kernel="matmul_rowmaj",
            n="128",
            k="128",
            dtype="f32",
            ext="",
            target=THIS_TARGET,
        ),
        variant=DATASET_VARIANTS["ttile"],
        m=range(8, 50, 2),
    ),
    "f64.ttile": expand(
        target_file(
            kernel="matmul_rowmaj",
            n="64",
            k="64",
            dtype="f64",
            ext="",
            target=THIS_TARGET,
        ),
        variant=DATASET_VARIANTS["ttile"], # + ["transform_xdsl"]
        m=range(9, 63, 3),
    ),
    "f64.small_matrices": expand(
        target_file(
            kernel="matmul_rowmaj",
            k="64",
            dtype="f64",
            ext="",
            target=THIS_TARGET,
        ),
        m=range(1, 17),
        n=range(1, 17),
        variant=DATASET_VARIANTS["f64.small_matrices"]
    )
}


def nano_kernel_sample_bases(target, dtype, nano_kernel_name):
    """Return one benchmark base path per supported M/N tile."""
    datatype = {"f32": builtin.f32, "f64": builtin.f64}[dtype]
    nano_kernel = get_skx_nano_kernel(nano_kernel_name)
    supported_tiles = nano_kernel.supported_tile_sizes(datatype, SkxTargetInfo())
    return [
        target_file(
            target=target,
            kernel="matmul_rowmaj",
            # The row-major wrapper swaps A/B and the GEMM M/N dimensions
            # before entering the column-major generator. Reverse them in the
            # build path so the generated MatmulK has exactly (tile.m, tile.n).
            m=str(tile.n),
            n=str(tile.m),
            k=str(NANO_KERNEL_BENCHMARK_K),
            variant=f"compxsmm-{nano_kernel_name}",
            dtype=dtype,
            ext="",
        )
        for tile in sorted(supported_tiles)
    ]


def nano_kernel_dataset_inputs(wildcards):
    """Derive all datapoints from the requested dataset filename."""
    return [
        base + "json"
        for base in nano_kernel_sample_bases(
            wildcards.target,
            wildcards.dtype,
            wildcards.nano_kernel,
        )
    ]


NANO_KERNEL_DATASET_TARGETS = ("tower", "pinocchio")
NANO_KERNEL_DATASET_OUTPUTS = (
    expand(
        "data/{target}/{dtype}.nano-kernel.{nano_kernel}.jsonl",
        target=THIS_TARGET,
        dtype=("f32", "f64"),
        nano_kernel=NANO_KERNEL_NAMES,
    )
    if THIS_TARGET in NANO_KERNEL_DATASET_TARGETS
    else []
)
NANO_KERNEL_SAMPLE_BASES = (
    [
        base
        for dtype in ("f32", "f64")
        for nano_kernel in NANO_KERNEL_NAMES
        for base in nano_kernel_sample_bases(THIS_TARGET, dtype, nano_kernel)
    ]
    if THIS_TARGET in NANO_KERNEL_DATASET_TARGETS
    else []
)
ALL_DATASET_SAMPLE_BASES = (
    *flatten(DATASET_BASES.values()),
    *NANO_KERNEL_SAMPLE_BASES,
)
DATASET_CODE_INPUTS = [p + "time.o" for p in ALL_DATASET_SAMPLE_BASES]
DATASET_VALIDATION_INPUTS = [p + "test.log" for p in ALL_DATASET_SAMPLE_BASES]
DATASET_OUTPUTS = [
    *expand(
        "data/{target}/{dataset}.jsonl",
        target=THIS_TARGET,
        dataset=tuple(
            dataset for dataset, samples in DATASET_BASES.items() if samples
        ),
    ),
    *NANO_KERNEL_DATASET_OUTPUTS,
]

# If a dataset has no samples skip it here and in the dataset rule below
for dataset, samples in DATASET_BASES.items():
    if samples:
        rule:
            input: [base + "json" for base in samples]
            output: f"data/{THIS_TARGET}/{dataset}.jsonl"
            shell: "cat {input} > {output}"

rule nano_kernel_dataset:
    wildcard_constraints:
        target="tower|pinocchio",
    input: nano_kernel_dataset_inputs
    output: "data/{target}/{dtype}.nano-kernel.{nano_kernel}.jsonl"
    shell: "cat {input} > {output}"

rule dataset_code:
    input: DATASET_CODE_INPUTS

rule dataset_validate:
    input: DATASET_VALIDATION_INPUTS

rule dataset:
    input: DATASET_OUTPUTS

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

# Kernel matrix sizes shared by CI test lists (path prefix is build/<target>/… per machine).
def testset_ci(target: str, ext: str):
    return expand(
        "build/" + target + "/{k.kernel}/{k.m}x{k.n}x{k.k}/{variant}.{dtype}." + ext,
        k=KERNELS_CI,
        variant=VARIANT_CI,
        dtype=["f32", "f64"],
    )

TESTSET_MAC = [
    # Validate CI test set neon executables
    *testset_ci(target="neon", ext="test.log"),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", target="neon", ext="test.log"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", target="neon", ext="time.txt"
    ),
    # Generate CI test set x86 assembly
    *testset_ci(target="ci", ext="S"),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", target="ci", ext="S"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="vector_intrinsic", dtype="f32", target="ci", ext="S"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="3", n="16", k="5",
        variant="transform_xdsl", dtype="f64", target="tower", ext="S"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="6", n="32", k="5",
        variant="transform_xdsl", dtype="f64", target="tower", ext="S"
    ),
]

TESTSET_CI = [
    # Generate CI test set neon assembly
    *testset_ci(target="neon", ext="S"),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", target="neon", ext="S"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="vector_intrinsic", dtype="f32", target="neon", ext="S"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="3", n="16", k="5",
        variant="transform_xdsl", dtype="f64", target="tower", ext="S"
    ),
    # Validate CI test set x86 executables
    *testset_ci(target="ci", ext="test.log"),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", target="ci", ext="test.log"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="transform_mlir", dtype="f32", target="ci", ext="time.txt"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="8", n="8", k="8",
        variant="vector_intrinsic", dtype="f32", target="ci", ext="time.txt"
    ),
    target_ll_file(
        kernel="matmul_rowmaj", m="5", n="6", k="7",
        variant="transform_mlir", dtype="f32", target="ci", ext="time.txt"
    ),
]

# Functional coverage for the Python XSMM generators.
KERNELS_XSMM_AVX = [
    Kernel3D("matmul_rowmaj", 5, 34, 16),  # fully unrolled K, multiple M tiles
    Kernel3D("matmul_rowmaj", 29, 16, 16),  # fully unrolled K, multiple N blocks
    Kernel3D("matmul_rowmaj", 5, 34, 24),  # exact tiled K loop
    Kernel3D("matmul_rowmaj", 5, 34, 25),  # tiled K loop plus remainder
]

# For targets that can execute AVX instructions
TESTSET_AVX = [
    *expand(
        target_ll_file(
            kernel="matmul_rowmaj", m="3", n="16", k="5", dtype="f64",
            target=THIS_TARGET,
            ext="test.log",
        ),
        variant=[
            "transform_xdsl",
            "llvm_intrinsics",
            "libxsmm",
            "xdsl_libxsmm",
            "compxsmm",
            "mkl",
        ],
    ),
    # Exercise the Python generators across M/N blocking and all K-loop strategies.
    *expand(
        "build/"
        + THIS_TARGET
        + "/{case.kernel}/{case.m}x{case.n}x{case.k}/{variant}.f64.test.log",
        case=KERNELS_XSMM_AVX,
        variant=["xdsl_libxsmm", "compxsmm"],
    ),
]

TESTSET = {
    "neon": TESTSET_MAC,
    "ci": TESTSET_CI,
    "tower": TESTSET_CI + TESTSET_AVX,
    "pinocchio": TESTSET_CI + TESTSET_AVX,
}[THIS_TARGET]

rule tests:
    input: TESTSET
