configfile: "default.yaml"

import os

########################################################################################
# Build
########################################################################################

# clang -dumpmachine
TARGET_TRIPLE_DICT = {
    "neon": "arm64-apple-darwin24.3.0 ", # Sasha's Mac
    "ci": "x86_64-unknown-linux-gnu", # Docker
    "tower": "x86_64-pc-linux-gnu",
    "pinocchio": "x86_64-pc-linux-gnu",
}

TARGET_FREQ_DICT = {
    "neon": 0.0,
    "ci": 0.0,
    "tower": 0.0,
    "pinocchio": 2.1
}

TARGET_ARCH_DICT = {
    "neon": "armv8.5-a",
    "ci": "x86-64", # TODO
    "tower": "znver5",
    "pinocchio": "cascadelake",
}

TARGET_PEAK_F32_DICT = {
    "neon": 0,
    "ci": 0,
    "tower": 0,
    "pinocchio": 64
}

def arch_to_xsmm(arch):
    match arch:
        case 'cascadelake':
            return 'clx'
        case 'znver5':
            return 'skx'
        case _:
            return 'noarch'

TARGET_XSMM_DICT = { k: arch_to_xsmm(v) for k, v in TARGET_ARCH_DICT.items() }

TARGET_LIBS_DICT = {
    "neon": [],
    "ci": [],
    "tower": ['papi'],
    "pinocchio": ['papi'],
}



if os.environ.get("INSIDE_DOCKER") == "1":
    TARGET_LIBS_DICT["ci"].append('papi')

def target_libs_opts(wildcards):
    return " ".join(f"-l{x}" for x in TARGET_LIBS_DICT[wildcards.target])

def target_freq(wildcards):
    return TARGET_FREQ_DICT[wildcards.target]

def target_triple(wildcards):
    return TARGET_TRIPLE_DICT[wildcards.target]

def target_peak_flops(wildcards):
    match wildcards.dtype:
        case 'f32':
            factor = 1.0
        case 'f64':
            factor = 2.0
        case _:
            assert False
    flops_per_cycle = TARGET_PEAK_F32_DICT[wildcards.target] / factor
    return flops_per_cycle

def target_arch(wildcards):
    return TARGET_ARCH_DICT[wildcards.target]

def target_xsmm(wildcards):
    return TARGET_XSMM_DICT[wildcards.target]


rule templated_tensor:
    input: "kernels/{kernel}/mlir.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/tensor.{dtype}.mlir"
    wildcard_constraints:
        dtype="f32|f64",
    template_engine:
        "jinja2"

rule templated_vector_intrinsic:
    input: "kernels/{kernel}/vector_intrinsic.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/vector_intrinsic.{dtype}.arith.mlir"
    template_engine:
        "jinja2"

rule transform_mlir:
    input:
        matmul="build/{kernel}/{m}x{n}x{k}/memref.{dtype}.mlir",
        transform="kernels/vectorize.transform.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.{dtype}.mlir"
    shell:
        './src/autotuner/merge_transform.awk {input.matmul} {input.transform} > {output}'

rule execute_transform:
    input: "build/{kernel}/{m}x{n}x{k}/transform_mlir.{dtype}.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.{dtype}.vector.mlir"
    shell:
        """mlir-opt {input} \
            --transform-interpreter \
            --mlir-print-op-generic \
        | xdsl-opt \
            -p test-transform-dialect-erase-schedule \
            --allow-unregistered-dialect \
            -o {output}"""

rule vector_to_arith:
    input: "build/{kernel}/{m}x{n}x{k}/transform_mlir.{dtype}.vector.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.{dtype}.arith.mlir"
    shell:
        """mlir-opt {input} \
            --canonicalize \
            --lower-affine \
            --convert-vector-to-scf \
            --convert-scf-to-cf \
            -o {output}"""

rule transform_xdsl:
    input: "build/{kernel}/{m}x{n}x{k}/memref.{dtype}.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_xdsl.{dtype}.tower.S"
    params:
        passes = ",".join(config["xdsl-opt-passes-vector"])
    shell:
        """xdsl-opt -p {params.passes} -t x86-asm {input} -o {output}"""

rule memref_mlir:
    input: "build/{kernel}/{m}x{n}x{k}/tensor.{dtype}.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/memref.{dtype}.mlir"
    shell:
        """mlir-opt {input} \
            --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
            -o {output}"""

rule naive_mlir:
    input: "build/{kernel}/{m}x{n}x{k}/memref.{dtype}.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/naive_mlir.{dtype}.arith.mlir"
    shell:
        """mlir-opt {input} \
            --convert-linalg-to-loops \
            --convert-scf-to-cf \
            --buffer-results-to-out-params \
            -o {output}"""

rule arith_to_llvm:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.arith.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.llvm.mlir"
    shell:
        """mlir-opt {input} \
            --convert-func-to-llvm=use-bare-ptr-memref-call-conv \
            --finalize-memref-to-llvm \
            --canonicalize --cse --sccp \
            --convert-vector-to-llvm=enable-x86vector \
            --convert-index-to-llvm \
            --convert-arith-to-llvm \
            --convert-cf-to-llvm \
            --canonicalize --cse --sccp \
            -o {output}"""

rule mlir_to_ll:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.llvm.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.ll"
    shell:
        "mlir-translate --mlir-to-llvmir {input} -o {output}"

rule asm_ll:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.ll"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.S"
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
    shell:
        "{params.cc} -O3 -S -fenable-matrix -target {params.target_triple} -march={params.target_arch} -o {output} {input}"

rule asm_c:
    input: "kernels/{kernel}/naive_c.c"
    output: "build/{kernel}/{m}x{n}x{k}/naive_c.{dtype}.{target}.S"
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -O3 -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -S -target {params.target_triple} -march={params.target_arch} -o {output} {input}"

rule libxsmm_colmaj_c:
    output: "build/matmul_colmaj/{m}x{n}x{k}/libxsmm.{dtype}.{target}.c"
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
    output: "build/matmul_rowmaj/{m}x{n}x{k}/libxsmm.{dtype}.{target}.c"
    params:
        target_xsmm=target_xsmm,
        dtype=lambda wildcards: {"f32": "SP", "f64": "DP"}[wildcards.dtype],
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        libxsmm_gemm_generator dense {output} matmul_bac \
            {wildcards.m} {wildcards.n} {wildcards.k} \
            {wildcards.m} {wildcards.k} {wildcards.m} \
            1 1 \
            1 1 \
            {params.target_xsmm} \
            nopf \
            {params.dtype} && \
        echo 'void matmul(const float *A, const float *B, float *C) {{matmul_bac(B, A, C);}}' >> {output}
        """


rule libxsmm_s:
    input: "build/{kernel}/{m}x{n}x{k}/libxsmm.{dtype}.{target}.c"
    output: "build/{kernel}/{m}x{n}x{k}/libxsmm.{dtype}.{target}.S"
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
    shell:
        "{params.cc} -O3 -mavx2 -DNDEBUG -c {input} -S -target {params.target_triple} -march={params.target_arch} -o {output}"

rule executable:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.S"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.{executable}.o"
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        target_libs_opts=target_libs_opts,
        target_freq=target_freq,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -DFREQ={params.target_freq} -target {params.target_triple} -march={params.target_arch} -o {output} kernels/{wildcards.kernel}/{wildcards.executable}.c {input} {params.target_libs_opts}"

rule validation:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.test.o"
    # A log won't be deleted by Snakemake if the script fails
    log: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.test.log"
    shell: '{input} > {log}'

########################################################################################
# Time
########################################################################################

rule time:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.time.o"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.time.txt"
    shell: '{input} > {output}'

rule flops:
    input: "kernels/{kernel}/flops.sh"
    output: "build/{kernel}/{m}x{n}x{k}/flops.txt"
    shell: "./{input} {wildcards.m} {wildcards.n} {wildcards.k} > {output}"

rule json:
    input:
        time_txt="build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.time.txt",
        flops_txt="build/{kernel}/{m}x{n}x{k}/flops.txt"
    output:
        json="build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.json"
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
        "cube.f32": ["naive_c", "transform_mlir", "vector_intrinsic"],
        "cube.f64": ["naive_c", "transform_mlir", "vector_intrinsic"],
    },
    "tower": {
        "ttile": ["naive_c", "libxsmm"],
        "cube.f32": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm"],
        "cube.f64": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm", "transform_xdsl"],
    },
    "pinocchio": {
        "ttile": ["naive_c", "libxsmm"],
        "cube.f32": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm"],
        "cube.f64": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm"],
    },
    "ci": {
        "ttile": ["naive_c"],
        "cube.f32": ["naive_c", "transform_mlir", "vector_intrinsic"],
        "cube.f64": ["naive_c", "transform_mlir", "vector_intrinsic"],
    },
}[THIS_TARGET]

DATASET_BASES = {
    "ttile.f32": expand(
        "build/matmul_rowmaj/{m}x128x128/{variant}.f32." + THIS_TARGET,
        variant=DATASET_VARIANTS["ttile"],
        m=range(8, 50, 2),
    ),
    "cube.f32": expand(
        "build/matmul_rowmaj/32x32x32/{variant}.f32." + THIS_TARGET,
        variant=DATASET_VARIANTS["cube.f32"],
    ),
    "cube.f64": expand(
        "build/matmul_rowmaj/32x32x32/{variant}.f64." + THIS_TARGET,
        variant=DATASET_VARIANTS["cube.f64"],
    ),
}

BARS_INPUTS = expand(
    "{base}/{variant}.f64." + THIS_TARGET + ".json",
    base="build/matmul_rowmaj/{m}x{n}x{k}",
    variant=DATASET_VARIANTS["cube.f64"]
)

rule bars_data:
    input: BARS_INPUTS
    output: "data/bars.{m}x{n}x{k}.f64." + THIS_TARGET + ".jsonl"
    shell: "cat {input} > {output}"

for dataset, samples in DATASET_BASES.items():
    rule:
        input: [base + ".json" for base in samples]
        output: f"data/{dataset}.{THIS_TARGET}.jsonl"
        shell: "cat {input} > {output}"

rule dataset_code:
    input: [p + ".time.o" for p in flatten(DATASET_BASES.values())]

rule dataset:
    input:
        expand(
            "data/{dataset_base}.{target}.jsonl",
            target=THIS_TARGET,
            dataset_base=DATASET_BASES,
        )

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

_TESTSET_CI = expand(
    "build/{k.kernel}/{k.m}x{k.n}x{k.k}/{variant}.{dtype}",
    k=KERNELS_CI,
    variant=VARIANT_CI,
    dtype=["f32", "f64"],
)

TESTSET_MAC = [
    # Validate CI test set neon executables
    *(f"{base}.neon.test.log" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.neon.test.log",
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.neon.time.txt",
    # Generate CI test set x86 assembly
    *(f"{base}.ci.S" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.ci.S",
    f"build/matmul_rowmaj/8x8x8/vector_intrinsic.f32.ci.S",
    f"build/matmul_rowmaj/4x4x4/transform_xdsl.f64.tower.S",
]

TESTSET_CI = [
    # Generate CI test set neon assembly
    *(f"{base}.neon.S" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.neon.S",
    f"build/matmul_rowmaj/8x8x8/vector_intrinsic.f32.neon.S",
    # Generate CI test set avx assembly
    f"build/matmul_rowmaj/4x4x4/transform_xdsl.f64.tower.S",
    # Validate CI test set x86 executables
    *(f"{base}.ci.test.log" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.ci.test.log",
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.ci.time.txt",
    f"build/matmul_rowmaj/8x8x8/vector_intrinsic.f32.ci.time.txt",
    f"build/matmul_rowmaj/5x6x7/vector_intrinsic.f32.ci.time.txt",
]

TESTSET = {
    "neon": TESTSET_MAC,
    "ci": TESTSET_CI,
    "tower": TESTSET_CI,
    "pinocchio": TESTSET_CI,
}[THIS_TARGET]

rule tests:
    input: TESTSET
