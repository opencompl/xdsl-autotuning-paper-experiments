configfile: "default.yaml"

########################################################################################
# Build
########################################################################################

# clang -dumpmachine
TARGET_TRIPLE_DICT = {
    "neon": "arm64-apple-darwin24.3.0 ", # Sasha's Mac
    "x86": "x86_64-unknown-linux-gnu", # Docker
}

def target_triple(wildcards):
    return TARGET_TRIPLE_DICT[wildcards.target]


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
        transform="kernels/{kernel}/vectorize.transform.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.{dtype}.mlir"
    shell:
        './src/merge_transform.awk {input.matmul} {input.transform} > {output}'

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
            --convert-vector-to-scf \
            --convert-scf-to-cf \
            -o {output}"""

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
        cc=config["cc"],
    shell:
        "{params.cc} -S -fenable-matrix -target {params.target_triple} -o {output} {input}"

rule asm_c:
    input: "kernels/{kernel}/naive_c.c"
    output: "build/{kernel}/{m}x{n}x{k}/naive_c.{dtype}.{target}.S"
    params:
        target_triple=target_triple,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -S -target {params.target_triple} -o {output} {input}"

rule libxsmm_colmaj_c:
    output: "build/matmul_colmaj/{m}x{n}x{k}/libxsmm.x86.c"
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        libxsmm_gemm_generator dense {output} matmul_colmaj_abc \
            {wildcards.m} {wildcards.n} {wildcards.k} \
            {wildcards.m} {wildcards.k} {wildcards.m} \
            1 1 1 1 hsw nopf SP && \
        echo 'void matmul_colmaj(float *C, const float *A, const float *B) {{matmul_colmaj_abc(A, B, C);}}' >> {output}
        """

rule libxsmm_rowmaj_c:
    output: "build/matmul_rowmaj/{m}x{n}x{k}/libxsmm.x86.c"
    shell:
        """
        # A = M * K, B = K * N, C = M * N    <- dimensions
        #     ^          ^          ^        <- leading dimensions
        libxsmm_gemm_generator dense {output} matmul_bac \
            {wildcards.n} {wildcards.m} {wildcards.k} \
            {wildcards.n} {wildcards.k} {wildcards.n} \
            1 1 1 1 hsw nopf SP && \
        echo 'void matmul(float *C, const float *A, const float *B) {{matmul_bac(B, A, C);}}' >> {output}
        """


rule libxsmm_s:
    input: "build/{kernel}/{m}x{n}x{k}/libxsmm.{target}.c"
    output: "build/{kernel}/{m}x{n}x{k}/libxsmm.{target}.S"
    params:
        target_triple=target_triple,
        cc=config["cc"],
    shell:
        "{params.cc} -mavx2 -DNDEBUG -c {input} -S -target {params.target_triple} -o {output}"

rule executable:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.S"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{dtype}.{target}.{executable}.o"
    params:
        target_triple=target_triple,
        cc=config["cc"],
        dtype=lambda wildcards: {"f32": "float", "f64": "double"}[wildcards.dtype],
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -target {params.target_triple} -o {output} kernels/{wildcards.kernel}/{wildcards.executable}.c {input}"

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
    shell:
        """
        M={wildcards.m}
        N={wildcards.n}
        K={wildcards.k}
        FLOPS=$(head -n 1 {input.flops_txt} | tr -d '[:space:]')
        TIME=$(head -n 1 {input.time_txt} | tr -d '[:space:]')
        VARIANT="{wildcards.variant}"
        TARGET="{wildcards.target}"
        DTYPE="{wildcards.dtype}"
        echo '{{"M":'${{M}}',"N":'${{N}}',"K":'${{K}}',"flops":'${{FLOPS}}',"time":'${{TIME}}',"variant":"'${{VARIANT}}'","target":"'${{TARGET}}'","dtype":"'${{DTYPE}}'"}}' > {output.json}
        """

########################################################################################
# Dataset
########################################################################################

import platform
# "Darwin" for macOS, "Linux" for Linux, etc.
THIS_SYSTEM = platform.system()

# NOTE: we should make this more precise in the future
THIS_TARGET = {
    "Darwin": "neon",
    "Linux": "x86"
}[THIS_SYSTEM]

DATASET_VARIANTS = {
    "neon": {
        "ttile": ["naive_c"],
        "cube.f32": ["naive_c", "transform_mlir", "vector_intrinsic"],
        "cube.f64": ["naive_c", "transform_mlir", "vector_intrinsic"],
    },
    "x86": {
        "ttile": ["naive_c", "libxsmm"],
        "cube.f32": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm"],
        "cube.f64": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm"],
    },
}[THIS_TARGET]

DATASET_BASES = {
    "ttile.f32": expand(
        "build/matmul_rowmaj/{m}x128x128/{variant}.f32." + THIS_TARGET,
        variant=DATASET_VARIANTS["ttile"],
        m=range(8, 50, 2),
    ),
    "cube.f32": expand(
        "build/matmul_rowmaj/8x8x8/{variant}.f32." + THIS_TARGET,
        variant=DATASET_VARIANTS["cube.f32"],
    ),
    "cube.f64": expand(
        "build/matmul_rowmaj/4x4x4/{variant}.f64." + THIS_TARGET,
        variant=DATASET_VARIANTS["cube.f64"],
    ),
}

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
    *(f"{base}.x86.S" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.x86.S",
    f"build/matmul_rowmaj/8x8x8/vector_intrinsic.f32.x86.S",
]

TESTSET_DOCKER = [
    # Validate CI test set x86 executables
    *(f"{base}.neon.S" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.neon.S",
    f"build/matmul_rowmaj/8x8x8/vector_intrinsic.f32.neon.S",
    # Generate CI test set neon assembly
    *(f"{base}.x86.test.log" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.x86.test.log",
    f"build/matmul_rowmaj/8x8x8/transform_mlir.f32.x86.time.txt",
    f"build/matmul_rowmaj/8x8x8/vector_intrinsic.f32.x86.time.txt",
    f"build/matmul_rowmaj/5x6x7/vector_intrinsic.f32.x86.time.txt",
]

TESTSET = {
    "neon": TESTSET_MAC,
    "x86": TESTSET_DOCKER,
}[THIS_TARGET]

rule tests:
    input: TESTSET
