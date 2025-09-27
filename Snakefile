configfile: "default.yaml"

import os

########################################################################################
# Build
########################################################################################

# Target-specific parameters

T = config["targets"]
if os.environ.get("USE_PAPI") == "1":
    T['ci']['libs'].append('papi')

def target_triple(wildcards):
    return T[wildcards.target]['triple']

def target_freq(wildcards):
    return T[wildcards.target]['freq']

def target_arch(wildcards):
    return T[wildcards.target]['arch']

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

def target_env(wildcards):
    return ' '.join([f"{k}={v}" for k,v in T[wildcards.target]['env'].items()])

def target_xsmm(wildcards):
    arch = target_arch(wildcards)
    return config['xsmm_map'][arch]

def target_libs_opts(wildcards):
    return " ".join(f"-l{x}" for x in T[wildcards.target]['libs'])

# Path management

def target_base(
        kernel = "{kernel}",
        m = "{m}",
        n = "{n}",
        k = "{k}"
):
    return f"build/{kernel}/{m}x{n}x{k}"

def target_variant(
        ext,
        variant = "{variant}",
        dtype = "{dtype}"
):
    return f"{variant}.{dtype}.{ext}"

def target_file(
    ext,
    kernel = "{kernel}",
    m = "{m}",
    n = "{n}",
    k = "{k}",
    variant = "{variant}",
    dtype = "{dtype}",
):
    base = target_base(kernel=kernel,m=m,n=n,k=k)
    var = target_variant(variant=variant,dtype=dtype,ext=ext)
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
    base = target_base(kernel=kernel,m=m,n=n,k=k)
    var = target_variant(variant=variant,dtype=dtype,ext=f"{target}.{ext}")
    return f"{base}/{var}"

# Rules

wildcard_constraints:
    dtype = "f32|f64",
    kernel="matmul_(rowmaj|colmaj)",
    variant="naive_c|naive_mlir|vector_intrinsic|transform_mlir|transform_xdsl|libxsmm"

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
        program = target_file(variant='memref',ext='mlir'),
    output: target_file(variant='transform_xdsl',ext='vector.mlir')
    shell:
        """xdsl-opt -p vectorize-libxsmm {input.program} -o {output}"""

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
        cc=config["cc"],
    shell:
        "{params.cc} -O3 -S -fenable-matrix -target {params.target_triple} -march={params.target_arch} -o {output} {input}"

rule asm_c:
    input: "kernels/{kernel}/naive_c.c"
    output: target_ll_file(variant='naive_c',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
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
    input: target_ll_file(variant='libxsmm',ext='c')
    output: target_ll_file(variant='libxsmm',ext='S')
    params:
        target_triple=target_triple,
        target_arch=target_arch,
        cc=config["cc"],
    shell:
        "{params.cc} -O3 -DNDEBUG -c {input} -S -target {params.target_triple} -march={params.target_arch} -o {output}"

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
        use_papi=lambda wildcards: "-DUSE_PAPI=1" if os.environ.get("USE_PAPI") == "1" else "",
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -DDTYPE={params.dtype} -DFREQ={params.target_freq} {params.use_papi} -target {params.target_triple} -march={params.target_arch} -o {output} kernels/{wildcards.kernel}/{wildcards.executable}.c {input} {params.target_libs_opts}"

rule validation:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.test.o"
    # A log won't be deleted by Snakemake if the script fails
    log: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.test.log"
    shell: '{input} > {log}'

########################################################################################
# Time
########################################################################################

rule time:
    input: target_ll_file(ext="time.o")
    output: target_ll_file(ext="time.txt")
    params: target_env=target_env,
    shell: '{params.target_env} {input} > {output}'

rule flops:
    input: "kernels/{kernel}/flops.sh"
    output: "build/{kernel}/{m}x{n}x{k}/flops.txt"
    shell: "./{input} {wildcards.m} {wildcards.n} {wildcards.k} > {output}"

rule json:
    input:
        time_txt=target_ll_file(ext="time.txt"),
        flops_txt="build/{kernel}/{m}x{n}x{k}/flops.txt"
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
        "cube_8.f64": ["naive_c", "transform_mlir"],
        "cube_16.f64": ["naive_c", "transform_mlir"],
        "cube_64.f64": ["naive_c", "transform_mlir"],
    },
    "tower": {
        "ttile": ["naive_c", "libxsmm"],
        "cube_8.f64": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm", "transform_xdsl"],
        "cube_16.f64": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm"],
        "cube_64.f64": ["naive_c", "transform_mlir", "libxsmm"],
    },
    "pinocchio": {
        "ttile": ["naive_c", "libxsmm"],
        "cube_8.f64": ["naive_c", "transform_mlir", "vector_intrinsic", "libxsmm","transform_xdsl"],
        "cube_16.f64": ["naive_c", "transform_mlir", "libxsmm"],
        "cube_64.f64": ["naive_c", "transform_mlir", "libxsmm"],
    },
    "ci": {
        "ttile": ["naive_c"],
        "cube_8.f64": ["naive_c", "transform_mlir"],
        "cube_16.f64": ["naive_c", "transform_mlir"],
        "cube_64.f64": ["naive_c", "transform_mlir"],
    },
}[THIS_TARGET]

DATASET_BASES = {
    "ttile.f32": expand(
        target_file(kernel="matmul_rowmaj",n="128",k="128",dtype="f32",ext=THIS_TARGET),
        variant=DATASET_VARIANTS["ttile"],
        m=range(8, 50, 2),
    ),
    "cube_8.f64": expand(
        target_file(kernel="matmul_rowmaj",m="8",n="8",k="8",dtype="f64",ext=THIS_TARGET),
        variant=DATASET_VARIANTS["cube_8.f64"],
    ),
    "cube_16.f64": expand(
        target_file(kernel="matmul_rowmaj",m="16",n="16",k="16",dtype="f64",ext=THIS_TARGET),
        variant=DATASET_VARIANTS["cube_16.f64"],
    ),
    "cube_64.f64": expand(
        target_file(kernel="matmul_rowmaj",m="64",n="64",k="64",dtype="f64",ext=THIS_TARGET),
        variant=DATASET_VARIANTS["cube_64.f64"],
    ),
}

BARS_INPUTS = expand(
    "{base}/" + target_variant(dtype="f64",ext="json"),
    base=target_base(kernel="matmul_rowmaj"),
    variant=DATASET_VARIANTS["cube_8.f64"]
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
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="transform_mlir",dtype="f32",ext="neon.test.log"
    ),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="transform_mlir",dtype="f32",ext="neon.time.txt"
    ),
    # Generate CI test set x86 assembly
    *(f"{base}.ci.S" for base in _TESTSET_CI),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="transform_mlir",dtype="f32",ext="ci.S"
    ),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="vector_intrinsic",dtype="f32",ext="ci.S"
    ),
    target_file(
        kernel="matmul_rowmaj",m="4",n="4",k="4",
        variant="transform_xdsl",dtype="f64",ext="tower.S"
    ),
]

TESTSET_CI = [
    # Generate CI test set neon assembly
    *(f"{base}.neon.S" for base in _TESTSET_CI),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="transform_mlir",dtype="f32",ext="neon.S"
    ),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="vector_intrinsic",dtype="f32",ext="neon.S"
    ),
    target_file(
        kernel="matmul_rowmaj",m="4",n="4",k="4",
        variant="transform_xdsl",dtype="f64",ext="tower.S"
    ),
    # Validate CI test set x86 executables
    *(f"{base}.ci.test.log" for base in _TESTSET_CI),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="transform_mlir",dtype="f32",ext="ci.test.log"
    ),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="transform_mlir",dtype="f32",ext="ci.time.txt"
    ),
    target_file(
        kernel="matmul_rowmaj",m="8",n="8",k="8",
        variant="vector_intrinsic",dtype="f32",ext="ci.time.txt"
    ),
    target_file(
        kernel="matmul_rowmaj",m="5",n="6",k="7",
        variant="transform_mlir",dtype="f32",ext="ci.time.txt"
    ),
]

# For targets that can execute AVX instructions
TESTSET_AVX = [
    target_file(
        kernel="matmul_rowmaj",m="3",n="16",k="5",
        variant="transform_xdsl",dtype="f64",ext="ci.test.log"
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
