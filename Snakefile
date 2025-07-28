configfile: "default.yaml"

from typing import NamedTuple

class Kernel3D(NamedTuple):
    kernel: str
    m: int
    n: int
    k: int

KERNELS_CI = [
    Kernel3D("matmul_rowmaj", 4, 4, 4),
    Kernel3D("matmul_rowmaj", 5, 6, 7),
]

########################################################################################

def target_dataset(wildcards):
    variants = {
        "neon": ["naive_c"],
        "x86": ["naive_c"],
    }
    sets = {
        "ttile": [
            *expand(
                f"build/matmul_rowmaj/{{m}}x128x128/{{variant}}.{wildcards.target}.json",
                m=range(8, 50, 2),
                variant=variants[wildcards.target],
            )
        ],
    }
    name = wildcards.testset
    if name not in sets:
        raise ValueError(
            f"unknown test set name '{name}', valid values are: {sets.keys()}"
        )
    return sets[name]


########################################################################################

_TESTSET_CI = [
    *expand("build/{k.kernel}/{k.m}x{k.n}x{k.k}", k=KERNELS_CI)
]

TESTSET_MAC = [
    # Validate CI test set neon executables
    *(f"{base}/naive_c.neon.test.log" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.neon.test.log" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/4x4x4/transform_mlir.neon.test.log",
    f"build/matmul_rowmaj/4x4x4/transform_mlir.neon.time.txt",
    # Generate CI test set x86 assembly
    *(f"{base}/naive_c.x86.S" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.x86.S" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/4x4x4/transform_mlir.x86.S",
]

rule test_mac:
    input: TESTSET_MAC
    output: "build/test_mac.txt"
    shell: 'echo "tests passed" > {output}'


TESTSET_DOCKER = [
    # Validate CI test set x86 executables
    *(f"{base}/naive_c.neon.S" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.neon.S" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/4x4x4/transform_mlir.neon.S",
    # Generate CI test set neon assembly
    *(f"{base}/naive_c.x86.test.log" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.x86.test.log" for base in _TESTSET_CI),
    f"build/matmul_rowmaj/4x4x4/transform_mlir.x86.test.log",
    f"build/matmul_rowmaj/4x4x4/transform_mlir.x86.time.txt",
]

rule test_docker:
    input: TESTSET_DOCKER
    output: "build/test_docker.txt"
    shell: 'echo "tests passed" > {output}'

########################################################################################

# clang -dumpmachine
TARGET_TRIPLE_DICT = {
    "neon": "arm64-apple-darwin24.3.0 ", # Sasha's Mac
    "x86": "x86_64-unknown-linux-gnu", # Docker
}

def target_triple(wildcards):
    return TARGET_TRIPLE_DICT[wildcards.target]

########################################################################################

rule templated_tensor:
    input: "kernels/{kernel}/mlir.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/tensor.mlir"
    shell:
        # Use awk to substitute {{M}} for m and so on
        # Use {{ to otuput a single { when executing command
        "awk '{{gsub(/{{{{M}}}}/, \"{wildcards.m}\"); gsub(/{{{{N}}}}/, \"{wildcards.n}\"); gsub(/{{{{K}}}}/, \"{wildcards.k}\")}} 1' {input} | mlir-opt > {output}"

rule templated_vector_intrinsic:
    input: "kernels/{kernel}/vector_intrinsic.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/vector_intrinsic.arith.mlir"
    shell:
        # Use awk to substitute {{M}} for m and so on
        # Use {{ to otuput a single { when executing command
        "awk '{{gsub(/{{{{M}}}}/, \"{wildcards.m}\"); gsub(/{{{{N}}}}/, \"{wildcards.n}\"); gsub(/{{{{K}}}}/, \"{wildcards.k}\")}} 1' {input} | mlir-opt > {output}"

rule transform_mlir:
    input:
        matmul="build/{kernel}/{m}x{n}x{k}/memref.mlir",
        transform="kernels/{kernel}/{m}x{n}x{k}/transform.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.mlir"
    shell:
        './src/merge_transform.awk {input.matmul} {input.transform} > {output}'

rule execute_transform:
    input: "build/{kernel}/{m}x{n}x{k}/transform_mlir.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.vector.mlir"
    shell:
        """mlir-opt {input} \
            --transform-interpreter \
            --mlir-print-op-generic \
        | xdsl-opt \
            -p test-transform-dialect-erase-schedule \
            --allow-unregistered-dialect \
            -o {output}"""

rule vector_to_arith:
    input: "build/{kernel}/{m}x{n}x{k}/transform_mlir.vector.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/transform_mlir.arith.mlir"
    shell:
        """mlir-opt {input} \
            --convert-vector-to-scf \
            --convert-scf-to-cf \
            -o {output}"""

rule memref_mlir:
    input: "build/{kernel}/{m}x{n}x{k}/tensor.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/memref.mlir"
    shell:
        """mlir-opt {input} \
            --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
            -o {output}"""

rule naive_mlir:
    input: "build/{kernel}/{m}x{n}x{k}/memref.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/naive_mlir.arith.mlir"
    shell:
        """mlir-opt {input} \
            --convert-linalg-to-loops \
            --convert-scf-to-cf \
            --buffer-results-to-out-params \
            -o {output}"""

rule arith_to_llvm:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.arith.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.llvm.mlir"
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
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.llvm.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.ll"
    shell:
        "mlir-translate --mlir-to-llvmir {input} -o {output}"

rule asm_ll:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.ll"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.S"
    params:
        target_triple=target_triple,
        cc=config["cc"],
    shell:
        "{params.cc} -S -fenable-matrix -target {params.target_triple} -o {output} {input}"

rule asm_c:
    input: "kernels/{kernel}/naive_c.c"
    output: "build/{kernel}/{m}x{n}x{k}/naive_c.{target}.S"
    params:
        target_triple=target_triple,
        cc=config["cc"],
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -S -target {params.target_triple} -o {output} {input}"

rule executable:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.S"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.{executable}.o"
    params:
        target_triple=target_triple,
        cc=config["cc"],
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -target {params.target_triple} -o {output} kernels/{wildcards.kernel}/{wildcards.executable}.c {input}"

rule validation:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.test.o"
    # A log won't be deleted by Snakemake if the script fails
    log: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.test.log"
    shell: '{input} > {log}'

rule time:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.time.o"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.time.txt"
    params:
        target_triple=target_triple
    shell: '{input} > {output}'

rule flops:
    input: "kernels/{kernel}/flops.sh"
    output: "build/{kernel}/{m}x{n}x{k}/flops.txt"
    shell: "./{input} {wildcards.m} {wildcards.n} {wildcards.k} > {output}"

rule json:
    input:
        time_txt="build/{kernel}/{m}x{n}x{k}/{variant}.{target}.time.txt",
        flops_txt="build/{kernel}/{m}x{n}x{k}/flops.txt"
    output:
        json="build/{kernel}/{m}x{n}x{k}/{variant}.{target}.json"
    shell:
        """
        M={wildcards.m}
        N={wildcards.n}
        K={wildcards.k}
        FLOPS=$(head -n 1 {input.flops_txt} | tr -d '[:space:]')
        TIME=$(head -n 1 {input.time_txt} | tr -d '[:space:]')
        VARIANT="{wildcards.variant}"
        TARGET="{wildcards.target}"
        echo '{{"M":'${{M}}',"N":'${{N}}',"K":'${{K}}',"flops":'${{FLOPS}}',"time":'${{TIME}}',"variant":"'${{VARIANT}}'","target":"'${{TARGET}}'"}}' > {output.json}
        """

rule target_dataset:
    input: target_dataset
    output: "data/{testset}.{target}.jsonl"
    shell: "cat {input} > {output}"

rule target_plot:
    input: "data/{testset}.{target}.jsonl"
    output: "plots/{testset}.{target}.png"
    shell:
        "plot-{wildcards.testset} {input} --output {output}"
