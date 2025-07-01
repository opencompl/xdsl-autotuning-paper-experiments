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

_TESTSET_CI = [
    *expand("build/{k.kernel}/{k.m}x{k.n}x{k.k}", k=KERNELS_CI)
]

TESTSET_MAC = [
    # Validate CI test set neon executables
    *(f"{base}/naive_c.neon.log" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.neon.log" for base in _TESTSET_CI),
    # Generate CI test set x86 assembly
    *(f"{base}/naive_c.x86.S" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.x86.S" for base in _TESTSET_CI),
]

rule test_mac:
    input: TESTSET_MAC
    output: "build/test_mac.txt"
    shell: 'echo "tests passed" > {output}'


TESTSET_DOCKER = [
    # Validate CI test set x86 executables
    *(f"{base}/naive_c.neon.S" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.neon.S" for base in _TESTSET_CI),
    # Generate CI test set neon assembly
    *(f"{base}/naive_c.x86.log" for base in _TESTSET_CI),
    *(f"{base}/naive_mlir.x86.log" for base in _TESTSET_CI),
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

rule templated:
    input: "kernels/{kernel}/mlir.mlir"
    output: "build/{kernel}/{m}x{n}x{k}/mlir.mlir"
    shell:
        # Use awk to substitute {{M}} for m and so on
        # Use {{ to otuput a single { when executing command
        "awk '{{gsub(/{{{{M}}}}/, \"{wildcards.m}\"); gsub(/{{{{N}}}}/, \"{wildcards.n}\"); gsub(/{{{{K}}}}/, \"{wildcards.k}\")}} 1' {input} > {output}"

rule naive_mlir:
    input: "build/{kernel}/{m}x{n}x{k}/mlir.mlir"
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
        "{params.cc} -S -target {params.target_triple} -o {output} {input}"

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
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.o"
    params:
        target_triple=target_triple,
        cc=config["cc"],
    shell:
        "{params.cc} -DCROWS={wildcards.m} -DCCOLS={wildcards.n} -DINNER={wildcards.k} -target {params.target_triple} -o {output} kernels/{wildcards.kernel}/main.c {input}"

rule validation:
    input: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.o"
    # A log won't be deleted by Snakemake if the script fails
    log: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.log"
    params:
        target_triple=target_triple
    shell: '{input} > {log}'
