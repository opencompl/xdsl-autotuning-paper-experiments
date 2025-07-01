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
    # Generate CI test set x86 assembly
    *(f"{base}/naive_c.x86.S" for base in _TESTSET_CI),
]

rule test_mac:
    input: TESTSET_MAC
    output: "build/test_mac.txt"
    shell: 'echo "tests passed" > {output}'


TESTSET_DOCKER = [
    # Validate CI test set x86 executables
    *(f"{base}/naive_c.neon.S" for base in _TESTSET_CI),
    # Generate CI test set neon assembly
    *(f"{base}/naive_c.x86.log" for base in _TESTSET_CI),
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

rule asm:
    input: "kernels/{kernel}/{variant}.c"
    output: "build/{kernel}/{m}x{n}x{k}/{variant}.{target}.S"
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
