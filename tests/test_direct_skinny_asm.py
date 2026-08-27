import pytest

from autotuner.direct_skinny_asm import generate_skinny_asm


@pytest.mark.parametrize(("n", "groups"), [(2, 16), (4, 32)])
def test_segmented_single_uses_full_zmm_lanes(n: int, groups: int) -> None:
    assembly = generate_skinny_asm(m=1, n=n, k=64, strategy="segmented-single")

    assert "accumulator_sets=1" in assembly
    assert assembly.count("vpermpd") == groups
    assert assembly.count("vmulpd") == 1
    assert assembly.count("vfmadd231pd") == groups - 1
    assert assembly.count("vmovdqa64") == 1


def test_segmented_multi_uses_independent_chains() -> None:
    assembly = generate_skinny_asm(m=1, n=2, k=64, strategy="segmented-multi")

    assert "accumulator_sets=4" in assembly
    assert assembly.count("vmulpd") == 4
    assert assembly.count("vfmadd231pd") == 12
    assert assembly.count("vaddpd\t%zmm") == 3


def test_segmented_n3_packs_two_k_rows_into_six_lanes() -> None:
    assembly = generate_skinny_asm(m=1, n=3, k=64, strategy="segmented-multi")

    assert "accumulator_sets=4" in assembly
    assert assembly.count("vfmadd231pd") == 28
    assert assembly.count("{%k1}{z}") == 32
    assert "{%k2}{z}" in assembly
    assert "{%k2}" in assembly
    assert ".Lreduce_indices" in assembly
    assert "1488(%rsi)" in assembly
    assert "1536(%rsi)" not in assembly


@pytest.mark.parametrize(("n", "register_name"), [(2, "xmm"), (4, "ymm")])
def test_outer_kernel_matches_output_width(n: int, register_name: str) -> None:
    assembly = generate_skinny_asm(m=1, n=n, k=64, strategy="outer-vl")

    assert "width-matched outer product" in assembly
    assert assembly.count(f"vmovupd\t%{register_name}") == 1
    assert f"%{register_name}0" in assembly
    assert f"{{1to{n}}}" in assembly
    assert "{%k" not in assembly


@pytest.mark.parametrize("strategy", ["segmented-single", "outer-vl"])
def test_skinny_kernel_rejects_other_n(strategy: str) -> None:
    with pytest.raises(ValueError):
        generate_skinny_asm(m=1, n=5, k=64, strategy=strategy)  # type: ignore[arg-type]
