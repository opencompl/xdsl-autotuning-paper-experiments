import pytest

from autotuner.direct_transpose_asm import generate_transpose_asm


@pytest.mark.parametrize("m", [8, 16])
@pytest.mark.parametrize("n", [2, 4])
def test_transpose_kernel_uses_contiguous_a_loads(m: int, n: int) -> None:
    assembly = generate_transpose_asm(m=m, n=n, k=64, strategy="single")
    blocks = m // 8

    assert assembly.count("vmovupd") == blocks * 64
    assert assembly.count("vunpck") == blocks * 8 * 8
    assert assembly.count("vshuff64x2") == blocks * 8 * 16
    assert assembly.count("vfmadd231pd") == blocks * n * 64
    assert assembly.count("vgatherqpd") == blocks * n
    assert assembly.count("vscatterqpd") == blocks * n


def test_transpose_multi_uses_four_accumulator_sets() -> None:
    assembly = generate_transpose_asm(m=16, n=4, k=64, strategy="multi")

    assert "accumulator_sets=4" in assembly
    assert "%zmm31" in assembly
    # 24 accumulator merges plus eight late C additions.
    assert assembly.count("vaddpd\t%zmm") == 32


@pytest.mark.parametrize(("m", "n", "k"), [(4, 2, 64), (8, 3, 64), (8, 4, 63)])
def test_transpose_kernel_rejects_unsupported_shapes(m: int, n: int, k: int) -> None:
    with pytest.raises(ValueError):
        generate_transpose_asm(m=m, n=n, k=k, strategy="single")
