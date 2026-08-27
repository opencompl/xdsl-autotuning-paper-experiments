import pytest

from autotuner.direct_gather_asm import generate_gather_asm


@pytest.mark.parametrize("m", [8, 16])
@pytest.mark.parametrize("n", [2, 4])
def test_gather_kernel_has_one_a_gather_per_m_block_and_k(m: int, n: int) -> None:
    assembly = generate_gather_asm(m=m, n=n, k=64, strategy="single")

    # Include one gather of C for every output column and M block.
    assert assembly.count("vgatherqpd") == (m // 8) * (64 + n)
    assert assembly.count("vscatterqpd") == (m // 8) * n
    assert assembly.count("vfmadd231pd") == (m // 8) * n * 64
    assert "accumulator_sets=1" in assembly
    assert "gather_streams=1" in assembly


@pytest.mark.parametrize(
    ("m", "n", "sets"), [(8, 2, 4), (8, 4, 4), (16, 2, 4), (16, 4, 3)]
)
def test_gather_multi_uses_available_accumulator_registers(
    m: int, n: int, sets: int
) -> None:
    assembly = generate_gather_asm(m=m, n=n, k=64, strategy="multi")

    assert f"accumulator_sets={sets}" in assembly
    assert f"%zmm{(m // 8) * n * sets}" in assembly
    assert "gather_streams=4" in assembly
    assert "%zmm29" in assembly
    assert "%k4" in assembly
    assert "%zmm30" in assembly
    assert "%zmm31" in assembly


@pytest.mark.parametrize(("m", "n"), [(4, 2), (8, 3), (24, 4)])
def test_gather_kernel_rejects_unsupported_shapes(m: int, n: int) -> None:
    with pytest.raises(ValueError):
        generate_gather_asm(m=m, n=n, k=64, strategy="single")
