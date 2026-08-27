import pytest

from autotuner.direct_kdot_asm import generate_kdot_asm


def test_single_accumulator_matches_the_llvm_shape() -> None:
    assembly = generate_kdot_asm(m=1, n=1, k=64, strategy="single")

    assert "accumulator_sets=1" in assembly
    assert assembly.count("vmovapd") == 8
    assert assembly.count("vmulpd") == 1
    assert assembly.count("vfmadd231pd") == 7
    assert assembly.count("vextractf64x4") == 1
    assert assembly.index("vfmadd231pd") < assembly.index("vaddsd\t(%rdx)")
    assert "\tjl\t" not in assembly


def test_multi_accumulator_shortens_the_dependency_chains() -> None:
    assembly = generate_kdot_asm(m=1, n=1, k=64, strategy="multi")

    assert "accumulator_sets=4" in assembly
    assert assembly.count("vmulpd") == 4
    assert assembly.count("vfmadd231pd") == 4
    assert assembly.count("vaddpd\t%zmm") == 3


def test_multi_accumulator_adapts_to_register_pressure() -> None:
    assembly = generate_kdot_asm(m=16, n=1, k=64, strategy="multi")

    assert "accumulator_sets=1" in assembly


@pytest.mark.parametrize(
    ("m", "n", "k", "message"),
    [
        (0, 1, 64, "M must be positive"),
        (1, 2, 64, "requires N=1"),
        (1, 1, 63, "multiple of eight"),
        (25, 1, 64, "do not fit"),
    ],
)
def test_invalid_shapes(m: int, n: int, k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_kdot_asm(m=m, n=n, k=k, strategy="multi")
