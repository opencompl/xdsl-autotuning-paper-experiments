"""Tests for the parts of machine detection that need no particular hardware.

Detection itself has to run on the machine it describes, so what is testable
here is the reasoning around it: how CPU flags become an ISA, and how the
fields that are not detected fall back.
"""

import pytest

from autotuner.machine_profile import (
    CpuInfo,
    base_frequency_ghz,
    detect_isa,
    detect_libxsmm_arch_amx,
    detect_libxsmm_arch_bf16,
    tidy_model,
)

AVX512 = frozenset(
    ("avx", "avx2", "avx512f", "avx512dq", "avx512bw", "avx512vl", "avx512cd")
)


def cpu(flags=frozenset(), model="", vendor="AuthenticAMD") -> CpuInfo:
    return CpuInfo(model=model, flags=frozenset(flags), vendor=vendor)


def isa(flags) -> str:
    """The ISA of an x86-64 host with these flags."""
    return detect_isa(cpu(flags), "x86_64")


# --------------------------------------------------------------------------- #
# ISA
# --------------------------------------------------------------------------- #


def test_full_avx512_is_avx512() -> None:
    assert isa(AVX512) == "avx512"


def test_avx512f_alone_is_not_enough() -> None:
    # libxsmm's skx kernels and our own avx512 pipelines need dq/bw/vl too, so
    # promising avx512 on the strength of avx512f alone would fail at codegen.
    assert isa({"avx512f"}) == "x86_64"


def test_avx2_only_is_plain_x86_64() -> None:
    assert isa({"avx", "avx2", "fma"}) == "x86_64"


# --------------------------------------------------------------------------- #
# BF16 and AMX
# --------------------------------------------------------------------------- #

BF16 = AVX512 | {"avx512_bf16"}
AMX = BF16 | {"amx_tile", "amx_bf16", "amx_int8"}


def test_avx512_bf16_gets_the_cpx_kernels() -> None:
    assert detect_libxsmm_arch_bf16(cpu(BF16), "avx512") == "cpx"
    assert detect_libxsmm_arch_bf16(cpu(AVX512), "avx512") is None


def test_amx_needs_both_the_tiles_and_the_bf16_dot_product() -> None:
    assert detect_libxsmm_arch_amx(cpu(AMX), "avx512") == "spr"
    assert detect_libxsmm_arch_amx(cpu(BF16 | {"amx_tile"}), "avx512") is None
    # Zen 4 and Zen 5 have AVX512-BF16 but no AMX.
    assert detect_libxsmm_arch_amx(cpu(BF16), "avx512") is None


def test_bf16_kernels_need_the_avx512_isa() -> None:
    # The flags without dq/bw/vl do not make an avx512 machine, and neither
    # BF16 kernel family is generated for anything else.
    assert detect_libxsmm_arch_bf16(cpu(AMX), "x86_64") is None
    assert detect_libxsmm_arch_amx(cpu(AMX), "x86_64") is None


# --------------------------------------------------------------------------- #
# Presentation and frequency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model,expected",
    [
        ("AMD EPYC 9754 128-Core Processor", "AMD EPYC 9754"),
        ("Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz", "Intel Xeon Gold 6130"),
        ("AMD Ryzen 9 9950X 16-Core Processor", "AMD Ryzen 9 9950X"),
    ],
)
def test_model_names_are_trimmed_to_title_a_plot(model, expected) -> None:
    assert tidy_model(model) == expected


def test_frequency_falls_back_to_the_model_name(tmp_path, monkeypatch) -> None:
    # Intel puts the nominal clock in the model name and AMD does not, so this
    # is a fallback, not a source: with PAPI the field is never read anyway.
    monkeypatch.chdir(tmp_path)
    assert base_frequency_ghz(cpu(model="Xeon Gold 6130 CPU @ 2.10GHz")) == 2.10


def test_frequency_is_zero_when_nothing_says() -> None:
    # 0.0 is what makes machine-profile refuse to write a profile that would
    # time by wall clock without a frequency to scale it by.
    assert base_frequency_ghz(cpu(model="AMD EPYC 9754 128-Core Processor")) == 0.0


def test_an_arm_host_is_neon() -> None:
    assert detect_isa(cpu(frozenset({"fp", "asimd"})), "aarch64") == "neon"
