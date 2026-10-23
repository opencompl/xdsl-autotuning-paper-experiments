from autotuner.machines import MACHINES


def test_machine_vocabulary_is_explicit() -> None:
    assert set(MACHINES) == {"m4", "neon", "ci", "tower", "pinocchio", "rapper"}
    for machine in MACHINES.values():
        assert machine.family
        assert machine.isa
        assert machine.display_name
        assert machine.target_triple
        assert machine.march
        assert machine.mtune


def test_avx512_machine_mappings() -> None:
    expected = {
        "tower": ("zen5", "znver5", "skx", "cpx"),
        "pinocchio": ("cascadelake", "cascadelake", "clx", None),
        "rapper": ("zen4", "znver4", "skx", "cpx"),
    }
    for name, (family, march, libxsmm_arch, bf16_arch) in expected.items():
        machine = MACHINES[name]
        assert machine.family == family
        assert machine.isa == "avx512"
        assert machine.march == march
        assert machine.libxsmm_arch == libxsmm_arch
        # Cascade Lake has no AVX512-BF16; both Zen generations do.
        assert machine.libxsmm_arch_bf16 == bf16_arch

    assert MACHINES["neon"].libxsmm_arch is None
    assert MACHINES["ci"].libxsmm_arch is None
