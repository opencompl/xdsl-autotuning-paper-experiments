import json

import pytest

from autotuner.machines import (
    MACHINES,
    PROFILE_ENV_VAR,
    STATIC_MACHINES,
    Machine,
    load_profiles,
    profile_directory,
)


def test_machine_vocabulary_is_explicit() -> None:
    # The machines the paper reports are written out by hand and stay that way;
    # a detected profile adds to this, it never edits it.
    assert set(STATIC_MACHINES) == {"neon", "ci", "tower", "pinocchio", "rapper"}
    assert set(STATIC_MACHINES) <= set(MACHINES)
    for machine in MACHINES.values():
        assert machine.family
        assert machine.isa
        assert machine.display_name
        assert machine.target_triple
        assert machine.march
        assert machine.mtune


def test_avx512_machine_mappings() -> None:
    expected = {
        "tower": ("zen5", "znver5", "skx"),
        "pinocchio": ("cascadelake", "cascadelake", "clx"),
        "rapper": ("zen4", "znver4", "skx"),
    }
    for name, (family, march, libxsmm_arch) in expected.items():
        machine = MACHINES[name]
        assert machine.family == family
        assert machine.isa == "avx512"
        assert machine.march == march
        assert machine.libxsmm_arch == libxsmm_arch

    assert MACHINES["neon"].libxsmm_arch is None
    assert MACHINES["ci"].libxsmm_arch is None


# --------------------------------------------------------------------------- #
# Detected profiles
# --------------------------------------------------------------------------- #


def profile(**overrides) -> dict:
    machine = {
        "family": "znver4",
        "isa": "avx512",
        "display_name": "AMD EPYC 9754 (grdix)",
        "target_triple": "x86_64-unknown-linux-gnu",
        "march": "znver4",
        "mtune": "znver4",
        "libxsmm_arch": "skx",
        "freq": 2.25,
        "peak_f32": 32,
        "libs": ["papi"],
        "linker_flag": "-fuse-ld=lld",
        "env": {},
    }
    machine.update(overrides)
    return {"machine": machine, "detected": {"cpu_model": "AMD EPYC 9754"}}


def write(directory, name: str, payload: dict) -> None:
    (directory / f"{name}.json").write_text(json.dumps(payload))


def test_a_profile_becomes_a_machine(tmp_path) -> None:
    write(tmp_path, "grdix", profile())
    profiles = load_profiles(tmp_path)

    assert set(profiles) == {"grdix"}
    machine = profiles["grdix"]
    assert machine.march == "znver4"
    # JSON has no tuples, but everything downstream treats libs as one.
    assert machine.libs == ("papi",)


def test_round_trip_through_a_profile_is_lossless() -> None:
    for machine in STATIC_MACHINES.values():
        assert Machine.from_dict(machine.to_dict()) == machine


def test_a_profile_may_not_shadow_a_reported_machine(tmp_path) -> None:
    # A stray file in a working tree redefining a published measurement is the
    # one failure mode worth being loud about.
    write(tmp_path, "tower", profile())
    with pytest.raises(ValueError, match="shadow"):
        load_profiles(tmp_path)


def test_a_missing_field_is_an_error_naming_the_file(tmp_path) -> None:
    payload = profile()
    del payload["machine"]["peak_f32"]
    write(tmp_path, "grdix", payload)
    with pytest.raises(ValueError, match="peak_f32"):
        load_profiles(tmp_path)


def test_an_unknown_field_is_an_error(tmp_path) -> None:
    write(tmp_path, "grdix", profile(peak_f64=16))
    with pytest.raises(ValueError, match="peak_f64"):
        load_profiles(tmp_path)


def test_a_profile_needs_the_machine_object(tmp_path) -> None:
    write(tmp_path, "grdix", {"march": "znver4"})
    with pytest.raises(ValueError, match="no 'machine' object"):
        load_profiles(tmp_path)


def test_no_profiles_is_not_an_error(tmp_path) -> None:
    assert load_profiles(tmp_path / "does-not-exist") == {}


def test_the_profile_directory_can_be_pointed_elsewhere(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(PROFILE_ENV_VAR, str(tmp_path))
    assert profile_directory() == tmp_path


def test_the_profile_directory_prefers_a_repository_shaped_cwd(
    tmp_path, monkeypatch
) -> None:
    # Found by working directory rather than by this module's location, so a
    # non-editable install does not silently ignore every profile.
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    (tmp_path / "Snakefile").write_text("")
    (tmp_path / "machines").mkdir()
    monkeypatch.chdir(tmp_path)
    assert profile_directory() == tmp_path / "machines"


def test_an_unrelated_cwd_is_not_mistaken_for_the_repository(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
    (tmp_path / "machines").mkdir()  # but no Snakefile
    monkeypatch.chdir(tmp_path)
    assert profile_directory() != tmp_path / "machines"
