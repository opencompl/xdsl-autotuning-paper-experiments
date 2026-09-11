"""The machines the harness knows how to measure on.

Two kinds of entry live here. The machines the paper reports are written out
below and never change, so their numbers stay reproducible. Everything else --
above all a Grid'5000 node, where the hardware is whatever OAR handed out --
comes from a JSON profile under `machines/`, detected on the machine itself by
`machine-profile` (see `src/autotuner/machine_profile.py`) and committed next to
the data it produced.

A profile file is named after the machine, `machines/<name>.json`, and holds:

    {
      "machine":  { ... the fields of Machine ... },
      "detected": { ... free-form provenance, ignored here ... }
    }

`detected` is what the probe saw (CPU model, PAPI outcome, what the clock was
doing); it is deliberately not part of `Machine`, so a profile records both the
configuration and the evidence for it.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Machine:
    family: str
    isa: str
    display_name: str
    target_triple: str
    march: str
    mtune: str
    libxsmm_arch: str | None
    freq: float
    peak_f32: int
    libs: tuple[str, ...]
    linker_flag: str
    env: Mapping[str, str]

    @classmethod
    def from_dict(cls, data: Mapping[str, object], source: str = "<dict>") -> "Machine":
        """Build a machine from a profile's `machine` object, strictly.

        A profile is an input to every measurement that follows it, so a typo
        in one has to be an error here rather than a silently defaulted field.
        """
        names = {field.name for field in fields(cls)}
        if unknown := sorted(set(data) - names):
            raise ValueError(
                f"{source}: unknown machine field(s): {', '.join(unknown)}"
            )
        if missing := sorted(names - set(data)):
            raise ValueError(
                f"{source}: missing machine field(s): {', '.join(missing)}"
            )
        values = dict(data)
        # JSON has no tuples, and `libs` is compared and joined as a sequence
        # everywhere else; normalise so a profile and a literal behave alike.
        values["libs"] = tuple(values["libs"])  # type: ignore[arg-type]
        values["env"] = dict(values["env"])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        """The inverse of `from_dict`, for writing a profile out."""
        data: dict[str, object] = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        data["libs"] = list(self.libs)
        data["env"] = dict(self.env)
        return data


NEON = Machine(
    family="apple-m2-max",
    isa="neon",
    display_name="Apple M2 Max",
    target_triple="arm64-apple-darwin",
    march="armv8.5-a",
    mtune="armv8.5-a",
    libxsmm_arch=None,
    freq=1.0,
    peak_f32=0,
    libs=(),
    linker_flag="",
    env={},
)

CI = Machine(
    family="generic-x86-64",
    isa="x86_64",
    display_name="x86-64 CI",
    target_triple="x86_64-unknown-linux-gnu",
    march="x86-64",
    mtune="x86-64",
    libxsmm_arch=None,
    freq=1.0,
    peak_f32=0,
    libs=(),
    linker_flag="-fuse-ld=lld",
    env={},
)

TOWER = Machine(
    family="zen5",
    isa="avx512",
    display_name="AMD Zen 5",
    target_triple="x86_64-unknown-linux-gnu",
    march="znver5",
    mtune="znver5",
    libxsmm_arch="skx",
    freq=4.3,
    peak_f32=64,
    libs=("papi",),
    linker_flag="-fuse-ld=lld",
    env={"LIBPFM_FORCE_PMU": "amd64"},
)

PINOCCHIO = Machine(
    family="cascadelake",
    isa="avx512",
    display_name="Intel Cascade Lake",
    target_triple="x86_64-unknown-linux-gnu",
    march="cascadelake",
    mtune="cascadelake",
    libxsmm_arch="clx",
    freq=2.1,
    peak_f32=64,
    libs=("papi",),
    linker_flag="-fuse-ld=lld",
    env={},
)

RAPPER = Machine(
    family="zen4",
    isa="avx512",
    display_name="AMD Zen 4",
    target_triple="x86_64-unknown-linux-gnu",
    march="znver4",
    mtune="znver4",
    libxsmm_arch="skx",
    freq=3.2,
    peak_f32=32,
    libs=("papi",),
    linker_flag="-fuse-ld=lld",
    env={},
)

# The machines the paper reports. Hand-written and frozen: a detected profile is
# never allowed to shadow one of these, or a stray file in a working tree would
# silently redefine a published measurement.
STATIC_MACHINES: Mapping[str, Machine] = {
    "neon": NEON,
    "ci": CI,
    "tower": TOWER,
    "pinocchio": PINOCCHIO,
    "rapper": RAPPER,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ENV_VAR = "AUTOTUNER_MACHINES_DIR"


def repo_root() -> Path:
    """The checkout to read `machines/`, `headers/` and `kernels/` from.

    `REPO_ROOT` is derived from this file's location, which is the checkout
    only while the project is installed editable -- as `uv sync` and
    `docker/entrypoint.sh` do install it. The working directory is preferred
    anyway, so that an ordinary install cannot silently ignore every profile
    and report the machine as unknown; it has to look like this repository to
    count, and everything that reads the tree agrees on the answer.
    """
    here = Path.cwd()
    return here if (here / "Snakefile").is_file() else REPO_ROOT


def profile_directory() -> Path:
    """Where detected machine profiles live."""
    if override := os.environ.get(PROFILE_ENV_VAR):
        return Path(override)
    return repo_root() / "machines"


def load_profiles(directory: Path | None = None) -> dict[str, Machine]:
    """Read every `machines/<name>.json` profile, keyed by file name."""
    directory = directory if directory is not None else profile_directory()
    profiles: dict[str, Machine] = {}
    for path in sorted(directory.glob("*.json")):
        name = path.stem
        if name in STATIC_MACHINES:
            raise ValueError(
                f"{path} would shadow the built-in machine {name!r}; rename the profile"
            )
        payload = json.loads(path.read_text())
        if "machine" not in payload:
            raise ValueError(f"{path}: profile has no 'machine' object")
        profiles[name] = Machine.from_dict(payload["machine"], source=str(path))
    return profiles


MACHINES: Mapping[str, Machine] = {**STATIC_MACHINES, **load_profiles()}
