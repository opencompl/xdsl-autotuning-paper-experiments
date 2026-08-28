from dataclasses import dataclass


@dataclass
class Target:
    triple: str
    arch: str
    freq: float
    peak_f32: int
    libs: list[str]
    linker_flag: str
    env: dict[str, str]


NEON = Target(
    triple="arm64-apple-darwin",
    arch="armv8.5-a",
    freq=1.0,
    peak_f32=0,
    libs=[],
    linker_flag="",
    env={},
)

CI = Target(
    triple="x86_64-unknown-linux-gnu",
    arch="x86-64",
    freq=1.0,
    peak_f32=0,
    libs=[],
    linker_flag="-fuse-ld=lld",
    env={},
)

TOWER = Target(
    triple="x86_64-unknown-linux-gnu",
    arch="znver5",
    freq=4.3,
    peak_f32=64,
    libs=["papi"],
    linker_flag="-fuse-ld=lld",
    env={"LIBPFM_FORCE_PMU": "amd64"},
)

PINOCCHIO = Target(
    triple="x86_64-unknown-linux-gnu",
    arch="cascadelake",
    freq=2.1,
    peak_f32=64,
    libs=["papi"],
    linker_flag="-fuse-ld=lld",
    env={},
)

RAPPER = Target(
    triple="x86_64-unknown-linux-gnu",
    arch="znver4",
    freq=3.2,
    peak_f32=32,
    libs=["papi"],
    linker_flag="-fuse-ld=lld",
    env={},
)

TARGETS = {
    "neon": NEON,
    "ci": CI,
    "tower": TOWER,
    "pinocchio": PINOCCHIO,
    "rapper": RAPPER,
}
