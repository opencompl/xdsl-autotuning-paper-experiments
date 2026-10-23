from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Machine:
    family: str
    isa: str
    display_name: str
    target_triple: str
    march: str
    mtune: str
    libxsmm_arch: str | None
    # The libxsmm arch code for BF16 x BF16 -> F32 kernels, on machines with
    # native AVX512-BF16.  `cpx` is the first arch libxsmm emits `vdpbf16ps`
    # for; like `libxsmm_arch`, it does not claim the CPU is a Cooper Lake.
    libxsmm_arch_bf16: str | None
    freq: float
    peak_f32: int
    libs: tuple[str, ...]
    linker_flag: str
    env: Mapping[str, str]


NEON = Machine(
    family="apple-m2-max",
    isa="neon",
    display_name="Apple M2 Max",
    target_triple="arm64-apple-darwin",
    march="armv8.5-a",
    mtune="armv8.5-a",
    libxsmm_arch=None,
    libxsmm_arch_bf16=None,
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
    libxsmm_arch_bf16=None,
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
    libxsmm_arch_bf16="cpx",
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
    libxsmm_arch_bf16=None,
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
    libxsmm_arch_bf16="cpx",
    freq=3.2,
    peak_f32=32,
    libs=("papi",),
    linker_flag="-fuse-ld=lld",
    env={},
)

M4 = Machine(
    family="apple-m4",
    isa="neon-sme",
    display_name="Apple M4 MacBook Air",
    target_triple="arm64-apple-darwin",
    march="armv8.7-a",
    mtune="apple-m4",
    libxsmm_arch=None,  # The standalone generator does not support ARM.
    libxsmm_arch_bf16=None,
    freq=1.0,  # Legacy schema placeholder, never a measured/assumed M4 clock.
    peak_f32=0,  # NEON and SME do not share a peak.
    libs=(),
    linker_flag="",
    env={},
)

MACHINES: Mapping[str, Machine] = {
    "m4": M4,
    "neon": NEON,
    "ci": CI,
    "tower": TOWER,
    "pinocchio": PINOCCHIO,
    "rapper": RAPPER,
}
