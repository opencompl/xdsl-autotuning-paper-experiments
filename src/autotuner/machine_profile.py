"""Detect this machine's profile, so a new target needs no hand-written entry.

`machines.py` carries the machines the paper reports. Everything else --
a Grid'5000 node above all, where the hardware is whatever OAR handed out --
is described by a profile this tool writes:

    uv run machine-profile --name grdix

Nothing here is a lookup table keyed on a CPU we happen to have seen before.
Each field is asked of the machine itself: the compiler is asked which CPU it
thinks it is targeting, the CPU flags say which ISA is available, and a PAPI
probe says whether hardware counters can actually be read here, which is the
one thing that decides how every later measurement is timed.

The exception is `peak_f32`, which is not discoverable at all: the vector width
is in the CPU flags, but the number of FMA pipes behind it is not. It is left
unset, which the plots read as "no peak known", and can be filled into the
profile whenever it is known -- the timings are cached per kernel, so only the
rows are rebuilt.

The result is written to `machines/<name>.json`, picked up by
`machines.load_profiles`, and meant to be committed next to the data it
produced -- the profile is an input to the measurements, not a by-product.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .machines import STATIC_MACHINES, Machine, repo_root

# libxsmm's `skx` code generation, and our own avx512 pass pipelines, need all
# of these; `avx512f` alone is not enough to promise the rest.
AVX512_FLAGS = ("avx512f", "avx512dq", "avx512bw", "avx512vl")

# A PMU libpfm4 does not recognise makes PAPI refuse to count, and forcing the
# generic AMD PMU is what makes it work; `tower` needs exactly this.
AMD_PMU_FALLBACK = {"LIBPFM_FORCE_PMU": "amd64"}

# --------------------------------------------------------------------------- #
# Asking the machine
# --------------------------------------------------------------------------- #


def run(
    command: Sequence[str], env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a probe command, capturing both streams and never raising."""
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
    )


def detect_target_cpu(cc: str) -> str | None:
    """Ask the compiler which CPU `-march=native` resolves to.

    This is the whole reason no vendor/family/model table is needed: clang
    already knows how to identify the host, and its answer is by construction
    the one it will use when it compiles the kernels.
    """
    completed = run([cc, "-march=native", "-###", "-x", "c", "-c", "/dev/null"])
    # `-###` prints the cc1 invocation on stderr, with every argument quoted.
    found = re.findall(r'"-target-cpu" "([^"]+)"', completed.stderr)
    return found[0] if found else None


def detect_triple(cc: str) -> str | None:
    completed = run([cc, "-dumpmachine"])
    triple = completed.stdout.strip()
    return triple or None


@dataclass(frozen=True)
class CpuInfo:
    model: str
    flags: frozenset[str]
    vendor: str


def read_cpuinfo(path: Path = Path("/proc/cpuinfo")) -> CpuInfo:
    """Model name, vendor and feature flags of the first core."""
    model, vendor = "", ""
    flags: frozenset[str] = frozenset()
    try:
        text = path.read_text()
    except OSError:
        return CpuInfo(model=platform.processor() or "", flags=flags, vendor="")
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "model name" and not model:
            model = value
        elif key == "vendor_id" and not vendor:
            vendor = value
        # x86 spells them `flags`, aarch64 `Features`.
        elif key in ("flags", "features") and not flags:
            flags = frozenset(value.split())
    return CpuInfo(model=model, flags=flags, vendor=vendor)


def detect_isa(cpu: CpuInfo, arch: str | None = None) -> str:
    """The ISA key the pass pipelines in `default.yaml` are chosen by.

    `avx512f` on its own is not enough to claim avx512: libxsmm's skx kernels
    and our own avx512 pipelines use the dq/bw/vl extensions too, so promising
    the ISA on the strength of the base flag would fail at code generation
    instead of here.
    """
    arch = arch if arch is not None else platform.machine()
    if arch in ("arm64", "aarch64"):
        return "neon"
    if all(flag in cpu.flags for flag in AVX512_FLAGS):
        return "avx512"
    return "x86_64"


def base_frequency_ghz(cpu: CpuInfo) -> float:
    """The clock the machine runs at with boost off, in GHz.

    Only used when hardware counters are unavailable and cycles have to be
    derived from wall time; with PAPI it is recorded but never read.
    """
    for name in ("base_frequency", "cpuinfo_max_freq"):
        path = Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{name}")
        try:
            return int(path.read_text().strip()) / 1e6  # kHz -> GHz
        except (OSError, ValueError):
            continue
    # Intel puts the nominal clock in the model name; AMD does not.
    if match := re.search(r"@\s*([\d.]+)\s*GHz", cpu.model):
        return float(match.group(1))
    return 0.0


def governors() -> dict[str, str]:
    """What the CPU frequency driver is currently set to, for the record."""
    observed: dict[str, str] = {}
    for name in ("scaling_governor", "scaling_cur_freq"):
        path = Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{name}")
        try:
            observed[name] = path.read_text().strip()
        except OSError:
            pass
    for path in (
        Path("/sys/devices/system/cpu/cpufreq/boost"),
        Path("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    ):
        try:
            observed[path.name] = path.read_text().strip()
        except OSError:
            pass
    return observed


def perf_event_paranoid() -> str | None:
    try:
        return Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Probing PAPI
# --------------------------------------------------------------------------- #

# Exercises exactly what the benchmarks do -- `headers/perf.h` with USE_PAPI --
# rather than asking `papi_avail` a related question. perf.h exits non-zero if
# the library, the event or the paranoid level is not up to it.
PAPI_PROBE = """#include "{perf_h}"
int main(void) {{
  time_init();
  time_start();
  volatile long long spin = 0;
  for (long long i = 0; i < 10000000; i++) spin += i;
  (void)time_end(1.0);
  return 0;
}}
"""


def compile_and_run(
    *,
    source: str,
    cc: str,
    flags: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> tuple[bool, str, str]:
    """Compile a C program and run it; return (ok, stdout, diagnostics)."""
    with tempfile.TemporaryDirectory(prefix="machine-profile-") as directory:
        binary = Path(directory) / "probe"
        source_path = Path(directory) / "probe.c"
        source_path.write_text(source)
        # `flags` last, so -lpapi follows the object that needs it.
        build = run([cc, "-O2", str(source_path), "-o", str(binary), *flags])
        if build.returncode != 0:
            return False, "", f"compilation failed: {build.stderr.strip()}"
        executed = run([str(binary)], env=env)
        if executed.returncode != 0:
            return False, executed.stdout, executed.stderr.strip() or "non-zero exit"
        return True, executed.stdout, executed.stderr.strip()


def probe_papi(cc: str, root: Path, vendor: str) -> tuple[bool, dict[str, str], str]:
    """Can this machine's hardware cycle counter actually be read?

    Returns whether PAPI works, the environment it needs to work, and what went
    wrong if it does not. This is the field worth probing rather than assuming:
    a wrong answer either silently falls back to wall-clock timing scaled by a
    guessed frequency, or fails a kernel at a time hours into a run.
    """
    source = PAPI_PROBE.format(perf_h=root / "headers" / "perf.h")
    flags = ["-DUSE_PAPI", "-lpapi"]
    ok, _, diagnostics = compile_and_run(source=source, cc=cc, flags=flags)
    if ok:
        return True, {}, ""
    if "AMD" in vendor.upper():
        retry, _, retry_diagnostics = compile_and_run(
            source=source, cc=cc, flags=flags, env=AMD_PMU_FALLBACK
        )
        if retry:
            return True, dict(AMD_PMU_FALLBACK), ""
        diagnostics = f"{diagnostics}; with LIBPFM_FORCE_PMU=amd64: {retry_diagnostics}"
    return False, {}, diagnostics


# --------------------------------------------------------------------------- #
# Building the profile
# --------------------------------------------------------------------------- #


def tidy_model(model: str) -> str:
    """A model name short enough to title a plot."""
    trimmed = re.sub(r"\((R|TM|r|tm)\)", "", model)
    trimmed = re.sub(r"\b\d+-Core\b", "", trimmed)
    trimmed = re.sub(r"@.*$", "", trimmed)
    trimmed = re.sub(r"\b(CPU|Processor)\b", "", trimmed)
    return " ".join(trimmed.split())


def build_profile(args: argparse.Namespace) -> tuple[Machine, dict[str, object]]:
    """Detect everything, or take what the caller insisted on."""
    root = Path(args.root).resolve()
    cpu = read_cpuinfo()
    isa = args.isa or detect_isa(cpu, platform.machine())

    detected_cpu = detect_target_cpu(args.cc)
    if args.march:
        march = args.march
    elif detected_cpu and platform.machine() in ("x86_64", "AMD64"):
        march = detected_cpu
    else:
        # On aarch64 clang wants -mcpu for a specific part and the Snakefile
        # always passes -march, so the host's own name for itself is not usable
        # there; `native` is, at the cost of a vaguer `compiler_march` field.
        march = "native"

    triple = args.target_triple or detect_triple(args.cc) or ""
    if not triple:
        raise SystemExit(
            f"error: {args.cc} -dumpmachine said nothing; pass --target-triple"
        )

    if args.no_papi:
        papi_ok, papi_env, papi_error = False, {}, "disabled with --no-papi"
    else:
        papi_ok, papi_env, papi_error = probe_papi(args.cc, root, cpu.vendor)

    freq = args.freq if args.freq is not None else base_frequency_ghz(cpu)
    if not papi_ok and freq <= 0:
        raise SystemExit(
            "error: hardware counters are unavailable "
            f"({papi_error or 'unknown reason'}) and this machine's base "
            "frequency could not be read, so wall-clock timings could not be "
            "converted to cycles. Pass --freq GHZ, or make PAPI work: "
            "`sudo sysctl -w kernel.perf_event_paranoid=-1` and, in a "
            "container, --cap-add=PERFMON."
        )

    detected: dict[str, object] = {
        "hostname": platform.node(),
        "cpu_model": cpu.model,
        "cpu_vendor": cpu.vendor,
        "uname": " ".join(platform.uname()),
        "clang_target_cpu": detected_cpu,
        "avx512_flags_present": sorted(f for f in AVX512_FLAGS if f in cpu.flags),
        "papi_usable": papi_ok,
        "papi_error": papi_error or None,
        "perf_event_paranoid": perf_event_paranoid(),
        "cpufreq": governors(),
        "base_frequency_ghz": base_frequency_ghz(cpu),
    }

    # The one field that cannot be detected: the vector width is in the flags,
    # but the number of FMA pipes behind it is not -- Zen 4 and Zen 4c
    # advertise avx512 over a 256-bit datapath, and an Intel avx512 part may or
    # may not have its second FMA unit. Left at 0 until someone says, which
    # `plot_ttile` reads as "no peak known" and falls back to absolute
    # throughput -- the two % of peak figures, `plot_squares` and `plot_grid`,
    # refuse to draw at all. It can be filled in afterwards without
    # re-measuring: the timings live in build/<machine>/**/time.txt and only
    # the rows are rebuilt from them.
    peak = args.peak_f32 if args.peak_f32 is not None else 0
    detected["peak_note"] = (
        f"given as --peak-f32 {peak}"
        if args.peak_f32 is not None
        else (
            "unknown, so the ttile plots use absolute throughput rather "
            "than a percentage of peak, and the % of peak figures cannot be "
            "drawn. Fill peak_f32 in later (vector lanes * FMA pipes * 2) and "
            "rebuild the rows; no re-timing is needed."
        )
    )

    if args.libxsmm_arch is not None:
        libxsmm_arch = args.libxsmm_arch or None
    else:
        # Every avx512 machine here generates `skx` kernels; `pinocchio` asks
        # for `clx` explicitly, which is what --libxsmm-arch is for.
        libxsmm_arch = "skx" if isa == "avx512" else None

    machine = Machine(
        family=march,
        isa=isa,
        display_name=args.display_name
        or (f"{tidy_model(cpu.model)} ({args.name})" if cpu.model else args.name),
        target_triple=triple,
        march=march,
        mtune=march,
        libxsmm_arch=libxsmm_arch,
        freq=freq,
        peak_f32=peak,
        libs=("papi",) if papi_ok else (),
        linker_flag=args.linker_flag,
        env=papi_env,
    )
    return machine, detected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect this machine's profile and write it to machines/<name>.json."
        ),
        epilog=(
            "Every field is detected unless overridden. The overrides exist for "
            "the cases detection cannot reach, above all a libxsmm "
            "architecture other than skx."
        ),
    )
    parser.add_argument(
        "--name",
        required=True,
        help=(
            "machine name, which keys data/<name>/ and build/<name>/. On "
            "Grid'5000 use the cluster: its nodes are identical by definition."
        ),
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="where to write the profile (default: <root>/machines/<name>.json)",
    )
    parser.add_argument(
        "--root",
        default=str(repo_root()),
        help="repository root, holding headers/ and machines/ (default: %(default)s)",
    )
    parser.add_argument(
        "--cc", default="clang", help="compiler to ask (default: %(default)s)"
    )

    parser.add_argument(
        "--peak-f32",
        type=int,
        metavar="N",
        help=(
            "f32 flops per cycle (vector lanes * FMA pipes * 2). Not "
            "detectable, and not needed up front: left out, the plots show "
            "absolute throughput instead of a percentage of peak, and it can "
            "be added to the profile later without re-timing anything."
        ),
    )

    overrides = parser.add_argument_group("overrides")
    overrides.add_argument("--isa", help="avx512, x86_64 or neon")
    overrides.add_argument(
        "--march", help="override the compiler's own -march=native answer"
    )
    overrides.add_argument("--target-triple")
    overrides.add_argument("--display-name", help="plot title for this machine")
    overrides.add_argument(
        "--libxsmm-arch",
        help="libxsmm architecture, e.g. clx; pass an empty string for none",
    )
    overrides.add_argument("--freq", type=float, metavar="GHZ")
    overrides.add_argument(
        "--linker-flag",
        default="-fuse-ld=lld",
        help="extra link flag (default: %(default)s)",
    )
    overrides.add_argument(
        "--no-papi",
        action="store_true",
        help="do not use hardware counters even if they work",
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="leave an existing profile alone instead of overwriting it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.name in STATIC_MACHINES:
        print(
            f"error: {args.name!r} is a built-in machine; "
            "detected profiles may not shadow one",
            file=sys.stderr,
        )
        return 2
    if not shutil.which(args.cc):
        print(f"error: {args.cc} is not on PATH", file=sys.stderr)
        return 2

    out = (
        Path(args.out)
        if args.out
        else Path(args.root) / "machines" / f"{args.name}.json"
    )
    if out.exists() and args.keep_existing:
        print(f"{out} exists and --keep-existing was given; leaving it alone")
        return 0

    machine, detected = build_profile(args)
    payload = {"machine": machine.to_dict(), "detected": detected}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {out}", file=sys.stderr)
    if not machine.libs:
        print(
            "warning: hardware counters are unavailable, so cycles come from "
            f"the monotonic clock scaled by freq={machine.freq} GHz. Fix the "
            "clock (performance governor, boost off) or the numbers will move "
            "with it.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
