# xdsl-autotuning-paper-experiments

This repository contains code to generate data and graphs for the xDSL Autotuning paper
(title TBD).

The objective is for the code to be easy to adapt and extend, and to be able to run on
four platforms:

1. macOS on ARM
2. Intel native
3. GitHub CI
4. Docker

Each of these has quirks and limitations.

`uiCA` can only be installed on x86 devices, and has a weird installation process, so
it's only installed exercised in the Docker container.
Running Docker on ARM macs lets us execute x86 code, but we have not set up ARM
simulation to test the ARM code on x86 devices.

## Setting Up

There are two kinds of actions to perform in this repository: running tests (to quickly
check that the code in this repo is correct), and compute the data and charts for the
paper.

### Machines, models, and ISAs

The configuration keeps four related concepts separate:

- **Machine:** a named execution environment, such as `tower`, `pinocchio`, or
  `rapper`. This selects measurement settings and names output directories.
- **Family:** the processor family in that machine, such as `zen5` or
  `cascadelake`.
- **ISA:** the instructions that generated code may use, such as `avx512` or
  `neon`.
- **Compiler configuration:** the compiler's target triple, `-march`, and
  `-mtune` values. These use compiler-specific spellings such as `znver5`.

The libxsmm generator deliberately retains libxsmm's own `arch` terminology and
codes such as `skx` and `clx`. A non-Intel machine may therefore have
`isa: avx512` and `libxsmm_arch: skx`: the latter is passed to libxsmm and does
not claim that the physical CPU is Skylake. The CompXSMM reimplementation uses
a separate `strategy` option, currently `libxsmm-skx`, to select the scheduling
and nano-kernel policy reproduced from libxsmm. This makes room for future
non-libxsmm strategies without confusing a policy with the machine's ISA.
Likewise, the llvm-mca analyzer exposes `arch` and `cpu`, matching llvm-mca's
`-march` and `-mcpu` options; those names are local to that tool boundary.

### Setting up a new machine

When running on a new machine, please create a `.env` file with the format:

```sh
MACHINE=your_machine_name_here
# Optional. Omit to use Snakemake's default ILP scheduler.
# On Apple Silicon, set greedy — PuLP's bundled CBC is x86_64-only.
# SNAKEMAKE_SCHEDULER=greedy
```

Then add the machine to
[`src/autotuner/machines.py`](src/autotuner/machines.py). Specify its family,
ISA, compiler settings, and—when libxsmm variants are supported—the
corresponding `libxsmm_arch`. Finally, populate `TESTSET` and
`DATASET_VARIANTS` in the Snakefile.

`neon` is retained as the historical machine identifier for the Apple M2 Max;
its `isa` field, rather than its name, is the authoritative ISA metadata. `ci`
is a synthetic generic x86-64 machine profile.

### Running Tests

We use two kinds of tests in this repository:

1. lit/filecheck
2. snakemake tests

Running `make tests` executes both of them, installing dependencies if necessary.
These should be able to run on the host computer, or on the Docker container.
In order to execute them in the Docker container, first run `make docker-run`, then
`make tests`.
The two test CI jobs test each of these flows, but we don't have an ARM CI so one tests
the Docker container, and the other host linux x86 execution, so please be mindful when
pushing things that affect ARM code, as these may have to be tested locally.

Running `make tests` will automatically detect the platform, and run only the tests that
can be executed on that machine.
For example, when executing `make tests` on macOS, x86 assembly will be created, but it
will not be executed.

### Computing Data

Generate data for the selected machine by running `make dataset`. JSONL outputs
are written under `data/<MACHINE>/` (with `MACHINE` from `.env` or the `machine`
setting in `default.yaml`); filenames use `<dtype>.<dataset>.jsonl` (for example
`f32.ttile.jsonl`). Build artifacts go under `build/<MACHINE>/`. Each new result
records its machine, family, ISA, compiler `march`, and libxsmm architecture.

[T-tile chart generation.](https://gitlab.inria.fr/ntollena/ics-experiments/-/tree/main/paper_versions/asplos/small_mm_figure_Gui?ref_type=heads)

### AOCL-BLAS baseline

On x86-64 Linux, the Nix toolchain builds the single-threaded CBLAS interface from
AMD AOCL-BLAS 5.3.2. It uses the `amdzen` configuration so the same package contains
the Zen 4 and Zen 5 AVX-512 kernels and selects the appropriate implementation at
runtime. The Docker image receives the same package through the copied Nix closure.

The `aocl` benchmark variant is enabled for the `tower`, `rapper`, and `pinocchio`
targets. Before collecting data on a new machine, verify the package and the selected
runtime code path:

```sh
pkg-config --modversion blis
BLIS_ARCH_DEBUG=1 uv run snakemake --cores 1 --forceall \
  build/tower/matmul_rowmaj/3x16x5/aocl.f64.test.log \
  --config target=tower
```

The debug run should report an architecture-specific path rather than `generic`.
Do not set `BLIS_ARCH_TYPE` for measured runs: it overrides AOCL's safety checks and
can force unsupported instructions. Measurements set both `OMP_NUM_THREADS=1` and
`BLIS_NUM_THREADS=1`; the packaged library itself is also built without threading.

Once this smoke test passes, `make dataset_validate TARGET=tower` validates the full
tower dataset and `make dataset TARGET=tower` collects its measurements.

### Lighthouse pipeline

The `lighthouse` variant compiles the matmul with the
[Lighthouse](https://github.com/libxsmm/lighthouse) project's own pipeline. The
repository is a Nix flake input (pinned in `flake.lock`), built by
[`nix/lighthouse.nix`](nix/lighthouse.nix) together with the MLIR Python bindings
it depends on ([`nix/mlir-python-bindings.nix`](nix/mlir-python-bindings.nix)),
and exposed as the `lighthouse-python` interpreter in the dev shell, the default
toolchain, and therefore the Docker image. Nothing from Lighthouse is copied into
this repository: the pipeline descriptor and the transform schedules it includes
are looked up in the installed package with `find_pipeline_file`, which selects
`x86_64/matmul/f32.yaml` for the host machine.

[`scripts/lighthouse_codegen.py`](scripts/lighthouse_codegen.py) drives it. Note
that this script runs under `lighthouse-python`, not under the project's uv
environment, so Lighthouse's nightly LLVM/MLIR stays out of `pyproject.toml`. It
takes [`kernels/matmul_rowmaj/lighthouse.mlir`](kernels/matmul_rowmaj/lighthouse.mlir)
— the same `linalg.matmul` payload as the other MLIR variants, with the result
returned rather than dropped, because Lighthouse schedules at the tensor level
and would otherwise fold a dead matmul away — and writes an object file, which
[`kernels/matmul_rowmaj/lighthouse_shim.c`](kernels/matmul_rowmaj/lighthouse_shim.c)
adapts to the `void matmul(A, B, C)` the drivers call.

Consequences of using the pipeline as it ships:

- **f32 and x86-64 only.** Lighthouse ships no f64 descriptor, so the variant
  takes part in the f32 half of the `ttile` sweep only.
- **Codegen runs on the machine being measured.** The object file comes out of
  Lighthouse's JIT, which targets the host, so unlike the other variants this
  one cannot be cross-compiled for another machine.
- **The kernel is parallelized with OpenMP** by the pipeline's `lower.yaml`
  stage, and packs its operands at run time. Measurements stay single-threaded
  because the `time` rule sets `OMP_NUM_THREADS=1`.

Smoke-test the flow on a new machine with:

```sh
nix develop
uv run snakemake --cores 1 --forceall \
  build/rapper/matmul_rowmaj/8x128x128/lighthouse.f32.test.log \
  --config machine=rapper
```

To move to a newer Lighthouse, run `nix flake update lighthouse` and set the
`version` and hashes in [`nix/mlir-python-bindings.nix`](nix/mlir-python-bindings.nix)
to the `mlir-python-bindings` pin in Lighthouse's `pyproject.toml`; the build
fails with a diagnostic if the two disagree.

### Plotting

Plot data using `make plots`; this command fails when required input data is
missing rather than starting measurements. PNGs are written under
`plots/<machine>/` for each machine with JSONL inputs (for example `neon`,
`tower`, or `pinocchio`). Plotting does not depend on the currently selected
`MACHINE`. Plot titles resolve each machine's `display_name` from
`src/autotuner/machines.py`; result files must use the current `machine` field
rather than the legacy `target` field.

## Building The Docker Container

Run `make docker-build`.

We have a CI script that publishes a new version of Docker automatically when a commit
in `main` is tagged with a tag like `v1.2.3`.
So far we've used 0ver (just incrementing the minor version, `v0.1.0`, `v0.2.0`, etc.).
After publishing a new image, bump the `container:` image tag in
[`.github/workflows/ci-docker.yml`](.github/workflows/ci-docker.yml) so the Docker CI job uses it.

### Virtual Environments

The aim is for this project to run both natively on the host and in Docker.

- **Host:** use `uv sync` (or `make` targets that rely on `uv run`) as usual; the project lives in a local `.venv` (default for uv).
- **Docker:** the image defines a single environment at `/opt/venv` via `UV_PROJECT_ENVIRONMENT` and `PATH`. The image is built with `pyproject.toml` and `uv.lock` (`uv sync --locked --no-install-project`). The [`docker/entrypoint.sh`](docker/entrypoint.sh) runs `uv sync --directory /src --locked --inexact` when the container starts so the mounted repository is installed in editable form and stays in sync with the lockfile. The `--inexact` flag keeps extra packages installed only in the image (for example uiCA) from being removed.

When you change the image or dependencies, rebuild the image (`make docker-build`) or pull a published CI image; you do not need a separate `venv_docker` directory on the host.

## Configuring Machines

### Disabling Frequency Switching

It's important for the cores to have predictable frequencies for a given machine.

On the tower (ASUS BIOS, AMD Ryzen 9 9950X):

- reboot computer and press del to go into BIOS
- Ai Tweaker
  - CPU Core Performance Boost → Disabled — prevents turbo frequencies that vary
    with thermal/power conditions
  - Ai Overclock Tuner → Manual — gives explicit control over clock settings
    instead of letting the board auto-adjust
  - Precision Boost Overdrive (PBO) → Disabled — prevents opportunistic boosting
    beyond stock limits based on thermal/power headroom
  - ASUS Performance Enhancement / MultiCore Enhancement → Disabled — prevents
    ASUS firmware from overriding AMD's default power limits
- Advanced → AMD CBS
  - Global C-state Control → Disabled — prevents cores from entering low-power
    sleep states, which cause variable wake-up latency
  - CPU Common Options
    - Power Supply Idle Control → Typical Current Idle — prevents the package
      from entering deep idle states that cause latency spikes on wake
  - DF Common Options
    - DF (Data Fabric) C-states → Disabled — prevents the Infinity Fabric
      (interconnect between CCDs) from entering idle states, which adds
      latency to cross-CCD memory accesses
  <!-- - CPPC → Disabled — disables AMD's per-core performance hinting to
    firmware, removing a source of asymmetric core behavior -->
  <!-- - CPPC Preferred Cores → Disabled — prevents the firmware from ranking
    cores by silicon quality and steering work to "preferred" cores -->
  <!-- - SMT → consider disabling if only benchmarking on physical cores —
    SMT causes shared execution resources (L1, ALUs) to contend -->
<!-- - Advanced → AMD CBS → NBIO
  - IOMMU → Disabled — removes DMA remapping overhead (only needed for
    virtualization/passthrough) -->

#### Kernel boot parameters

Add to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then run
`sudo update-grub` and reboot:

```text
isolcpus=2-15 nohz_full=2-15 rcu_nocbs=2-15
```

- `isolcpus=2-15` — removes cores 2-15 from the general scheduler so only
  explicitly pinned tasks run there (leaves cores 0-1 for the OS)
- `nohz_full=2-15` — disables the periodic timer tick on isolated cores,
  eliminating a source of regular interrupts
- `rcu_nocbs=2-15` — offloads RCU callbacks away from isolated cores,
  preventing kernel bookkeeping from interrupting benchmarks

#### Runtime setup (run before benchmarking)

```sh
# 0. Disable simultaneous multithreading (offline sibling hardware threads)
echo off | sudo tee /sys/devices/system/cpu/smt/control

# verify — should print 0
cat /sys/devices/system/cpu/smt/active

# 1. Switch amd-pstate from EPP to passive mode (hands control to cpufreq)
echo passive | sudo tee /sys/devices/system/cpu/amd_pstate/status

# 2. Set performance governor on all cores
sudo cpupower frequency-set -g performance

# 3. Disable boost (prevents unsustainable frequency spikes)
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

# 4. Pin all cores to base clock (4.3 GHz = sustained, below thermal throttle)
sudo cpupower frequency-set -f 4300000

# 5. Stop thermald if running (it will fight the above settings)
sudo systemctl stop thermald

# 6. Disable NMI watchdog (generates periodic interrupts on every core)
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog

# 7. Disable Transparent Huge Pages (THP compaction causes latency spikes)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# 8. Disable ASLR (address randomization causes layout-dependent variance)
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# 9. Allow access to hardware counters for PAPI/perf.
# This is a host kernel setting, so it also affects runs inside Docker containers.
sudo sysctl -w kernel.perf_event_paranoid=-1

# 10. Move IRQs away from benchmark cores (pin all IRQs to cores 0-1)
for irq_dir in /proc/irq/*/; do
    echo 3 | sudo tee "$irq_dir/smp_affinity" 2>/dev/null || true
done

# 11. Stop unnecessary services that cause background activity
sudo systemctl stop unattended-upgrades snapd cron atd 2>/dev/null || true

# 12. Verify — all cores should show ~4300 MHz
cat /proc/cpuinfo | grep "cpu MHz"
```

### Integrate within the measurement harness

Every new machine requires an entry in `src/autotuner/machines.py`.

Fill in `family`, `isa`, `display_name`, `target_triple`, `march`, and `mtune`.
The compiler fields must be accepted by the selected compiler. Set
`libxsmm_arch` only when the machine can run the corresponding
libxsmm-generated instructions.

For a Linux computer, we offer the possibility to access hardware counters (which are more precise than the monotonic clock) through the PAPI library. It is necessary to first install PAPI (on Ubuntu: `sudo apt install papi-tools libpapi-dev`), then configure the system to grant access to the counters with `sudo sysctl -w kernel.perf_event_paranoid=-1`. Finally, verify that the `PAPI_TOT_CYC` event is available using the command `papi_avail`. If it is, add `papi` to the machine's `libs` in `src/autotuner/machines.py`. If this value is too restrictive on the host (for example `4`), PAPI-based timers can fail with errors like `Event does not exist`, including when running inside Docker.

To obtain the `freq` and `peak_f32` keys, use the information generated by this script: <https://gitlab.inria.fr/CORSE/perf-fma>

The `freq` key is provided directly, and the `peak_f32` key is calculated using the formula: `vector_size * n_fma * 2`
