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

### Setting up a new target

The scripts in this repository refer to the machine that the code is run on as the "target".

When running on a new machine, please create a `.env` file with the format:

```sh
TARGET=your_target_name_here
# Optional. Omit to use Snakemake's default ILP scheduler.
# On Apple Silicon, set greedy — PuLP's bundled CBC is x86_64-only.
# SNAKEMAKE_SCHEDULER=greedy
```

Then add a specification of the machine to the `targets` field in [[default.yaml]], and populate the `TESTSET` and `DATASET_VARIANTS` in the Snakefile.

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

Generate the data for the host platform by running `make dataset`. JSONL outputs are written under `data/<TARGET>/` (with `TARGET` from `.env` or `default.yaml`); filenames use `<dtype>.<dataset>.jsonl` (for example `f32.ttile.jsonl`). Build artifacts go under `build/<TARGET>/`.

[T-tile chart generation.](https://gitlab.inria.fr/ntollena/ics-experiments/-/tree/main/paper_versions/asplos/small_mm_figure_Gui?ref_type=heads)

### Plotting

Plot data using `make plots`, this command will fail if all the data necessary to generate the plots is not present, instead of running the data generation. PNGs are written under `plots/<machine>/` for each platform that has JSONL inputs in the repo (for example `neon`, `tower`, `pinocchio`); plotting does not depend on your current `TARGET`.

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

It's important for the cores to have predictable frequencies for a given target.

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

Every new hardware target requires an entry in the default.yaml file.

The `triple` and `arch` keys must be filled with the CPU manufacturer’s information.

For a Linux computer, we offer the possibility to access hardware counters (which are more precise than the monotonic clock) through the PAPI library. It is necessary to first install PAPI (on Ubuntu: `sudo apt install papi-tools libpapi-dev`), then configure the system to grant access to the counters with `sudo sysctl -w kernel.perf_event_paranoid=-1`. Finally, verify that the `PAPI_TOT_CYC` event is available using the command `papi_avail`. If it is, add 'papi' to the 'lib' key in default.yaml. If this value is too restrictive on the host (for example `4`), PAPI-based timers can fail with errors like `Event does not exist`, including when running inside Docker.

To obtain the `freq` and `peak_f32` keys, use the information generated by this script: <https://gitlab.inria.fr/CORSE/perf-fma>

The `freq` key is provided directly, and the `peak_f32` key is calculated using the formula: `vector_size * n_fma * 2`
