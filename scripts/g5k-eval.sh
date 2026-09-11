#!/usr/bin/env bash
# Run the evaluation on whatever machine we happen to be on, and leave the
# results, the profile they were taken with, and the evidence for both in
# /results.
#
# Meant as the container command of a g5k-run invocation. Nothing here names a
# CPU: the machine is named after the Grid'5000 cluster the job landed on, and
# everything about it is detected on the spot.
#
#   uv run g5k-run --microarch "zen 5" ... -- bash scripts/g5k-eval.sh
#
# Knobs.  PEAK is required; the rest are optional:
#   MACHINE   machine name (default: $G5K_CLUSTER, which g5k-run sets)
#   OUT       where to leave results (default: /results, which g5k-run fetches)
#   CORES     parallel code generation jobs (default: from RAM and thread count)
#   PIN_CPU   the core the timed runs are pinned to (default: 2, as `make docker-run`)
#   VALIDATE  check the kernels compute the right thing first (default: 1)
#   PEAK      f32 flops per cycle; required, because it cannot be detected and
#             every row of every dataset records it -- see below.  PEAK=0 says
#             the omission is deliberate and records no peak.

set -euo pipefail

# A container's hostname is its own id, so the node it is on has to be told to
# it: g5k-run passes G5K_CLUSTER/G5K_NODE/G5K_SITE/G5K_JOB_ID in.
MACHINE="${MACHINE:-${G5K_CLUSTER:-}}"
if [ -z "$MACHINE" ]; then
  echo "error: no MACHINE and no G5K_CLUSTER in the environment." >&2
  echo "Set MACHINE=<name> to name this machine's dataset." >&2
  exit 2
fi

OUT="${OUT:-/results}"
PIN_CPU="${PIN_CPU:-2}"
VALIDATE="${VALIDATE:-1}"
PEAK="${PEAK:-}"

# The one thing about a node that nothing here can detect.  `evaluate` writes
# `peak` into every row, so a run without it comes home with datasets the
# three % of peak figures -- the squares sweep, the nano-kernel grid, the
# baselines -- refuse to draw from, and those are the paper's.  Repairable, in
# that `peak` is one constant per dtype, but only by editing the jsonl by hand
# or by re-running `evaluate` where the build tree still is, which is the node
# the job has since given back (machines/README.md assumes the latter).  So
# ask in the first second instead.  And it is the *node's* number: 64 on a
# part that can only do 32 is not an error anywhere downstream, just figures
# at half scale, which is the reason to think about it here and not later.
if [ -z "$PEAK" ]; then
  cat >&2 <<'NOPEAK'
error: PEAK is unset.

Pass this node's f32 FLOP/cycle -- vector lanes * FMA pipes * 2:

  64   Skylake-SP, Cascade Lake, Ice Lake, Sapphire Rapids, Emerald Rapids,
       Zen 5 (two 512-bit FMA pipes)
  32   Zen 4, Zen 4c (512-bit instructions over a 256-bit datapath)

Check it against the node rather than against this list, which is only the
microarchitectures already run.  PEAK=0 records no peak on purpose: the
datasets are still collected, the % of peak figures are not drawable.
NOPEAK
  exit 2
fi

# One code generation job per two GiB, capped by the thread count: one worker
# per hardware thread on a many-core node can ask for more memory than the node
# has, and the default -- Snakemake's `--cores all`, `build-dataset`'s one
# worker per available core -- counts cores, not memory.
if [ -z "${CORES:-}" ]; then
  ram_gib=$(awk '/MemTotal/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
  threads=$(nproc)
  CORES=$(( ram_gib / 2 ))
  [ "$CORES" -gt "$threads" ] && CORES="$threads"
  [ "$CORES" -lt 1 ] && CORES=1
fi

mkdir -p "$OUT/sysinfo"

say() { echo; echo "=== $* ==="; }

say "machine '$MACHINE' on ${G5K_NODE:-unknown node} (job ${G5K_JOB_ID:-none})"
echo "cores=$CORES  pin=$PIN_CPU  validate=$VALIDATE  peak=$PEAK"

# --------------------------------------------------------------------------- #
# What machine is this?
# --------------------------------------------------------------------------- #

say "detecting the profile"
# Always passed now that it is required, so the profile records where the
# number came from even when it is the deliberate 0.
profile_args=(--name "$MACHINE" --peak-f32 "$PEAK")
uv run machine-profile "${profile_args[@]}"
# The profile is an input to every number that follows, so it travels with them.
cp "machines/$MACHINE.json" "$OUT/$MACHINE.json"

# --------------------------------------------------------------------------- #
# The evidence for it
# --------------------------------------------------------------------------- #

say "recording system information"
capture() {
  local name=$1
  shift
  "$@" > "$OUT/sysinfo/$name" 2>&1 || echo "(failed: $*)" >> "$OUT/sysinfo/$name"
}

capture lscpu.txt lscpu
capture lscpu.json lscpu -J
capture cpuinfo.txt cat /proc/cpuinfo
capture meminfo.txt cat /proc/meminfo
capture uname.txt uname -a
capture env.txt env
capture df.txt df -h
capture clang-version.txt clang --version
capture mlir-opt-version.txt mlir-opt --version
capture blis-version.txt pkg-config --modversion blis
capture packages.txt uv pip freeze

# The timing mode hinges on these, so they belong in the record even though the
# profile has already read them and decided.
capture perf-event-paranoid.txt cat /proc/sys/kernel/perf_event_paranoid
if command -v papi_avail >/dev/null 2>&1; then
  capture papi-avail.txt papi_avail -a
  capture papi-components.txt papi_component_avail
fi

# What the clock was actually doing, not what it was asked to do.
{
  for knob in scaling_governor scaling_driver scaling_cur_freq \
              cpuinfo_min_freq cpuinfo_max_freq base_frequency; do
    path="/sys/devices/system/cpu/cpu$PIN_CPU/cpufreq/$knob"
    [ -r "$path" ] && echo "cpu$PIN_CPU/$knob: $(cat "$path")"
  done
  for path in /sys/devices/system/cpu/cpufreq/boost \
              /sys/devices/system/cpu/intel_pstate/no_turbo \
              /sys/devices/system/cpu/smt/active; do
    [ -r "$path" ] && echo "$path: $(cat "$path")"
  done
  true
} > "$OUT/sysinfo/cpufreq.txt" 2>&1

# The source rsync excludes .git, so a revision file is the usual answer here.
if [ -r .g5k-revision ]; then
  cp .g5k-revision "$OUT/sysinfo/git-revision.txt"
else
  git rev-parse HEAD > "$OUT/sysinfo/git-revision.txt" 2>/dev/null || true
fi

# --------------------------------------------------------------------------- #
# The baseline whose code path cannot be predicted from CPU flags
# --------------------------------------------------------------------------- #

say "AOCL-BLAS smoke test (README: must not report a 'generic' code path)"
BLIS_ARCH_DEBUG=1 uv run snakemake --cores 1 --forceall \
  "build/$MACHINE/matmul_colmaj/16x3x5/aocl.f64.test.log" \
  --config "machine=$MACHINE" > "$OUT/sysinfo/aocl-arch.txt" 2>&1 || true
tail -n 15 "$OUT/sysinfo/aocl-arch.txt" || true

# --------------------------------------------------------------------------- #
# Measure
# --------------------------------------------------------------------------- #

if [ "$VALIDATE" = 1 ]; then
  # Check the kernels compute the right thing before spending hours timing
  # them. First, so the --forceall rebuild it does is not thrown-away work.
  say "validating (CORES=$CORES)"
  make dataset_validate MACHINE="$MACHINE" CORES="$CORES"
fi

say "generating code (CORES=$CORES)"
make dataset_code MACHINE="$MACHINE" CORES="$CORES"

# `evaluate` times the kernels one at a time; pinning is what `make docker-run`
# does with taskset, and for the same reason. `--no-build` because the code is
# already generated above -- taskset's affinity mask is inherited by the build
# pool's workers, so leaving the generation to this step would run all of it on
# the single pinned core.
say "measuring, pinned to cpu $PIN_CPU"
taskset -c "$PIN_CPU" make dataset MACHINE="$MACHINE" EVAL_FLAGS=--no-build

# --------------------------------------------------------------------------- #
# Hand it all back
# --------------------------------------------------------------------------- #

say "collecting into $OUT"
mkdir -p "$OUT/data"
# Not `cp -a`: that preserves ownership, and $OUT is the site's NFS home, which
# Grid'5000 exports with root_squash -- the container is root, so the chown is
# refused. The copy itself succeeds, but cp still exits non-zero, and `set -e`
# would end the run here, just short of everything below.
cp -R --preserve=timestamps "data/$MACHINE/." "$OUT/data/"

# What actually came out, in the log, so a short dataset is visible here
# rather than at plotting time days later.  `datasets.default_variants` gives
# an AVX-512 node the same comparison `rapper` measures, so this should read
# the same as rapper's: six datasets, and every variant of each present.
say "collected"
for dataset in "$OUT"/data/*.jsonl; do
  [ -e "$dataset" ] || continue
  printf '%-26s %6d rows  %s\n' \
    "$(basename "$dataset")" \
    "$(wc -l < "$dataset")" \
    "$(grep -o '"variant":"[^"]*"' "$dataset" | sort -u | sed 's/.*:"//;s/"$//' | paste -sd, -)"
done
# The per-kernel cycle counts behind the jsonl: cheap to keep, and the only way
# to spot a single outlier after the fact.
find "build/$MACHINE" -name 'time.txt' -print0 \
  | tar czf "$OUT/time-txt.tar.gz" --null -T - 2>/dev/null || true
cp -R --preserve=timestamps .snakemake/log "$OUT/snakemake-log" 2>/dev/null || true

find "$OUT" -maxdepth 2 | sort
echo
echo "done: $MACHINE"
