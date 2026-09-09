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
# Knobs, all optional:
#   MACHINE   machine name (default: $G5K_CLUSTER, which g5k-run sets)
#   OUT       where to leave results (default: /results, which g5k-run fetches)
#   CORES     parallel code generation jobs (default: from RAM and thread count)
#   PIN_CPU   the core the timed runs are pinned to (default: 2, as `make docker-run`)
#   VALIDATE  check the kernels compute the right thing first (default: 1)
#   PEAK      f32 flops per cycle, if known; left out, the ttile plots show
#             absolute throughput, the % of peak figures cannot be drawn, and
#             it can be filled into the profile later

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
echo "cores=$CORES  pin=$PIN_CPU  validate=$VALIDATE  peak=${PEAK:-unset}"

# --------------------------------------------------------------------------- #
# What machine is this?
# --------------------------------------------------------------------------- #

say "detecting the profile"
profile_args=(--name "$MACHINE")
[ -n "$PEAK" ] && profile_args+=(--peak-f32 "$PEAK")
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
cp -a "data/$MACHINE/." "$OUT/data/"
# The per-kernel cycle counts behind the jsonl: cheap to keep, and the only way
# to spot a single outlier after the fact.
find "build/$MACHINE" -name 'time.txt' -print0 \
  | tar czf "$OUT/time-txt.tar.gz" --null -T - 2>/dev/null || true
cp -a .snakemake/log "$OUT/snakemake-log" 2>/dev/null || true

find "$OUT" -maxdepth 2 | sort
echo
echo "done: $MACHINE"
