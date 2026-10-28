# g5k-tools

Grid'5000 resource queries and job submission for the xDSL autotuning
experiments: `g5k-availability` says which machines you can get, `g5k-run`
reserves one and runs a command in a Docker container on it, and `g5k-eval`
is the one command that runs the paper's evaluation on a named node.

This is a **separate uv project with its own venv**, deliberately not part of
the main `autotuner` environment: enoslib pins `rich ~= 12.0`, while the main
venv's `xdsl[dev]` needs `rich >= 13.3.3` through textual. Keeping the two
apart means neither has to be overridden.

## Setup

```sh
cd g5k-tools
direnv allow      # once: repoints UV_PROJECT_ENVIRONMENT at this venv
uv sync
```

> **Careful:** do not run `uv sync --project g5k-tools` from the repo root.
> The root `.envrc` exports `UV_PROJECT_ENVIRONMENT=<repo>/.venv` and direnv
> applies it to subdirectories, so uv would install enoslib into the *main*
> venv and evict xdsl from it. `g5k-tools/.envrc` overrides that, but only
> once you are inside the directory.

Grid'5000 API credentials go in `~/.python-grid5000.yaml`:

```yaml
username: <login>
password: <password>
```

or in the `G5K_USER` / `G5K_PASSWORD` environment variables. On a Grid'5000
frontend the anonymous connection is enough.

## Usage: g5k-availability

List every machine of a given microarchitecture across all sites, with the
earliest moment each one is free for an uninterrupted window:

```sh
cd g5k-tools
uv run g5k-availability                    # Zen 4 + Zen 4c, 1h slot
uv run g5k-availability --walltime 4:00:00
uv run g5k-availability --microarch "zen 5"
uv run g5k-availability --json
```

From anywhere, without depending on the environment at all:

```sh
g5k-tools/.venv/bin/g5k-availability
```

```
MACHINE  SITE    CLUSTER  MICROARCH  CORES/THREADS  RAM   GPU  QUEUE       FREE FOR 2:00:00
gres-1   nancy   gres     Zen 4      48/96          512G  1    production  now
gres-2   nancy   gres     Zen 4      48/96          512G  1    production  2026-09-04 14:27 (in 2h30m)
grdix-1  nancy   grdix    Zen 4c     128/256        128G  -    default     2026-09-06 13:57 (in 2d2h)
musa-1   sophia  musa     Zen 4      24/48          128G  -    default     unavailable (dead)
```

Notes on what the output means:

- The reference API spells microarchitectures with a space (`Zen 4`), and the
  dense EPYC 9754 parts are a distinct `Zen 4c`. The default `--microarch`
  matches both; the MICROARCH column says which you got.
- Besteffort jobs are ignored when computing availability, since OAR preempts
  them as soon as a regular job wants the node.
- QUEUE matters: production nodes are not in `default`, so reserving one needs
  an explicit `oarsub -q production`.
- Nodes that are dead, absent, or capped below the requested `--walltime` are
  reported as unavailable rather than sorted to the front.
- A node in `standby` has been powered down to save energy but is idle and
  reservable, so it counts as free; it shows as `now (standby)` because OAR has
  to boot it first, which costs a couple of minutes at the start of the job.

## Usage: g5k-eval

The evaluation, on one node, from one command:

```sh
# from anywhere in the repository
g5k-tools/.venv/bin/g5k-eval chirop-3
g5k-tools/.venv/bin/g5k-eval chirop-3 --walltime 3:00:00 --dry-run
g5k-tools/.venv/bin/g5k-eval --microarch "zen 5" --walltime 6:00:00
```

A node name is enough: the site comes from the reference API, so `chirop-3`,
`chirop-3.lille` and the full name all work, and `--cluster` or `--microarch`
choose a node instead. It prints what it decided before it does anything:

```
source:   /home/you/xdsl-autotuning-paper-experiments @ 9f58150...
image:    ghcr.io/opencompl/xdsl-autotuning-ci:0.33.0
machine:  chirop (2x Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz)
peak:     PEAK=64  [Ice Lake-SP: two 512-bit FMA pipes]
```

Then it stages this checkout on the site's home, reserves the node and runs
`scripts/g5k-eval.sh` in the container, streaming the output and fetching the
results into `g5k-runs/` — and finishes by printing the `cp` lines that fold
the run into the repository.

What it works out, so that it cannot be carried over from the last machine:

- **the image**: the semver of the `v*` tag this checkout describes to, which
  is the tag the publish workflow built the image from (`--image` overrides).
- **`PEAK`**, the node's f32 FLOP/cycle, from the microarchitecture and the
  SKU the reference API reports: 64 where there are two 512-bit FMA pipes, 32
  on Zen 4 and Zen 4c (AVX-512 over a 256-bit datapath) and on the Intel SKUs
  with one pipe (Bronze, Silver, Gold 5xxx). It is a guess from a table, which
  is why it is printed with the CPU it was made for; `--peak N` overrides it,
  and a microarchitecture the table does not know is an error rather than a
  default. Check it against the node — see `machines/README.md`.
- **turbo**: `intel_pstate`'s `no_turbo` or `cpufreq`'s `boost`, whichever
  that node has, decided on the node.
- **the revision**: `.g5k-revision`, since the staging rsync excludes `.git`.
  A dirty worktree is recorded as such, because the sha alone would claim
  more than the run can back up.

Everything else is g5k-run's, and the useful options are forwarded:
`--walltime`, `--queue`, `--wait`, `--reservation`, `--job-name`, `--dry-run`,
`--no-fetch`, `--fetch-to`, `--verbose`, and `--env` for the container's own
knobs (`--env VALIDATE=0`, `--env CORES=32`, `--env PIN_CPU=2`). An option it
does not name goes through attached, as `--run-arg=--poll=30`. The node is
held after the run, as `--keep` does, unless `--no-keep`; the source is staged
every time unless `--no-stage`.

`--dry-run` is the habit worth keeping: it prints the plan, the staging
command and the node script, and reserves nothing.

## Usage: g5k-run

Reserve one node and run a command in a Docker container on it. A single
invocation picks the node, reserves it through enoslib (OAR), installs Docker
with `g5k-setup-docker -t`, makes the image available, runs the container,
streams its output here, and rsyncs the run directory back:

```sh
cd g5k-tools

# print the plan and the script that would run on the node; reserves nothing
uv run g5k-run --dry-run --microarch "zen 4" -- lscpu

# take the soonest free Zen 4 node, pull the toolchain image, run something
uv run g5k-run --microarch "zen 4" --pull \
    --image ghcr.io/opencompl/xdsl-autotuning-ci:0.31.0 \
    -- bash -lc 'lscpu | head'
```

Anything after `--` is the command inside the container. `--dry-run` first is
the habit worth keeping: it shows the exact node script without touching OAR.

### Getting the image onto the node

Three ways, in decreasing order of how much they depend on the outside world:

```sh
# 1. a local `docker save` tarball, copied once to the site's NFS home
#    (shared with that site's nodes, so `docker load` on the node is local)
docker save -o /tmp/xdsl-autotuner.tar xdsl-autotuner
uv run g5k-run --cluster gros --image-tar /tmp/xdsl-autotuner.tar -- lscpu

# 2. a tarball already sitting on the site's home
uv run g5k-run --cluster gros --remote-image-tar g5k-run/images/xdsl-autotuner.tar -- lscpu

# 3. pulled on the node from a registry
uv run g5k-run --cluster gros --pull --image ghcr.io/opencompl/xdsl-autotuning-ci:0.31.0 -- lscpu
```

The tarball is only copied when the site's copy is missing or a different size
(`--force-push` overrides). With `--image-tar` you can leave `--image` out: the
name is read from what `docker load` reports. Note that Docker Hub rate-limits
pulls from Grid'5000's address space, which is the reason the tarball route
exists at all; ghcr.io is fine.

### Running the evaluation by hand

`g5k-eval` is the g5k-run invocation below, filled in. It is worth reading
once, and it is what to fall back to for a run that needs something the
wrapper does not offer.

`scripts/g5k-eval.sh` in the main repository is the container command for this.
Nothing in the invocation names a CPU: the machine is named after the cluster
OAR gave you, and everything about it is detected on the node
(`uv run machine-profile`, see `machines/README.md`). Changing
`--microarch "zen 4"` to `"zen 5"` or `"cascade lake"` is the whole difference
between machines.

The image is only the toolchain — the repository is bind-mounted into it, as in
`make docker-run` — so the source goes to the site's home first:

```sh
# once per site; .git is excluded, so leave the revision behind in a file
git rev-parse HEAD > .g5k-revision
rsync -az --delete --exclude .venv --exclude build --exclude .git \
    ./ <login>@access.grid5000.fr:nancy/xdsl-autotuning-paper-experiments/
```

```sh
uv run g5k-run --keep --microarch "zen 4" --walltime 6:00:00 \
    --pull --image ghcr.io/opencompl/xdsl-autotuning-ci:0.32.0 \
    --node-setup 'sudo sysctl -w kernel.perf_event_paranoid=-1' \
    --node-setup 'echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null' \
    --node-setup 'echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null' \
    --node-setup 'sudo chown -R $(id -u):$(id -g) /tmp/eval || true' \
    --node-setup 'rsync -a --delete --exclude build $HOME/xdsl-autotuning-paper-experiments/ /tmp/eval/' \
    --mount /tmp/eval:/src --workdir /src \
    --env IN_DOCKER=1 --env SNAKEMAKE_SCHEDULER=greedy \
    --docker-arg=--cap-add=SYS_ADMIN --docker-arg=--cap-add=PERFMON \
    --docker-arg=--security-opt --docker-arg=seccomp=unconfined \
    --docker-arg=--pid=host \
    -- bash scripts/g5k-eval.sh
```

Extra `docker run` arguments must be written attached (`--docker-arg=--pid=host`)
so argparse does not mistake them for options of ours.

`--keep` earns its place here: the run is hours long, and holding the node means
a failure late in it costs neither the reservation nor the warm image, and the
retry lands on the same node so the second attempt is comparable to the first.

#### Why those `--node-setup` lines

They are the things a container cannot do for itself, and they are the ones
`g5k-eval` passes:

- `kernel.perf_event_paranoid` is **not** namespaced, so it cannot be written
  from inside the container however privileged it is. Without it PAPI cannot
  read `PAPI_TOT_CYC`, and the harness falls back to the monotonic clock scaled
  by a nominal frequency. That matters more than it looks: the datasets record
  *cycles*, and `peak` is in flops per cycle, so with counters the numbers do
  not depend on what the clock did. `machine-profile` probes this and records
  which mode it got.
- The governor and boost settings fix the clock. Everything in the main
  README's "Disabling Frequency Switching" section except `isolcpus` and
  `nohz_full` is reachable through sysfs as root; those two would need a
  `kadeploy` job with its own kernel command line.
- The last two lines stage the source on the node's **local** disk. `build/` is
  tens of thousands of small files, and writing that from a hundred parallel
  jobs onto the site's NFS server is both slow and antisocial. Results still
  come back, because the container copies them into `/results`, which is on the
  NFS home.

  They are two lines because of who owns what. The container runs as root, so
  everything it wrote into `/tmp/eval` last time -- `build/`, `.snakemake/`,
  `data/<cluster>/`, the detected `machines/<cluster>.json` -- is root's, and
  the rsync, which runs as you, cannot delete it: a re-run on a held node fails
  in the setup with `delete_file: unlink(...) failed: Permission denied`. The
  `chown` hands it back. `--exclude build` then keeps the build cache across
  re-runs instead of deleting it for want of a copy in the source, which is
  safe because `build.py` keys every artifact on a digest of the generator that
  produced it -- a changed generator rebuilds, an unchanged one is reused. Drop
  the exclude, or `sudo rm -rf /tmp/eval/build`, for a clean rebuild.

`sudo` works on a reserved node because `g5k-setup-docker` has already called
`sudo-g5k` by the time these run.

#### What the container does

`scripts/g5k-eval.sh`, in order: detects the profile and writes it to
`/results/<cluster>.json`; records the system information it was detected from
into `/results/sysinfo/`; runs the AOCL-BLAS smoke test from the main README;
validates the kernels; generates code in parallel; then measures with the
timing pinned to one core, as `make docker-run` does with `taskset`; finally
it prints the datasets it collected, one line each with the row count and the
variants, which on an AVX-512 node is the comparison `rapper` is configured
for — check it against `datasets.VARIANTS`, not against `data/rapper/`, which
lags the definition whenever a sweep is widened and rapper has not re-run
since. Knobs are environment variables — `MACHINE`, `CORES`, `PIN_CPU`,
`VALIDATE`, `PEAK` — passed with `--env`.

`PEAK` is the one it refuses to start without: the node's f32 FLOP/cycle is
not detectable, `evaluate` writes it into every row, and the three % of peak
figures cannot be drawn from a dataset that records none. It is 64 on the
Intel AVX-512 parts and on Zen 5, and 32 on Zen 4 and Zen 4c, whose AVX-512 is
256 bits wide underneath — so it has to be decided per cluster, not carried
over from the last run. `PEAK=0` runs anyway and records no peak.

Two of its defaults are worth knowing. `CORES` is derived from the node's RAM
rather than its thread count, because the default of one worker per available
core starts 512 `xdsl-opt` processes on a 512-thread node, against however much
memory that node happens to have. And `MACHINE` defaults to `$G5K_CLUSTER`: a container's own hostname
is its container id, so `g5k-run` passes `G5K_NODE`, `G5K_CLUSTER`, `G5K_SITE`
and `G5K_JOB_ID` in.

#### Afterwards

The run directory comes back with `results/data/*.jsonl`, the profile, and
`results/sysinfo/`. To fold it into the repository:

```sh
cp g5k-runs/<run_id>/results/<cluster>.json ../machines/
mkdir -p ../data/<cluster>
cp g5k-runs/<run_id>/results/data/*.jsonl ../data/<cluster>/
(cd .. && make plots-machine MACHINE=<cluster>)
```

`g5k-eval` prints these three lines with the run and the cluster filled in,
and without the `../`: it works from the repository root, so the runs land in
`<root>/g5k-runs` whichever directory you start it in.

Commit the profile next to the data: it is the input those numbers were taken
with, including which timing mode was in force. `peak_f32` can be filled into
it later without re-measuring — see `machines/README.md`.

#### Machines worth running

`default.yaml` defines the xdsl backend and compxsmm pass pipelines under an
`avx512` key only, so on a node without AVX-512 our own variants cannot be
generated at all and the dataset is baselines only (the Snakefile derives this
rather than failing). The machines that exercise the paper's code paths are the
AVX-512 ones — Zen 4, Zen 4c, Zen 5, Skylake-SP, Cascade Lake, Ice Lake,
Sapphire Rapids, Emerald Rapids. `g5k-availability --microarch "<name>"` says
which of them you can get.

### What lands where

Everything for a run lives in one directory on the site's home,
`<--remote-dir>/<timestamp>-<node>/`, which is rsynced into `<--fetch-to>` at
the end (`--no-fetch` leaves it there):

| file | what it is |
| --- | --- |
| `run.sh` | the script that ran on the node |
| `run.log` | everything the setup and the container printed |
| `exit_code` | the container's exit status |
| `meta.json` | node, job, image, command, walltime, `--node-setup` |
| `node.json` | the reference API's description of the node it ran on |
| `results/` | mounted at `/results` in the container |

`node.json` is the independent account of the hardware — exact CPU stepping,
cache sizes, DIMM layout, BIOS version — which is what a profile detected from
inside a container can be checked against.

The container's exit code is also this command's exit code, so `g5k-run` can be
chained. While a run is going you can follow it from the frontend with
`tail -f` on `run.log`; the run is detached from our SSH session (Ansible's own
detached mode, not a hand-rolled `nohup`), so a dropped connection does not
kill it — if the node stops answering, the poll says so once and keeps
retrying, and the results are fetched either way.

`run.log` appearing on the node is checked right after the launch: if the
script never started, that is an error here rather than a silent wait.

### Not piling up reservations

- The job is *named* (`--job-name`, by default `g5k-run-<node or cluster>`), and
  enoslib reloads a job of that name instead of submitting a second one. A
  re-run therefore lands on the same node instead of taking another one.
- The job is deleted as soon as the command finishes. `--keep` holds it until
  the walltime instead, which is the fast path while iterating: Docker stays
  installed and the image stays loaded, so the next run starts in seconds.
  Delete it with `oardel <id>` on the frontend when done.
- Ctrl-C stops the container and releases the node.
- `--microarch` refuses to submit unless a matching node is free *right now*;
  `--wait` queues the job anyway, `--reservation` books a date.
- One node per run, on purpose. Nothing here fans out over several machines.

Two consequences of how enoslib reserves are worth knowing:

- It grants itself root with `sudo-g5k` while reserving, which tags the job and
  makes Grid'5000 **reinstall the node when the job ends**. The node therefore
  shows up as `absent` for a few minutes after every run — expected, not a
  failure, but another reason to iterate with `--keep` rather than one job per
  attempt.
- A node woken from `standby`, or one still coming back from that reinstall,
  can accept an SSH connection before it has settled, so the tool waits for the
  node to answer before it does anything on it, and `--microarch` prefers a
  node that is already `alive` over one OAR would have to boot.

Selection is either explicit or by microarchitecture:

```sh
uv run g5k-run --server gres-1.nancy ...    # this exact node
uv run g5k-run --cluster gros ...           # any node of a cluster
uv run g5k-run --microarch "zen 5" ...      # soonest free node with that microarch
```

The OAR queue is worked out from the node (production clusters need
`-q production`, which is easy to forget by hand); `--queue` overrides it.

Commands run on the node as your Grid'5000 user, not root, so `$HOME` is the
site's NFS home and the files the container writes stay yours. `g5k-setup-docker`
does its own privilege escalation.

## Tests

```sh
cd g5k-tools && uv run pytest
```

The tests need no credentials: they cover the pure parts, above all the node
script and the log-polling protocol `g5k-run` speaks.
