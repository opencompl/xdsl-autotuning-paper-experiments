# Detected machine profiles

The machines the paper reports are written out in
[`src/autotuner/machines.py`](../src/autotuner/machines.py) and stay there.
Every other machine — a Grid'5000 node above all, where the hardware is
whatever OAR handed out — is described by a `<name>.json` file in this
directory, and `machines.load_profiles` picks it up so that `MACHINE=<name>`
works everywhere a built-in name does.

Write one by asking the machine itself, on the machine itself:

```sh
uv run machine-profile --name <name>
```

`<name>` keys `data/<name>/`, `build/<name>/` and `plots/<name>/`. On Grid'5000
use the **cluster** name: its nodes are identical by definition, so the cluster
is the identity of the hardware, and `scripts/g5k-eval.sh` defaults to it.

A profile has two halves:

```json
{
  "machine":  { "march": "znver4", "isa": "avx512", "libs": ["papi"], ... },
  "detected": { "cpu_model": "...", "papi_usable": true, "peak_measured": 31.8, ... }
}
```

`machine` is the configuration the harness reads. `detected` is the evidence
for it — what the CPU said it was, whether hardware counters could actually be
read, what the clock was doing — and is ignored when loading. A profile is an
input to every measurement taken with it, so **commit it next to the data it
produced**; that is what makes a number traceable to the machine and the timing
mode it came from.

Detection is not a lookup table. `march` is whatever the compiler resolves
`-march=native` to, `isa` follows from the CPU flags, and `libs`/`env` follow
from a PAPI probe that runs the same `headers/perf.h` path the benchmarks use.

## peak_f32 comes later

`peak_f32` is the one field nothing can discover: the vector width is in the
CPU flags, but the number of FMA pipes behind it is not — Zen 4 and Zen 4c
advertise AVX-512 over a 256-bit datapath, and an Intel AVX-512 part may or may
not have its second FMA unit. A fresh profile leaves it at `0`, which
`plot_ttile` reads as "no peak known": those plots then show absolute
throughput rather than a percentage of peak. The two paper figures that are
percentages of peak by construction — `plot_squares` and `plot_grid` — refuse
to draw without it; everything else is unaffected.

Filling it in afterwards costs nothing but a rebuild of the rows. Edit the
profile:

```json
"peak_f32": 32,
```

then re-run `make dataset MACHINE=<name>`. The measurements themselves are
cached per kernel in `build/<name>/**/time.txt`, and `peak` only enters when
`evaluate` assembles the rows from them — so nothing is re-timed, and the new
rows are directly comparable with the machine they were compared against. The value is
`vector lanes * FMA pipes * 2` (see the note at the end of the top-level
README); 32 and 64 are the usual answers for an AVX-512 machine.

Overrides exist for what detection cannot reach — `--libxsmm-arch clx`,
`--peak-f32 N`, `--march`, `--freq`. Run `uv run machine-profile --help`.
