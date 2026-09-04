# g5k-tools

Grid'5000 resource queries for the xDSL autotuning experiments.

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

## Usage

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

## Tests

```sh
cd g5k-tools && uv run pytest
```

The tests drive a fake Grid'5000 API, so they need no credentials.
