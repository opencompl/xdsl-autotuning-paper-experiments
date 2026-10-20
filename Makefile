# On macs, the default machine is "neon"
ifeq ($(shell uname -s),Darwin)
MACHINE := neon
endif

ifneq ("$(wildcard .env)","")
	include .env
	export
endif

# Snakemake's main loop dispatches ~120 jobs/s whichever scheduler is chosen
# (measured on rapper: greedy 126/s, ILP 119/s), so on a DAG of sub-second jobs
# it, not the machine, is the bottleneck -- which is why the datasets are built
# by `build-dataset` instead.  Greedy still wins slightly here, and it is the
# only option on Apple Silicon, where PuLP's bundled CBC is x86_64-only.
# Override with SNAKEMAKE_SCHEDULER=ilp in .env.
SNAKEMAKE_SCHEDULER ?= greedy
SCHEDULER_FLAG = --scheduler $(SNAKEMAKE_SCHEDULER)

# only use rich logging in interactive terminal
ifeq ($(MAKE_TERMOUT),)
PROGRESS_FLAG = --quiet rules host reason
else
PROGRESS_FLAG = --logger rich
endif

RATE_FLAG = --max-jobs-per-timespan 100000/1s

.PHONY: pytest
pytest:
	uv run pytest -W error

.PHONY: filecheck
filecheck:
	uv run lit -v --order=smart tests/filecheck

.PHONY: snakemake
snakemake:
	uv run snakemake tests $(RATE_FLAG) $(SCHEDULER_FLAG) --quiet all --cores all --forceall $(if $(MACHINE),--config machine=$(MACHINE),)

.PHONY: tests
tests: pytest filecheck snakemake
	@echo "All tests passed successfully"
	@exit 0

# Code generation is embarrassingly parallel, but one job per hardware thread
# is not always the right number: on a many-core node that many `xdsl-opt`
# processes can want more memory than the machine has, and neither Snakemake's
# `--cores all` nor `build-dataset`'s worker count knows that. Cap both with
# CORES=N -- scripts/g5k-eval.sh derives one from the node's RAM.
CORES ?= all
JOBS_FLAG = $(if $(filter-out all,$(CORES)),--jobs $(CORES),)

# One process, one task per shape, every core busy -- see src/autotuner/build.py.
.PHONY: dataset_code
dataset_code:
	uv run build-dataset $(JOBS_FLAG) $(if $(MACHINE),--machine $(MACHINE),)

# Builds the same kernels against the test harness instead of the timing one,
# then runs them all -- see src/autotuner/validate.py.
.PHONY: dataset_validate
dataset_validate:
	uv run validate-dataset $(JOBS_FLAG) $(if $(MACHINE),--machine $(MACHINE),)

# --cores 1 to avoid contention issues when measuring performance.
# Run `make clean` to re-measure everything.
# Run `make clean-ours` to re-measure just our code.
# `evaluate` drives the whole thing: it builds the kernels across every core,
# then times them one at a time, then writes each dataset's jsonl.  Datasets
# sharing a shape build and measure it once.  EVAL_FLAGS passes the rest of its
# options through -- `EVAL_FLAGS=--no-build` when the code is already generated
# and only the serialised timing should run, which is what pinning the whole
# target to one core (scripts/g5k-eval.sh) needs.
.PHONY: dataset
dataset:
	uv run evaluate $(JOBS_FLAG) $(EVAL_FLAGS) $(if $(MACHINE),--machine $(MACHINE),)

# Prevent Make from deleting this intermediate file
.PRECIOUS: data/$(MACHINE)/f64.bars.jsonl
data/$(MACHINE)/f64.bars.jsonl:
	uv run snakemake $(RATE_FLAG) $(SCHEDULER_FLAG) --cores 1 $@ --config machine=$(MACHINE)

PLOTS =

# PLOTS += plots/neon/f64.ttile_squares.png
# PLOTS += plots/neon/f64.ttile_combined.png
# PLOTS += plots/neon/f64.heatmap.png

PLOTS += plots/tower/f64.ttile_squares.png
PLOTS += plots/tower/f64.ttile_combined.png
PLOTS += plots/tower/f64.heatmap.png
PLOTS += plots/f64.squares.tower.pdf

# PLOTS += plots/pinocchio/f64.ttile_squares.png
# PLOTS += plots/pinocchio/f64.ttile_combined.png
# PLOTS += plots/pinocchio/f64.heatmap.png

PLOTS += plots/rapper/f64.ttile_squares.png
PLOTS += plots/rapper/f64.ttile_combined.png
PLOTS += plots/rapper/f64.heatmap.png
PLOTS += plots/f64.squares.rapper.pdf

# The grid is not committed until a machine has actually run it, so plot
# whichever machines have the data.  Naming them outright breaks `make plots`
# everywhere else: a pattern rule whose prerequisite cannot be built is
# "No rule to make target".
PLOTS += $(foreach m,$(patsubst data/%/f64.nanokernel_grid.jsonl,%,$(wildcard data/*/f64.nanokernel_grid.jsonl)),plots/f64.nanokernel_grid.$(m).pdf)

# The f32 counterpart of the squares figure, for the machines whose f32 sweep
# has actually been run with every implementation it draws.  Same reason the
# grid is globbed rather than named -- a target whose data is absent stops
# `make plots` everywhere -- except that here the file can also be present and
# stale: f32 gained CompXSMM - RA and CompXSMM + narrow after it was last
# swept, and `plot-squares` refuses a dataset missing a variant rather than
# quietly drawing a thinner figure.  So this tests the contents, not the name,
# and a machine joins the list once it has re-run `make dataset`.
F32_SQUARES_READY = $(patsubst data/%/f32.squares.jsonl,%,\
    $(shell grep -l compxsmm_plusnarrow data/*/f32.squares.jsonl 2>/dev/null))
PLOTS += $(foreach m,$(F32_SQUARES_READY),plots/f32.squares.$(m).pdf)

# One paper figure per machine, its two data types side by side, with the
# machine in the file name rather than in the figure.  The PDF is what LaTeX
# includes; the PNG is the same figure, in the machine's directory with the
# other PNGs, for looking at outside the paper.
PLOTS += plots/baselines.tower.pdf
PLOTS += plots/tower/baselines.png
PLOTS += plots/baselines.rapper.pdf
PLOTS += plots/rapper/baselines.png

BASELINES_SRC = src/autotuner/plot_baselines.py src/autotuner/plot_style.py

# A paper figure, so a PDF rather than a PNG, straight in plots/ with the
# machine last in the name; here `%` is the machine on its own.  One machine per
# figure: its two data types are the two panels, each a square sweep.
plots/baselines.%.pdf: data/%/f32.squares.jsonl data/%/f64.squares.jsonl $(BASELINES_SRC)
	uv run plot-baselines data/$*/f32.squares.jsonl data/$*/f64.squares.jsonl --output $@

# The same figure as a PNG, which is not a paper file, so it goes in the
# machine's directory the way the other PNGs do; here `%` is the machine.
plots/%/baselines.png: data/%/f32.squares.jsonl data/%/f64.squares.jsonl $(BASELINES_SRC)
	uv run plot-baselines data/$*/f32.squares.jsonl data/$*/f64.squares.jsonl --output $@

# The plots one machine's data supports, derived from the data that is actually
# there. This is what a machine with no entry above -- a Grid'5000 cluster --
# gets plotted with: `make plots-machine MACHINE=<name>`.
MACHINE_SMALL = $(wildcard data/$(MACHINE)/*.small_matrices.jsonl)
MACHINE_SQUARES = $(wildcard data/$(MACHINE)/f64.squares.jsonl)
MACHINE_GRID = $(wildcard data/$(MACHINE)/f64.nanokernel_grid.jsonl)
MACHINE_PLOTS  = $(patsubst data/%.small_matrices.jsonl,plots/%.ttile_squares.png,$(MACHINE_SMALL))
MACHINE_PLOTS += $(patsubst data/%.small_matrices.jsonl,plots/%.ttile_combined.png,$(MACHINE_SMALL))
MACHINE_PLOTS += $(patsubst data/%.small_matrices.jsonl,plots/%.heatmap.png,$(MACHINE_SMALL))
# The paper figures name the machine last, so these two do not fall out of a
# `data/%` patsubst the way the others do.
MACHINE_PLOTS += $(patsubst data/%/f64.squares.jsonl,plots/f64.squares.%.pdf,$(MACHINE_SQUARES))
MACHINE_PLOTS += $(patsubst data/%/f64.nanokernel_grid.jsonl,plots/f64.nanokernel_grid.%.pdf,$(MACHINE_GRID))

plots/%.ttile_squares.png: data/%.small_matrices.jsonl src/autotuner/plot_ttile_squares.py
	uv run plot-ttile-squares $< --output $@

plots/%.ttile_combined.png: data/%.small_matrices.jsonl src/autotuner/plot_ttile_combined.py
	uv run plot-ttile-combined $< --output $@

plots/%.heatmap.png: data/%.small_matrices.jsonl src/autotuner/plot_heatmap.py
	uv run plot-heatmap $< --output $@

SQUARES_SRC = src/autotuner/plot_squares.py src/autotuner/plot_style.py

# A paper figure, so a PDF rather than a PNG: LaTeX gets the vector text.  It
# goes straight in plots/ with the machine last in the name, the way the paper
# includes it; here `%` is the machine on its own.  One rule per data type
# rather than one shared rule: a pattern rule gets a single `%`, and the
# machine has it, so the dtype has to be spelled out in the target.
plots/f64.squares.%.pdf: data/%/f64.squares.jsonl $(SQUARES_SRC)
	uv run plot-squares $< --output $@

# The f32 counterpart, drawing the same six curves against the f32 peak, which
# the dataset carries per row.  `plot-squares` refuses a dataset missing one of
# them rather than quietly drawing a thinner figure, so this target fails until
# the machine has swept f32 with CompXSMM - RA and CompXSMM + narrow in it --
# which is what the readiness test above keeps out of `make plots`.
plots/f32.squares.%.pdf: data/%/f32.squares.jsonl $(SQUARES_SRC)
	uv run plot-squares $< --output $@

# A grid of K sweeps over the nano-kernels: sixteen M values down the rows
# against seven N values across, so M keeps the y axis and the figure comes out
# one column wide and tall.  A PDF rather than a PNG so LaTeX gets the vector
# text at that size.  A paper figure, so like the squares plot it goes straight
# in plots/ with the machine last in the name; here `%` is the machine alone.
plots/f64.nanokernel_grid.%.pdf: data/%/f64.nanokernel_grid.jsonl src/autotuner/plot_grid.py src/autotuner/plot_style.py
	uv run plot-grid $< --output $@

.PHONY: plots
plots: $(PLOTS)

.PHONY: plots-machine
plots-machine:
	@test -n "$(MACHINE)" || { echo "set MACHINE=<name>"; exit 1; }
	@test -n "$(strip $(MACHINE_PLOTS))" || { echo "no data/$(MACHINE)/*.jsonl to plot"; exit 1; }
	$(MAKE) $(MACHINE_PLOTS)

# set up all precommit hooks
.PHONY: precommit-install
precommit-install:
	uv run prek install

# run all precommit hooks and apply them
.PHONY: precommit
precommit:
	uv run prek run --all-files

# build docker image
.PHONY: docker-build
docker-build:
	docker build -t xdsl-autotuner . --platform linux/amd64

# pull and tag the CI image locally
.PHONY: docker-pull
docker-pull:
	docker pull --platform linux/amd64 ghcr.io/opencompl/xdsl-autotuning-ci:latest
	docker tag ghcr.io/opencompl/xdsl-autotuning-ci:latest xdsl-autotuner

# run docker image
# Arjun's tip:
# Call nice to make your processor not nice (It won't let other processes run)
# Pin to core 2
# To run nice without sudo, add `your_username - nice -20` to `etc/security/limits.conf`
.PHONY: docker-run
docker-run:
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		docker run --platform linux/amd64 -v .:/src -ti xdsl-autotuner; \
	else \
		nice -n -15 taskset -c 2 docker run -e IN_DOCKER=1 --platform linux/amd64 --cap-add=SYS_ADMIN --cap-add=PERFMON --security-opt seccomp=unconfined --security-opt apparmor=unconfined --pid=host -v .:/src -ti xdsl-autotuner; \
	fi

.PHONY: docker-run-fast
docker-run-fast:
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		docker run --platform linux/amd64 -v .:/src -ti xdsl-autotuner; \
	else \
		nice -n -15 docker run -e IN_DOCKER=1 --platform linux/amd64 --cap-add=SYS_ADMIN --cap-add=PERFMON --security-opt seccomp=unconfined --security-opt apparmor=unconfined --pid=host -v .:/src -ti xdsl-autotuner; \
	fi

.PHONY: clean-ours
clean-ours:
	find build -name 'xdsl_libxsmm.*' -exec rm -f {} + 2>/dev/null || true
	find build -name 'compxsmm*' -exec rm -f {} + 2>/dev/null || true
	rm -f data/$(MACHINE)/*

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -rf build 2>/dev/null || true
	rm -f data/$(MACHINE)/*
