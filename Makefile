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

# One process, one task per shape, every core busy -- see src/autotuner/build.py.
.PHONY: dataset_code
dataset_code:
	uv run build-dataset $(if $(MACHINE),--machine $(MACHINE),)

.PHONY: dataset_validate
dataset_validate:
	uv run snakemake $(RATE_FLAG) $(SCHEDULER_FLAG) $(PROGRESS_FLAG) --cores all dataset_validate --forceall $(if $(MACHINE),--config machine=$(MACHINE),)

# --cores 1 to avoid contention issues when measuring performance.
# Run `make clean` to re-measure everything.
# Run `make clean-ours` to re-measure just our code.
# `evaluate` drives the whole thing: it has Snakemake build the kernels across
# every core, then times them one at a time, then writes each dataset's jsonl.
# Datasets sharing a shape build and measure it once.
.PHONY: dataset
dataset:
	uv run evaluate $(if $(MACHINE),--machine $(MACHINE),)

# Prevent Make from deleting this intermediate file
.PRECIOUS: data/$(MACHINE)/f64.bars.jsonl
data/$(MACHINE)/f64.bars.jsonl:
	uv run snakemake $(RATE_FLAG) $(SCHEDULER_FLAG) --cores 1 $@ --config machine=$(MACHINE)

PLOTS =

PLOTS += plots/neon/f32.ttile.png
PLOTS += plots/neon/f64.ttile.png
# PLOTS += plots/neon/f64.ttile_squares.png
# PLOTS += plots/neon/f64.ttile_combined.png
# PLOTS += plots/neon/f64.heatmap.png

PLOTS += plots/tower/f32.ttile.png
PLOTS += plots/tower/f64.ttile.png
PLOTS += plots/tower/f64.ttile_squares.png
PLOTS += plots/tower/f64.ttile_combined.png
PLOTS += plots/tower/f64.heatmap.png

# PLOTS += plots/pinocchio/f32.ttile.png
# PLOTS += plots/pinocchio/f64.ttile.png
# PLOTS += plots/pinocchio/f64.ttile_squares.png
# PLOTS += plots/pinocchio/f64.ttile_combined.png
# PLOTS += plots/pinocchio/f64.heatmap.png

PLOTS += plots/rapper/f32.ttile.png
PLOTS += plots/rapper/f64.ttile.png
PLOTS += plots/rapper/f64.ttile_squares.png
PLOTS += plots/rapper/f64.ttile_combined.png
PLOTS += plots/rapper/f64.heatmap.png

PLOTS += plots/ttile.pdf

# `%` is e.g. neon/f32 or tower/f64 (dtype first in the basename)
plots/%.ttile.png: data/%.ttile.jsonl src/autotuner/plot_ttile.py
	uv run plot-ttile $< --output $@

plots/ttile.pdf: data/tower/f32.ttile.jsonl data/tower/f64.ttile.jsonl data/pinocchio/f32.ttile.jsonl data/pinocchio/f64.ttile.jsonl src/autotuner/plot_ttile.py
	uv run plot-ttile --output $@

plots/%.ttile_squares.png: data/%.small_matrices.jsonl src/autotuner/plot_ttile_squares.py
	uv run plot-ttile-squares $< --output $@

plots/%.ttile_combined.png: data/%.small_matrices.jsonl src/autotuner/plot_ttile_combined.py
	uv run plot-ttile-combined $< --output $@

plots/%.heatmap.png: data/%.small_matrices.jsonl src/autotuner/plot_heatmap.py
	uv run plot-heatmap $< --output $@

.PHONY: plots
plots: $(PLOTS)

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
	find build -name 'compxsmm.*' -exec rm -f {} + 2>/dev/null || true
	rm -f data/$(MACHINE)/*

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -rf build 2>/dev/null || true
	rm -f data/$(MACHINE)/*
