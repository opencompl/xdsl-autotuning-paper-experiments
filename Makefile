# On macs, the default machine is "neon"
ifeq ($(shell uname -s),Darwin)
MACHINE := neon
endif

ifneq ("$(wildcard .env)","")
	include .env
	export
endif

# Empty by default (Snakemake uses its ILP scheduler). Override in .env, e.g.
# SNAKEMAKE_SCHEDULER=greedy — needed on Apple Silicon where PuLP's bundled CBC is x86_64-only.
SCHEDULER_FLAG = $(if $(SNAKEMAKE_SCHEDULER),--scheduler $(SNAKEMAKE_SCHEDULER),)

.PHONY: pytest
pytest:
	uv run pytest -W error

.PHONY: filecheck
filecheck:
	uv run lit -v --order=smart tests/filecheck

.PHONY: snakemake
snakemake:
	uv run snakemake tests $(SCHEDULER_FLAG) --quiet all --cores all --forceall $(if $(MACHINE),--config machine=$(MACHINE),)

.PHONY: tests
tests: pytest filecheck snakemake
	@echo "All tests passed successfully"
	@exit 0

.PHONY: dataset_code
dataset_code:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all dataset_code $(if $(MACHINE),--config machine=$(MACHINE),)

.PHONY: dataset_validate
dataset_validate:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all dataset_validate --forceall $(if $(MACHINE),--config machine=$(MACHINE),)

# --cores 1 to avoid contention issues when measuring performance.
# Run `make clean` to re-measure everything.
# Run `make clean-ours` to re-measure just our code.
.PHONY: dataset
dataset: dataset_code
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 dataset $(if $(MACHINE),--config machine=$(MACHINE),)


# Prevent Make from deleting this intermediate file
.PRECIOUS: data/$(MACHINE)/f64.bars.jsonl
data/$(MACHINE)/f64.bars.jsonl:
	uv run snakemake $(SCHEDULER_FLAG) --cores 1 $@ --config machine=$(MACHINE)

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
	find build -name 'compxsmm_kdot.*' -exec rm -f {} + 2>/dev/null || true
	find build -name 'asm_kdot.*' -exec rm -f {} + 2>/dev/null || true
	rm -f data/$(MACHINE)/*

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -rf build 2>/dev/null || true
	rm -f data/$(MACHINE)/*
