# On macs, the default target is "neon"
ifeq ($(shell uname -s),Darwin)
TARGET := neon
endif

ifneq ("$(wildcard .env)","")
	include .env
	export
endif

.PHONY: pytest
pytest:
	uv run pytest -W error

.PHONY: filecheck
filecheck:
	uv run lit -v --order=smart tests/filecheck

.PHONY: snakemake
snakemake:
	uv run snakemake --quiet all --cores all tests --forceall $(if $(TARGET),--config target=$(TARGET),)

.PHONY: tests
tests: pytest filecheck snakemake
	@echo "All tests passed successfully"
	@exit 0

.PHONY: dataset_code
dataset_code:
	uv run snakemake --quiet --cores all dataset_code $(if $(TARGET),--config target=$(TARGET),)

.PHONY: dataset_validate
dataset_validate:
	uv run snakemake --quiet --cores all dataset_validate --forceall $(if $(TARGET),--config target=$(TARGET),)

# --cores 1 to avoid contention issues when measuring performance
# re-run time measurement every time
.PHONY: dataset
dataset: dataset_code
	uv run snakemake --quiet --cores 1 dataset --forcerun time $(if $(TARGET),--config target=$(TARGET),)


# Prevent Make from deleting this intermediate file
.PRECIOUS: data/$(TARGET)/f64.bars.jsonl
data/$(TARGET)/f64.bars.jsonl:
	uv run snakemake --cores 1 $@ --config target=$(TARGET)

PLOTS =

PLOTS += plots/neon/f32.ttile.png
PLOTS += plots/neon/f64.ttile.png
PLOTS += plots/neon/f64.cube_8.png
PLOTS += plots/neon/f64.cube_16.png
PLOTS += plots/neon/f64.cube_64.png
# PLOTS += plots/neon/f64.ttile_squares.png
# PLOTS += plots/neon/f64.ttile_combined.png
# PLOTS += plots/neon/f64.heatmap.png

PLOTS += plots/tower/f32.ttile.png
PLOTS += plots/tower/f64.ttile.png
PLOTS += plots/tower/f64.cube_8.png
PLOTS += plots/tower/f64.cube_16.png
PLOTS += plots/tower/f64.cube_64.png
PLOTS += plots/tower/f64.ttile_squares.png
PLOTS += plots/tower/f64.ttile_combined.png
PLOTS += plots/tower/f64.heatmap.png

PLOTS += plots/pinocchio/f32.ttile.png
PLOTS += plots/pinocchio/f64.ttile.png
PLOTS += plots/pinocchio/f64.cube_8.png
PLOTS += plots/pinocchio/f64.cube_16.png
PLOTS += plots/pinocchio/f64.cube_64.png
PLOTS += plots/pinocchio/f64.ttile_squares.png
PLOTS += plots/pinocchio/f64.ttile_combined.png
PLOTS += plots/pinocchio/f64.heatmap.png

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

plots/%.cube_8.png: data/%.cube_8.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%.cube_16.png: data/%.cube_16.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%.cube_64.png: data/%.cube_64.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%.bars.png: data/%.bars.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%.heatmap.png: data/%.small_matrices.jsonl src/autotuner/plot_heatmap.py
	uv run plot-heatmap $< --output $@

.PHONY: plots
plots: $(PLOTS)

# set up all precommit hooks
.PHONY: precommit-install
precommit-install:
	uv run pre-commit install

# run all precommit hooks and apply them
.PHONY: precommit
precommit:
	uv run pre-commit run --all

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

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -rf build 2>/dev/null || true
