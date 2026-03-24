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
	uv run snakemake --cores all tests --forceall $(if $(TARGET),--config target=$(TARGET),)

.PHONY: tests
tests: pytest filecheck snakemake
	@echo "All tests passed successfully"
	@exit 0

.PHONY: dataset_code
dataset_code:
	uv run snakemake --quiet --cores all dataset_code $(if $(TARGET),--config target=$(TARGET),)

# --cores 1 to avoid contention issues when measuring performance
# re-run time measurement every time
.PHONY: dataset
dataset: dataset_code
	uv run snakemake --quiet --cores 1 dataset --forcerun time $(if $(TARGET),--config target=$(TARGET),)


# Prevent Make from deleting this intermediate file
.PRECIOUS: data/$(TARGET)/bars.%.jsonl
data/$(TARGET)/bars.%.jsonl:
	uv run snakemake --cores 1 $@ $(if $(TARGET),--config target=$(TARGET),)

PLOTS =

PLOTS += plots/neon/ttile.f32.png
PLOTS += plots/neon/ttile.f64.png
PLOTS += plots/neon/cube_8.f64.png
PLOTS += plots/neon/cube_16.f64.png
PLOTS += plots/neon/cube_64.f64.png
# PLOTS += plots/neon/tiny.f64.png
# PLOTS += plots/neon/heatmap.f64.png

PLOTS += plots/tower/ttile.f32.png
PLOTS += plots/tower/ttile.f64.png
PLOTS += plots/tower/cube_8.f64.png
PLOTS += plots/tower/cube_16.f64.png
PLOTS += plots/tower/cube_64.f64.png
PLOTS += plots/tower/tiny.f64.png
PLOTS += plots/tower/heatmap.f64.png

PLOTS += plots/pinocchio/ttile.f32.png
PLOTS += plots/pinocchio/ttile.f64.png
PLOTS += plots/pinocchio/cube_8.f64.png
PLOTS += plots/pinocchio/cube_16.f64.png
PLOTS += plots/pinocchio/cube_64.f64.png
PLOTS += plots/pinocchio/tiny.f64.png
PLOTS += plots/pinocchio/heatmap.f64.png

PLOTS += plots/ttile.pdf

plots/%/ttile.f32.png: data/%/ttile.f32.jsonl src/autotuner/plot_ttile.py
	uv run plot-ttile $< --output $@

plots/%/ttile.f64.png: data/%/ttile.f64.jsonl src/autotuner/plot_ttile.py
	uv run plot-ttile $< --output $@

plots/ttile.pdf: data/tower/ttile.f32.jsonl data/tower/ttile.f64.jsonl data/pinocchio/ttile.f32.jsonl data/pinocchio/ttile.f64.jsonl src/autotuner/plot_ttile.py
	uv run plot-ttile --output $@

plots/%/cube_8.f64.png: data/%/cube_8.f64.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%/cube_16.f64.png: data/%/cube_16.f64.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%/cube_64.f64.png: data/%/cube_64.f64.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/neon/bars.%.png: data/neon/bars.%.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/tower/bars.%.png: data/tower/bars.%.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/pinocchio/bars.%.png: data/pinocchio/bars.%.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/%/tiny.f64.png: data/%/small_matrices.f64.jsonl src/autotuner/plot_small_matrices.py
	uv run plot-tiny-line $< --output $@

plots/%/heatmap.f64.png: data/%/small_matrices.f64.jsonl src/autotuner/plot_small_matrices.py
	uv run plot-tiny-heatmap $< --output $@

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
