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
	uv run snakemake --cores all dataset_code $(if $(TARGET),--config target=$(TARGET),)

# --cores 1 to avoid contention issues when measuring performance
# re-run time measurement every time
.PHONY: dataset
dataset: dataset_code
	uv run snakemake --cores 1 dataset --forcerun time $(if $(TARGET),--config target=$(TARGET),)


# Prevent Make from deleting this intermediate file
.PRECIOUS: data/bars.%.jsonl
data/bars.%.jsonl:
	uv run snakemake --cores 1 $@ $(if $(TARGET),--config target=$(TARGET),)

PLOTS =

PLOTS += plots/ttile.f32.neon.png
PLOTS += plots/ttile.f32.tower.png
PLOTS += plots/ttile.f32.pinocchio.png
PLOTS += plots/cube_256.f32.neon.png
PLOTS += plots/cube_256.f32.tower.png
PLOTS += plots/cube_256.f32.pinocchio.png
PLOTS += plots/cube_256.f64.neon.png
PLOTS += plots/cube_256.f64.tower.png
PLOTS += plots/cube_256.f64.pinocchio.png
PLOTS += plots/cube_2048.f32.neon.png
PLOTS += plots/cube_2048.f32.tower.png
PLOTS += plots/cube_2048.f32.pinocchio.png
PLOTS += plots/cube_2048.f64.neon.png
PLOTS += plots/cube_2048.f64.tower.png
PLOTS += plots/cube_2048.f64.pinocchio.png

plots/ttile.%.png: data/ttile.%.jsonl src/autotuner/plot_ttile.py
	uv run plot-ttile $< --output $@

plots/cube%.png: data/cube%.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

plots/bars.%.png: data/bars.%.jsonl src/autotuner/plot_cube.py
	uv run plot-cube $< --output $@

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
		docker run --platform linux/amd64 -v .:/src -ti xdsl-autotuner /src/launch.sh; \
	else \
		nice -n -15 taskset -c 2 docker run --platform linux/amd64 --cap-add=SYS_ADMIN --cap-add=PERFMON --security-opt seccomp=unconfined --security-opt apparmor=unconfined --pid=host -v .:/src -ti xdsl-autotuner /src/launch.sh; \
	fi

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -r build 2>/dev/null || true
