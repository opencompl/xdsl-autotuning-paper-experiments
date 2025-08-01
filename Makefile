ifneq ("$(wildcard .env)","")
	include .env
	export
endif

.PHONY: filecheck
filecheck:
	uv run lit -v --order=smart tests/filecheck

.PHONY: snakemake
snakemake:
	uv run snakemake --cores all tests --forceall $(if $(TARGET),--config target=$(TARGET),)

.PHONY: tests
tests: filecheck snakemake
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


PLOTS = plots/ttile.f32.neon.png plots/ttile.f32.x86.png plots/cube.f32.neon.png plots/cube.f32.x86.png plots/cube.f64.neon.png plots/cube.f64.x86.png

plots/ttile.%.png: data/ttile.%.jsonl src/plot_ttile.py
	uv run plot-ttile $< --output $@

plots/cube.%.png: data/cube.%.jsonl src/plot_cube.py
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
		nice -n -15 taskset -c 2 docker run --platform linux/amd64 -v .:/src -ti xdsl-autotuner /src/launch.sh; \
	fi

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -r build 2>/dev/null || true
