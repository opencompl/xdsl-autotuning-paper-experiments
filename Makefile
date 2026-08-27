# On macs, the default target is "neon"
ifeq ($(shell uname -s),Darwin)
TARGET := neon
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
	uv run snakemake $(SCHEDULER_FLAG) --quiet all --cores all tests --forceall $(if $(TARGET),--config target=$(TARGET),)

.PHONY: tests
tests: pytest filecheck snakemake
	@echo "All tests passed successfully"
	@exit 0

.PHONY: dataset_code
dataset_code:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all dataset_code $(if $(TARGET),--config target=$(TARGET),)

.PHONY: dataset_validate
dataset_validate:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all dataset_validate --forceall $(if $(TARGET),--config target=$(TARGET),)

# --cores 1 to avoid contention issues when measuring performance.
# Run `make clean` to re-measure everything.
# Run `make clean-ours` to re-measure just our code.
.PHONY: dataset
dataset: dataset_code
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 dataset $(if $(TARGET),--config target=$(TARGET),)

# Validate and freshly measure the direct N=1, K=64 assembly experiments.
# This target is currently available for the AVX-512 tower and pinocchio targets.
.PHONY: benchmark-kdot
benchmark-kdot:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall kdot_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.kdot_n1.jsonl $(if $(TARGET),--config target=$(TARGET),)

# Compare segmented-K and width-matched outer-product schedules for N=2 and N=4.
.PHONY: benchmark-skinny
benchmark-skinny:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall skinny_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.skinny_n2_n4.jsonl $(if $(TARGET),--config target=$(TARGET),)

# Test whether AVX-512 gathers make vectorizing across M competitive for M=8/16.
.PHONY: benchmark-gather-m
benchmark-gather-m:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall gather_m_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.gather_m.jsonl $(if $(TARGET),--config target=$(TARGET),)

# Replace M-vectorized gathers with contiguous 8x8 loads and register transposes.
.PHONY: benchmark-transpose-m
benchmark-transpose-m:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall transpose_m_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.transpose_m.jsonl $(if $(TARGET),--config target=$(TARGET),)

# Pack two N=3 K rows into six active ZMM lanes.
.PHONY: benchmark-skinny-n3
benchmark-skinny-n3:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall skinny_n3_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.skinny_n3.jsonl $(if $(TARGET),--config target=$(TARGET),)

# Test an exact-width N=3 epilogue and only one safety-masked B load.
.PHONY: benchmark-skinny-n3-narrow
benchmark-skinny-n3-narrow:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall skinny_n3_narrow_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.skinny_n3_narrow.jsonl $(if $(TARGET),--config target=$(TARGET),)

# Map the M=16 direct-kernel crossover over K=8/16/32/64/128 and N=1/2/3/4.
.PHONY: benchmark-k-sweep
benchmark-k-sweep:
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores all --forceall k_sweep_m16_validate $(if $(TARGET),--config target=$(TARGET),)
	uv run snakemake $(SCHEDULER_FLAG) --quiet --cores 1 --forceall data/$(TARGET)/f64.k_sweep_m16.jsonl $(if $(TARGET),--config target=$(TARGET),)


# Prevent Make from deleting this intermediate file
.PRECIOUS: data/$(TARGET)/f64.bars.jsonl
data/$(TARGET)/f64.bars.jsonl:
	uv run snakemake $(SCHEDULER_FLAG) --cores 1 $@ --config target=$(TARGET)

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

PLOTS += plots/pinocchio/f32.ttile.png
PLOTS += plots/pinocchio/f64.ttile.png
PLOTS += plots/pinocchio/f64.ttile_squares.png
PLOTS += plots/pinocchio/f64.ttile_combined.png
PLOTS += plots/pinocchio/f64.heatmap.png

NANO_KERNEL_PLOT_TARGETS = tower
NANO_KERNEL_PLOT_DTYPES = f32 f64
NANO_KERNELS = libxsmm skx-fsdbcst skx-nofsdbcst
NANO_KERNEL_PLOTS = $(foreach target,$(NANO_KERNEL_PLOT_TARGETS),$(foreach dtype,$(NANO_KERNEL_PLOT_DTYPES),$(foreach nano_kernel,$(NANO_KERNELS),plots/$(target)/$(dtype).nano-kernel.$(nano_kernel).png)))

PLOTS += $(NANO_KERNEL_PLOTS)

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

define NANO_KERNEL_PLOT_RULE
plots/$(1)/$(2).nano-kernel.$(3).png: data/$(1)/$(2).nano-kernel.$(3).jsonl src/autotuner/plot_nano_kernel_heatmap.py
	uv run plot-nano-kernel-heatmap $$< --output $$@
endef

$(foreach target,$(NANO_KERNEL_PLOT_TARGETS),$(foreach dtype,$(NANO_KERNEL_PLOT_DTYPES),$(foreach nano_kernel,$(NANO_KERNELS),$(eval $(call NANO_KERNEL_PLOT_RULE,$(target),$(dtype),$(nano_kernel))))))

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

.PHONY: clean-ours
clean-ours:
	find build -name 'xdsl_libxsmm.*' -exec rm -f {} + 2>/dev/null || true
	find build -name 'compxsmm.*' -exec rm -f {} + 2>/dev/null || true
	rm -f data/$(TARGET)/*

.PHONY: clean
clean:
	find tests/filecheck -type d -name "Output" -exec rm -rf {} \; 2>/dev/null || true
	rm -rf build 2>/dev/null || true
	rm -f data/$(TARGET)/*
