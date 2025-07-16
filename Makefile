.PHONY: filecheck
filecheck:
	uv run lit -v --order=smart tests/filecheck

.PHONY: tests
tests: filecheck snakemake
	@echo "All tests passed successfully"
	@exit 0

.PHONY: snakemake-mac
snakemake-mac:
	uv run snakemake build/test_mac.txt

.PHONY: snakemake-docker
snakemake-docker:
	uv run snakemake build/test_docker.txt

.PHONY: snakemake
snakemake:
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		$(MAKE) snakemake-mac; \
	else \
		$(MAKE) snakemake-docker; \
	fi

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

# run docker image
# Arjun's tip:
# Call nice to make your processor not nice (It won't let other processes run)
# Pin to core 2
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
