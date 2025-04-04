.PHONY: filecheck
filecheck:
	uv run lit -v --order=smart tests/filecheck

.PHONY: tests
tests: filecheck
	@echo "All tests passed successfully"
	@exit 0

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
.PHONY: docker-run
docker-run:
	docker run --platform linux/amd64 -v .:/src -ti xdsl-autotuner
