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
