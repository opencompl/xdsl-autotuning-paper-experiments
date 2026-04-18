#!/usr/bin/env bash

set -euo pipefail

# TODO: remove once the Docker image ships mlir-opt on PATH directly
if command -v mlir-opt-20 &>/dev/null && ! command -v mlir-opt &>/dev/null; then
  ln -sf "$(command -v mlir-opt-20)" /usr/local/bin/mlir-opt
fi

# Avoid hardlinks across bind mounts (uv warns on Docker volumes).
export UV_LINK_MODE=copy

# Install editable project + refresh deps from the mounted tree.
# --inexact keeps packages installed only in the image (e.g. uiCA) from being removed.
if [ -f /src/pyproject.toml ]; then
  uv sync --directory /src --locked --inexact
fi

exec "$@"
