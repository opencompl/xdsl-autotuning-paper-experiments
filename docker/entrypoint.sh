#!/usr/bin/env bash

set -euo pipefail

# Avoid hardlinks across bind mounts (uv warns on Docker volumes).
export UV_LINK_MODE=copy

# Install editable project + refresh deps from the mounted tree.
# --inexact keeps packages installed only in the image (e.g. uiCA) from being removed.
if [ -f /src/pyproject.toml ]; then
  uv sync --directory /src --locked --inexact
fi

exec "$@"
