#!/bin/bash

# Check if we're inside Docker
if [ "$INSIDE_DOCKER" != "1" ]; then
    echo "This script should be run inside the Docker container"
    exit 1
fi

# Get the directory where launch.sh is located
# Although we should always mount with src, the GH CI hardcodes a path to mount the
# repo on.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set and export the environment variable
export UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/venv_docker"

# Copy dependencies from cache instead of hardlinking to silence UV complaining
export UV_LINK_MODE=copy

# Check if venv_docker exists
if [ ! -d "$SCRIPT_DIR/venv_docker" ]; then
    echo "Setting up virtual environment..."
    cp -r /opt/venv_template "$SCRIPT_DIR/venv_docker"
    echo "Virtual environment setup complete"
else
    echo "Virtual environment already exists"
fi

# Start an interactive shell
echo "Starting interactive session..."
exec /bin/bash
