FROM ghcr.io/xdslproject/llvm:20.1.1

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Install xz-utils
RUN apt-get update && apt-get install -y \
    libz3-dev libedit-dev libzstd-dev git make gpg libxml2 binutils \
    build-essential gcc libc6-dev \
    graphviz \
    binutils-aarch64-linux-gnu binutils-x86-64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh

# Install Python
RUN /root/.local/bin/uv python install

# Put cache in src
ENV UV_CACHE_DIR="/src/.cache/uv"

# Use "venv_docker" venv inside Docker
ENV UV_PROJECT_ENVIRONMENT="venv_docker"

# Set env variable to mark that we're in a docker container
ENV INSIDE_DOCKER=1

# Install the virtual environement
RUN /root/.local/bin/uv venv $UV_PROJECT_ENVIRONMENT

# Install additionnal Python dependencies
RUN /root/.local/bin/uv pip install --python $UV_PROJECT_ENVIRONMENT plotly
RUN /root/.local/bin/uv pip install --python $UV_PROJECT_ENVIRONMENT setuptools
RUN /root/.local/bin/uv pip install --python $UV_PROJECT_ENVIRONMENT git+https://gitlab.inria.fr/tbastian/staticdeps.git

# Setup the INRIA distribution of uiCA
WORKDIR /src/
RUN git clone https://gitlab.inria.fr/CORSE/uica-staticdeps.git
WORKDIR /src/uica-staticdeps
RUN /root/.local/bin/uv run --python ../../$UV_PROJECT_ENVIRONMENT ./setup.sh
WORKDIR /src/
