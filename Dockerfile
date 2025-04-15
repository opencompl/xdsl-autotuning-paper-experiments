FROM ghcr.io/xdslproject/llvm:20.1.1

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Install dependencies and clean up in a single layer
RUN apt-get update && apt-get install -y \
    libz3-dev libedit-dev libzstd-dev git make gpg libxml2 binutils \
    build-essential gcc libc6-dev \
    graphviz \
    binutils-aarch64-linux-gnu binutils-x86-64-linux-gnu \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV UV_CACHE_DIR="/src/.cache/uv"
ENV UV_PROJECT_ENVIRONMENT="venv_docker"
ENV INSIDE_DOCKER=1

# Install uv and Python in a single layer
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv python install && \
    /root/.local/bin/uv venv $UV_PROJECT_ENVIRONMENT

RUN /root/.local/bin/uv pip install --python $UV_PROJECT_ENVIRONMENT \
    plotly setuptools git+https://gitlab.inria.fr/tbastian/staticdeps.git

# Install Python dependencies and setup uiCA in a single layer
WORKDIR /src/
RUN git clone https://gitlab.inria.fr/CORSE/uica-staticdeps.git && \
    cd uica-staticdeps && \
    /root/.local/bin/uv run --python ../../$UV_PROJECT_ENVIRONMENT ./setup.sh && \
    cd .. && \
    rm -rf /src/.cache/uv/*
WORKDIR /src/
