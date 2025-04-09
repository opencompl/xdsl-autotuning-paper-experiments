FROM ubuntu:latest

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Install wget and xz-utils for clang installation
RUN apt-get update && apt-get install -y \
    wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the official LLVM 19.1.7 binary
WORKDIR /tmp
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-Linux-X64.tar.xz && \
    tar xf LLVM-19.1.7-Linux-X64.tar.xz -C /usr/local --strip-components=1 && \
    rm LLVM-19.1.7-Linux-X64.tar.xz
WORKDIR /

# Install xz-utils
RUN apt-get update && apt-get install -y \
    libz3-dev libedit-dev libzstd-dev git make gpg libxml2 binutils \
    build-essential gcc libc6-dev \
    graphviz \
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
RUN /root/.local/bin/uv venv

# Install additionnal Python dependencies
RUN /root/.local/bin/uv pip install plotly
RUN /root/.local/bin/uv pip install git+https://gitlab.inria.fr/tbastian/staticdeps.git

# Setup the INRIA distribution of uiCA
WORKDIR /src/
RUN git clone https://gitlab.inria.fr/CORSE/uica-staticdeps.git
WORKDIR /src/uica-staticdeps
RUN /root/.local/bin/uv run ./setup.sh
