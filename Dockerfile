FROM ubuntu:latest AS builder

# Install xz-utils
RUN apt-get update && apt-get install -y \
    wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the official LLVM 19.1.7 binary
WORKDIR /tmp
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-Linux-X64.tar.xz && \
    tar xf LLVM-19.1.7-Linux-X64.tar.xz -C /usr/local --strip-components=1 && \
    rm LLVM-19.1.7-Linux-X64.tar.xz

# Create a smaller final image
FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Install only the minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    wget libz3-dev libedit-dev libzstd-dev git make gpg \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /usr/local/bin /usr/local/lib

# Copy only the needed tools and their required libraries
COPY --from=builder /usr/local/bin/mlir-cpu-runner /usr/local/bin/
COPY --from=builder /usr/local/bin/mlir-opt /usr/local/bin/
COPY --from=builder /usr/local/bin/mlir-translate /usr/local/bin/
COPY --from=builder /usr/local/bin/clang /usr/local/bin/
COPY --from=builder /usr/local/bin/llc /usr/local/bin/
COPY --from=builder /usr/local/bin/llvm-mca /usr/local/bin/

# Copy required shared libraries
COPY --from=builder /usr/local/lib/*.so* /usr/local/lib/

# Copy necessary headers from the builder stage
COPY --from=builder /usr/include/* /usr/include/

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
