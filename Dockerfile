# Build LLVM toolchain + uv from Nix flake
FROM nixos/nix:latest AS nix-builder
COPY flake.nix flake.lock /tmp/build/
WORKDIR /tmp/build
RUN nix \
    --extra-experimental-features "nix-command flakes" \
    --option filter-syscalls false \
    build
RUN mkdir /tmp/nix-store-closure \
    && cp -R $(nix-store -qR result/) /tmp/nix-store-closure

# Multi-stage build for libxsmm
FROM ubuntu:22.04 AS libxsmm-builder

# Install minimal build dependencies for libxsmm
RUN apt-get update && apt-get install -y \
    git make gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/
RUN uv python install 3.12

# Build libxsmm
RUN git clone --depth 1 https://github.com/libxsmm/libxsmm.git /opt/libxsmm && \
    cd /opt/libxsmm && \
    make STATIC=0 PYTHON='/bin/uv run --python 3.12' && \
    rm -rf /opt/libxsmm/.git /opt/libxsmm/tests /opt/libxsmm/samples

# Main image - minimal Ubuntu base with extracted LLVM tools
FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Copy Nix store closure and toolchain (LLVM tools + uv)
COPY --from=nix-builder /tmp/nix-store-closure /nix/store
COPY --from=nix-builder /tmp/build/result /opt/toolchain
ENV PATH="/opt/toolchain/bin:$PATH"

# Install basic dependencies first (including ca-certificates for Intel repo)
RUN apt-get update && apt-get install -y \
    ca-certificates wget gpg \
    && rm -rf /var/lib/apt/lists/*

# Pointer to Intel repos
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
  && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | tee /etc/apt/sources.list.d/oneapi.list \
  && apt update

# Install dependencies and clean up aggressively in a single layer
RUN apt-get update && apt-get install -y \
    git make gpg libxml2 binutils \
    papi-tools libpapi-dev \
    build-essential gcc libc6-dev \
    pkg-config intel-oneapi-mkl-2021.4.0 intel-oneapi-mkl-devel-2021.4.0 libomp-dev \
    graphviz \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/*

ENV INSIDE_DOCKER=1

RUN uv python install 3.12 && uv cache clean

# Single project venv for build-time sync and entrypoint runtime sync.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv venv /opt/venv -p 3.12 \
    && uv sync --locked --no-install-project

# Copy libxsmm from builder stage
COPY --from=libxsmm-builder /opt/libxsmm /opt/libxsmm
RUN ln -sf /opt/libxsmm/bin/libxsmm_gemm_generator /usr/bin/libxsmm_gemm_generator

# Install uiCA (not in pyproject.toml); uses setuptools from the venv.
RUN git clone --depth 1 https://gitlab.inria.fr/CORSE/uica-staticdeps.git /opt/uica-staticdeps && \
    cd /opt/uica-staticdeps && \
    uv run --python 3.12 --with setuptools ./setup.sh && \
    rm -rf /opt/uica-staticdeps/.git && \
    uv cache clean

WORKDIR /src

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
