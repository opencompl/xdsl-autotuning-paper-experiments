# Build LLVM toolchain + uv from Nix flake
FROM docker.io/nixos/nix:latest AS nix-builder
COPY flake.nix flake.lock /tmp/build/
WORKDIR /tmp/build
RUN nix \
    --extra-experimental-features "nix-command flakes" \
    --option filter-syscalls false \
    build
RUN mkdir /tmp/nix-store-closure \
    && cp -R $(nix-store -qR result/) /tmp/nix-store-closure

# Main image
FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Copy Nix store closure and toolchain (LLVM tools + uv + MKL)
COPY --from=nix-builder /tmp/nix-store-closure /nix/store
COPY --from=nix-builder /tmp/build/result /opt/toolchain
ENV PATH="/opt/toolchain/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/toolchain/lib"
ENV LIBRARY_PATH="/opt/toolchain/lib"
ENV C_INCLUDE_PATH="/opt/toolchain/include"
ENV PKG_CONFIG_PATH="/opt/toolchain/lib/pkgconfig"

# Install remaining system dependencies
RUN apt-get update && apt-get install -y \
    git make gpg libxml2 binutils \
    build-essential gcc libc6-dev \
    graphviz wget \
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

# Currently disabled as unused in our experiments, can re-enable later.
# Install uiCA (not in pyproject.toml); uses setuptools from the venv.
# RUN git clone --depth 1 https://gitlab.inria.fr/CORSE/uica-staticdeps.git /opt/uica-staticdeps && \
#     cd /opt/uica-staticdeps && \
#     uv run --python 3.12 --with setuptools ./setup.sh && \
#     rm -rf /opt/uica-staticdeps/.git && \
#     uv cache clean

WORKDIR /src

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
