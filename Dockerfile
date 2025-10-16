# Extract LLVM tools from the original base image
FROM ghcr.io/xdslproject/llvm:20.1.1 AS llvm-extractor

# Multi-stage build for libxsmm
FROM ubuntu:22.04 AS libxsmm-builder

# Install minimal build dependencies for libxsmm
RUN apt-get update && apt-get install -y \
    git make gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv python install 3.12

# Build libxsmm
RUN git clone --depth 1 https://github.com/libxsmm/libxsmm.git /opt/libxsmm && \
    cd /opt/libxsmm && \
    make STATIC=0 PYTHON='/root/.local/bin/uv run --python 3.12' && \
    rm -rf /opt/libxsmm/.git /opt/libxsmm/tests /opt/libxsmm/samples

# Main image - minimal Ubuntu base with extracted LLVM tools
FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/opencompl/xdsl-autotuning-paper-experiments
LABEL org.opencontainers.image.description="LLVM Docker image for xdsl autotuner experiments"
LABEL org.opencontainers.image.licenses=MIT

# Copy essential LLVM tools from the original base image
COPY --from=llvm-extractor /usr/local/bin/llvm-mca /usr/local/bin/
COPY --from=llvm-extractor /usr/local/bin/mlir-translate /usr/local/bin/
COPY --from=llvm-extractor /usr/local/bin/mlir-opt /usr/local/bin/
COPY --from=llvm-extractor /usr/local/bin/clang-20 /usr/local/bin/
COPY --from=llvm-extractor /usr/local/bin/lld /usr/local/bin/
RUN ln -s clang-20 /usr/local/bin/clang && \
    ln -s lld /usr/local/bin/ld.lld && \
    ln -s lld /usr/local/bin/wasm-ld && \
    ln -s lld /usr/local/bin/lld-link

# Copy LLVM include directories
COPY --from=llvm-extractor /usr/local/lib/clang/20/include/ /usr/local/lib/clang/20/include/
COPY --from=llvm-extractor /usr/local/include/ /usr/local/include/

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
    libz3-dev libedit-dev libzstd-dev git make gpg libxml2 binutils \
    papi-tools libpapi-dev \
    build-essential gcc libc6-dev \
    pkg-config intel-oneapi-mkl-2021.4.0 intel-oneapi-mkl-devel-2021.4.0 libomp-dev \
    graphviz \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/*

# Set environment variables
ENV INSIDE_DOCKER=1

# Install uv and Python in a single layer with cache cleanup
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv python install 3.12 && \
    /root/.local/bin/uv cache clean

# Copy libxsmm from builder stage
COPY --from=libxsmm-builder /opt/libxsmm /opt/libxsmm
RUN ln -sf /opt/libxsmm/bin/libxsmm_gemm_generator /usr/bin/libxsmm_gemm_generator

# Install Python dependencies globally (not in a venv)
RUN /root/.local/bin/uv pip install --python python3.12 --system --break-system-packages \
    plotly setuptools git+https://gitlab.inria.fr/tbastian/staticdeps.git

# Install TVM globally
RUN /root/.local/bin/uv pip install --python python3.12 --system --break-system-packages \
    --index-url https://gitlab.inria.fr/api/v4/groups/corse/-/packages/pypi/simple tvm==0.19.0.2025010903

# Install uiCA and its dependencies globally
RUN git clone --depth 1 https://gitlab.inria.fr/CORSE/uica-staticdeps.git /opt/uica-staticdeps && \
    cd /opt/uica-staticdeps && \
    /root/.local/bin/uv run --python python3.12 --with setuptools ./setup.sh && \
    rm -rf /opt/uica-staticdeps/.git && \
    /root/.local/bin/uv cache clean

RUN /root/.local/bin/uv pip install --python python3.12 --system --break-system-packages \
        "jinja2>=3.1.6" \
        "matplotlib>=3.10.3" \
        "pandas>=2.3.1" \
        "snakemake>=8.30.0" \
        "xdsl[dev]"
