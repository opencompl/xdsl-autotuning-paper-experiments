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
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV INSIDE_DOCKER=1

# Install uv and Python in a single layer
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv python install && \
    /root/.local/bin/uv venv /opt/build_venv

# Install Python dependencies in build venv
RUN /root/.local/bin/uv pip install --python /opt/build_venv \
    plotly setuptools git+https://gitlab.inria.fr/tbastian/staticdeps.git

# Install uiCA and its dependencies in build venv
RUN git clone https://gitlab.inria.fr/CORSE/uica-staticdeps.git /opt/uica-staticdeps && \
    cd /opt/uica-staticdeps && \
    /root/.local/bin/uv run --python /opt/build_venv ./setup.sh && \
    /root/.local/bin/uv cache clean

# Copy the entire build venv to a template location
RUN cp -r /opt/build_venv /opt/venv_template
