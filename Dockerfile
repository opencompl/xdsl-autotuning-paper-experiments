FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add LLVM apt repository key (no sudo needed in Dockerfile)
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

# Add LLVM apt repository for Ubuntu Noble (24.04)
RUN echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main" > /etc/apt/sources.list.d/llvm.list && \
    echo "deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main" >> /etc/apt/sources.list.d/llvm.list

# Update package lists and install LLVM 19
RUN apt-get update && apt-get install -y \
    clang-19 \
    lld-19 \
    lldb-19 \
    llvm-19 \
    libmlir-19-dev \
    mlir-19-tools \
    # Add any other LLVM packages you need here
    && rm -rf /var/lib/apt/lists/*

# Set up alternatives to make clang-19 the default clang
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100

# Optional: Set working directory
WORKDIR /app

# Optional: Default command when container starts
CMD ["bash"]
