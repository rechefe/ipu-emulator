FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Base tools. No C/C++ toolchain: the repository builds none.
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    pkg-config \
    zip \
    unzip \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Pinned bazelisk from the shared bootstrap script, the
# same one CI and the Claude Code SessionStart hook run. Installed system-wide,
# so the non-root user below needs no installs of its own.
COPY tools/bootstrap.sh /tmp/bootstrap.sh
RUN BIN_DIR=/usr/local/bin /tmp/bootstrap.sh \
    && rm /tmp/bootstrap.sh

# Set working directory
WORKDIR /workspace

# Create non-root user for dev-container
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME

# Make shell scripts executable
COPY --chown=$USERNAME:$USERNAME . /workspace
RUN find /workspace -type f -name "*.sh" -exec chmod +x {} \;

CMD ["/bin/bash"]
