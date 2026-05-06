# Development Environment Setup

This guide will help you set up your development environment for building and running IPU applications.

## Prerequisites

The IPU emulator and assembler require the following tools:

- **Bazel** - Build system (managed via Bazelisk)
- **C/C++ Compiler** - GCC or Clang for building the emulator
- **Python 3.10+** - For the assembler and build tools
- **uv** - Fast Python package manager (optional but recommended)
- **Git** - Version control

## Quick Start

### Linux / macOS

1. **Install Bazelisk** (manages Bazel versions automatically):

```bash
# Linux
wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel

# macOS
brew install bazelisk
```

2. **Install build essentials**:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential gcc g++ git curl python3 python3-pip

# macOS (install Xcode Command Line Tools)
xcode-select --install
```

3. **Install uv** (recommended for faster Python package management):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

4. **Clone the repository**:

```bash
git clone https://github.com/rechefe/ipu-emulator.git
cd ipu-emulator
```

### Windows (WSL2)

For Windows users, we recommend using Windows Subsystem for Linux 2 (WSL2):

1. **Install WSL2** with Ubuntu:

```powershell
wsl --install
```

2. **Follow the Linux instructions above** inside your WSL2 Ubuntu environment.

## Using Docker

For a consistent development environment, you can use Docker:

1. **Build the Docker image**:

```bash
docker build -t ipu-emulator .
```

2. **Run the container**:

```bash
docker run -it --rm -v $(pwd):/workspace ipu-emulator bash
```

3. **Build and test inside the container**:

```bash
bazel build //...
bazel test //...
```

## Using VS Code Dev Container

The repository includes a Dev Container configuration for VS Code:

1. **Install** [VS Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open the repository** in VS Code

3. **Reopen in Container**: Press `F1` and select "Dev Containers: Reopen in Container"

4. **Wait for the container to build** - VS Code will automatically set up the environment

The Dev Container includes all required tools pre-installed.

## Verifying Your Setup

After installation, verify everything is working:

```bash
# Check Bazel version
bazel --version

# Check Python version
python3 --version

# Check GCC version
gcc --version

# Build the project
bazel build //...

# Run tests
bazel test //...
```

Expected output:
- Bazel should be 7.0.0 or higher
- Python should be 3.10 or higher
- GCC should be 11.0 or higher
- All builds and tests should pass

## Dependencies

The project uses Bazel to automatically manage dependencies:

### C/C++ Dependencies
- **GoogleTest** (1.15.2) - Testing framework
- **cxxopts** (3.0.0) - Command-line argument parsing

### Python Dependencies
- **Click** (≥8.3.1) - CLI framework
- **Lark** (≥1.3.1) - Parser generator
- **Jinja2** (≥3.1.4) - Template engine
- **MkDocs** (≥1.6.0) - Documentation generator (optional)
- **MkDocs Material** (≥9.5.0) - Documentation theme (optional)

All dependencies are automatically fetched and cached by Bazel on first build.

## Building the Assembler

The IPU assembler (`ipu-as`) is built automatically when needed:

```bash
# Use through Bazel (recommended)
bazel run //src/tools/ipu-as-py:ipu-as -- --help

# Or build and install locally with uv
cd src/tools/ipu-as-py
uv pip install -e .
ipu-as --help
```

## Common Issues

### Bazel build fails with "external repository not found"

**Solution**: Clean the Bazel cache and rebuild:
```bash
bazel clean --expunge
bazel build //...
```

### Python version mismatch

**Solution**: Ensure Python 3.10+ is installed and set as default:
```bash
python3 --version
# If needed, install Python 3.10
sudo apt-get install python3.10
```

### Permission denied when running Bazel

**Solution**: Ensure Bazelisk is executable:
```bash
chmod +x /usr/local/bin/bazel
```

### WSL2 performance issues

**Solution**: Clone the repository inside WSL2 filesystem (not `/mnt/c/`):
```bash
# Good - inside WSL2 filesystem
cd ~
git clone https://github.com/rechefe/ipu-emulator.git

# Bad - on Windows filesystem (slow)
cd /mnt/c/Users/...
```

## IDE Setup

### VS Code

Recommended extensions:
- **C/C++** (Microsoft) - C/C++ IntelliSense
- **Bazel** (BazelBuild) - Bazel support
- **Python** (Microsoft) - Python language support
- **Pylance** (Microsoft) - Python type checking

### CLion

CLion supports Bazel projects natively. Open the repository and select "Import Bazel Project".

## Next Steps

Now that your environment is set up:

1. [Build your first application](building-applications.md)
2. Learn the [Assembly Syntax](assembly-syntax.md)
3. Explore the [Instruction Reference](instructions.md)

## Additional Resources

- [Bazel Documentation](https://bazel.build/docs)
- [uv Documentation](https://docs.astral.sh/uv/)
- [GitHub Repository](https://github.com/rechefe/ipu-emulator)
