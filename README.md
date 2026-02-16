IPU Development
===============

[![CI](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml)
[![Documentation](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://rechefe.github.io/ipu-emulator/)

IPU emulator and assembler toolchain implemented in Python. Includes a sample `fully_connected` app and comprehensive test suite.

Python Packages
---------------

| Package | Path | Description |
|---------|------|-------------|
| `ipu_common` | `src/tools/ipu-common/` | Shared types, register schema, instruction spec |
| `ipu_as` | `src/tools/ipu-as-py/` | IPU assembler (`.asm` → binary) |
| `ipu_emu` | `src/tools/ipu-emu-py/` | IPU emulator with debug CLI |

Quick Start
-----------

```bash
# Run all Python emulator tests
cd src/tools/ipu-emu-py
uv run pytest

# Run the fully_connected app
uv run python -m ipu_emu.apps.fully_connected \
    path/to/instructions.bin \
    path/to/inputs.bin \
    path/to/weights.bin \
    output.bin INT8
```

Build (Bazel)
-------------

```bash
# Build all targets
bazel build //...

# Run all tests
bazel test //...

# Assemble an IPU program
bazel build //src/apps/fully_connected:assemble_fully_connected
```

Notes
-----
- The assembler and emulator share a single source of truth via `ipu_common` (instruction spec, register schema).
- Assembly files (`.asm`) live in `src/apps/` alongside test data.
- Bazel uses hermetic builds with automatic caching and parallelization.
