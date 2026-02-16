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



Build (Bazel)
-------------

```bash
# Build all targets
bazel build //...

# Run all tests
bazel test //...

# Assemble an IPU program
bazel build //src/tools/ipu-apps:assemble_fully_connected
```

Notes
-----
- The assembler and emulator share a single source of truth via `ipu_common` (instruction spec, register schema).
- Applications live in `src/tools/ipu-apps/` — each app is a subpackage under `ipu_apps/` with its assembly, test data, and Python harness together.
- Bazel uses hermetic builds with automatic caching and parallelization.
