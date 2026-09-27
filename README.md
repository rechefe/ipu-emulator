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


## Debug a registered kernel

```bash
bazel run --config=debug //src/tools/ipu-apps:identity
bazel run --config=debug //src/tools/ipu-apps:softmax_rows_partial -- --n 32 --rows 8
```

This uses the same registry harness and input cases as normal execution,
opening the TUI after setup and before the first instruction. Kernel code
needs no debugger imports or special entry point. F8 steps, F5 continues, F9
toggles a breakpoint, F10 runs to the selected instruction, and F11 maximizes
the focused pane. `q` or Ctrl-C cancels cleanly without checking partial output.
The terminal must be interactive; the deprecated CLI is not a TUI fallback.

`--config=debug` is a native repository Bazel configuration. Use full targets
and normal `test_<kernel>` labels; no custom `bazel debug` command or wrapper
is installed. See [debugger documentation](docs/content/debugging.md) for all controls.
