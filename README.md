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
| `ipu_isa` | `src/tools/ipu-isa/` | Shared types, register schema, instruction spec |
| `ipu_as` | `src/tools/ipu-as-py/` | IPU assembler (`.asm` → binary) |
| `ipu_emu` | `src/tools/ipu-emu-py/` | IPU emulator with debug CLI |

Quick Start
-----------

1. Install [Bazelisk](https://github.com/bazelbuild/bazelisk) as `bazel`; see
   [Setup](docs/content/setup.md) for details.
2. Run the test suite:

   ```bash
   bazel test //...
   ```

3. Run one kernel end to end (assemble, load inputs, emulate, check outputs):

   ```bash
   bazel run //src/tools/ipu-apps:identity
   ```

The full documentation is at <https://rechefe.github.io/ipu-emulator/>.

Build (Bazel)
-------------

```bash
# Build all targets
bazel build //...

# Run the default test tier
bazel test //...

# Include the heavy (nightly) tier
bazel test --config=full //...

# Assemble an IPU program
bazel build //src/tools/ipu-apps:assemble_fully_connected
```

Test tiers
----------

Tests are split into two tiers by Bazel test size. The named configs in
`.bazelrc` select them, so CI and developers run the same command.

| Tier | Holds | Command |
|------|-------|---------|
| Default | Every test except size `enormous`: what a change must pass before review. CI runs it on every pull request and every push to master. | `bazel test //...` (CI: `bazel test --config=ci //...`, which also skips `docs` targets and uses the CI caches) |
| Heavy | Size `enormous` tests, such as the per-kernel `test_alias_profile_kernels_*` and `test_debug_launch_kernels_*` sweeps. Runs nightly. | `bazel test --config=full //...` |

`manual` targets (for example the `test_full_*` MobileViT stacks) are in
neither tier. Run them by name with `--config=full`, for example
`bazel test --config=full //src/tools/ipu-apps:test_full_layer_l4`. The same
goes for any heavy-tier target named on the command line: the default
`--test_size_filters=-enormous` applies to named targets too, so without
`--config=full` Bazel finds no test to run.

Put a new test in the heavy tier (`size = "enormous"`) only when it repeats
coverage that a default-tier test already gives, or when it takes more than
about 2 minutes on CI. Run `bazel test --config=full //...` before review when
a change touches code that only heavy-tier tests exercise.

Development
-----------

### Python dependencies

Bazel installs third-party Python packages from `requirements_lock.txt`, a
committed, hash-pinned lock compiled from the four `pyproject.toml` files and
`tools/requirements-tools.in` (repository tools such as `uv`). Fetching never
resolves versions, so every checkout of a commit tests against the same
packages. After adding or changing a dependency, refresh the lock and commit
it; no host tool is needed:

```bash
bazel run //tools:update_requirements_lock                              # keep current versions
bazel run //tools:update_requirements_lock -- --upgrade-package numpy   # move one package
bazel run //tools:update_requirements_lock -- --upgrade                 # move everything
```

Each package's `py_library` carries only its runtime packages, the
`[project].dependencies` of its `pyproject.toml`. They are written once, as
`RUNTIME_DEPS` in the package's `deps.bzl`. Test and docs packages (the `dev`
and `docs` extras) go on the test and docs targets that use them, as
`requirement("pytest")`. Changing a runtime dependency means editing both
`pyproject.toml` and `deps.bzl`, then refreshing the lock;
`//tools:deps_consistency_test` fails if either step is missed.

Notes
-----
- The assembler and emulator share a single source of truth via `ipu_isa` (instruction spec, register schema).
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
