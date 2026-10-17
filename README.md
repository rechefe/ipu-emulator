IPU Development
===============

[![CI](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml)
[![Documentation](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://rechefe.github.io/ipu-emulator/)

IPU emulator and assembler toolchain implemented in Python. Includes a sample `fully_connected` app and comprehensive test suite.

Python Packages
---------------

The toolchain is one Python package per directory under `src/tools/`. [Architecture](docs/content/explanation/architecture.md) lists them, what each owns, which way their dependencies point, and where to make a change.

Quick Start
-----------

On Linux, from a fresh clone:

```bash
./tools/bootstrap.sh                               # install the pinned bazelisk as `bazel`
bazel test //...                                   # run the default test tier
bazel run //kernels/projections:proj_qkv_144_p4    # run one FP32 kernel case end to end
```

[Getting started](docs/content/getting-started/index.md) explains each step,
what the kernel run prints, and the other environments (WSL2, macOS, Docker,
the dev container).

The full documentation is at <https://rechefe.github.io/ipu-emulator/>: `dev/` follows master, and each release `vX.Y` has its own version (`latest` is the newest).

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

The nightly workflow (`.github/workflows/nightly.yml`) runs
`bazel coverage --config=full //...` on master: every non-manual test once,
with coverage (per-package line coverage in the job summary, the lcov report
and an HTML report as a 30-day artifact). It also runs the `test_full_*`
stacks below, and the heavy tier again in prototyping mode
(`--config=proto`). A failed nightly run opens a "Nightly failure" issue
(label `nightly`), which the next green run closes. Start it by hand from
the Actions tab (workflow_dispatch).

After the tests, the nightly `benchmarks` job records import time, assemble
time, emulator cycles/s, peak memory and CI wall times on the
`benchmark-data` branch, as a history with no gate. Run them locally with
`bazel run //tools/bench:bench -- --quick`; see
[tools/bench/README.md](tools/bench/README.md).

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
committed, hash-pinned lock compiled from the packages' `pyproject.toml` files and
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

PyTorch is test-only (the shared kernel references, `ipu-kernels`' `dev`
extra). It is the CPU wheel from PyTorch's own index: `ipu-kernels`'
`pyproject.toml` names the index for `uv`, and `pip.parse` in `MODULE.bazel`
adds it as an extra index. Only targets with `requirement("torch")` fetch it,
but `//src/tools/ipu-kernels:test_reference_ops` is one of them and is in the
default tier, so a cold `bazel test //...` downloads the ~200 MB wheel once.

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

The run stops in the debugger before the first instruction word, with the
same case and options as an ordinary run. An interactive terminal is required.
[Debug a kernel](docs/content/how-to/debug-a-kernel.md) explains the workflow,
and the [debugger controls](docs/content/reference/cli.md#debugger) list every
key.
