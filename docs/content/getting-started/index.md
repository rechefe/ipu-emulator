# Getting started

Go from a fresh clone to a passing test run and one kernel run. Every command
runs from the repository root.

## Prerequisites

- Linux on x86-64 or ARM64, with `git`, `curl` and `python3`. On Windows, use
  [WSL2](#windows-wsl2); on macOS, see [macOS](#macos).
- About 5 GB of free disk for the Bazel cache.

Bazel manages every other tool and Python package. The repository has no C or
C++ code, so no compiler is needed.

## Install the toolchain

```bash
git clone https://github.com/rechefe/ipu-emulator.git
cd ipu-emulator
./tools/bootstrap.sh
```

`tools/bootstrap.sh` installs bazelisk as `bazel`, at the pinned version and
checksum that CI, the dev container and the Claude Code hook also use.
Bazelisk then runs the Bazel version in `.bazelversion`. The script installs
into `/usr/local/bin`, or into `~/.local/bin` when `/usr/local/bin` is not
writable; set `BIN_DIR` to choose. A second run downloads nothing.

Check the install:

```bash
bazel version
```

## Run the tests

```bash
bazel test //...
```

The first run downloads Bazel, Python and the locked packages (including the
~200 MB PyTorch CPU wheel), then runs the default test tier. On a GitHub-hosted
CI runner the test step takes about 20 minutes. Bazel caches results, so a
second run with nothing changed takes seconds.

The run passes when `bazel test` exits 0 and its summary ends `N tests pass.`
[Test tiers](https://github.com/rechefe/ipu-emulator#test-tiers) in the
README explains the default and heavy tiers.

## Run one kernel

Run the QKV projection kernel's default FP32 case:

```bash
bazel run //src/tools/ipu-apps:proj_qkv_144_p4
```

The run assembles the kernel, loads its inputs, emulates it to `HALT` and
compares the outputs with a reference. It takes about a minute. The output
starts with:

```text
proj_qkv_144_p4/default: PASS (572006 cycles)
=== Run summary ===
Total cycles:          572006
Mult active:           497664  (87.0%)
...
Effective MAC:          87.0%
```

| Line | Meaning |
|---|---|
| `PASS (N cycles)` | Every output matched the reference within the case's tolerance. A mismatch prints `error:` and the reason instead, and the command exits 1. |
| `Total cycles` | Instruction words retired. |
| `Mult active` | Cycles whose MULT slot was not a no-op, and their fraction of the total cycles. |
| `Effective MAC` | Multiplier work that was real arithmetic, as a fraction of the run's peak (MAC utilisation). |

[Run statistics](../reference/cli.md#run-statistics) lists every counter.
`--list-cases` lists a kernel's cases, and `--case NAME` runs one of them.

## Next steps

| Page | What it covers |
|---|---|
| [Write a kernel](../building-applications.md) | Write an assembly kernel and the Python code that feeds it. |
| [Add an instruction](../how-to/add-an-instruction.md) | Add an instruction to the ISA, the assembler and the emulator. |
| [Debug a kernel](../debugging.md) | Step through a kernel in the debugger and inspect registers. |
| [Use the reference model from cocotb](../how-to/reference-model-cocotb.md) | Drive the emulator as a reference model from an RTL testbench. |

The [Getting Started guide (PDF)](https://github.com/rechefe/ipu-emulator/releases/download/guide-latest/getting-started.pdf)
is a printable introduction to the IPU, its assembly language and the emulator.
Contributors read [Code conventions](../code-conventions.md) before writing code.

## Other environments

### Windows (WSL2)

Install WSL2 with Ubuntu, then follow this page inside it:

```powershell
wsl --install
```

Clone into the WSL2 file system (for example `~/ipu-emulator`), not under
`/mnt/c/`: Bazel is much slower on the Windows file system.

### macOS

`tools/bootstrap.sh` supports Linux only. Install bazelisk with Homebrew
instead; it reads the same `.bazelversion`:

```bash
brew install bazelisk
```

### Docker

The `Dockerfile` builds an Ubuntu image with the pinned toolchain from
`tools/bootstrap.sh`:

```bash
docker build -t ipu-emulator .
docker run -it --rm -v "$PWD":/workspace ipu-emulator bash
```

Inside the container, run the commands from [Run the tests](#run-the-tests).

### VS Code and the dev container

`.devcontainer/devcontainer.json` builds the same image. Install the
[Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers),
open the repository, press `F1` and select **Dev Containers: Reopen in Container**.

VS Code offers the recommended extensions from `.vscode/extensions.json`, and
the dev container installs them:

- **IPU Assembly** (`yardencarmi.ipu-asm`): highlighting, diagnostics, completion
  and kernel run and debug buttons for `.asm` files
- **Python** and **Pylance** (Microsoft): Python language support and type checking
- **Ruff** (Astral): Python linting and formatting
- **Bazel** (BazelBuild): Bazel support
- **GitLess** (maattdd): inline Git history
- **Material Icon Theme** (PKief): file icons

If **IPU Assembly** is not in the Marketplace, install the latest build:

```bash
curl -LO https://github.com/rechefe/ipu-emulator/releases/download/vscode-latest/ipu-asm.vsix
code --install-extension ipu-asm.vsix
```

## Build this documentation

The site builds offline through Bazel:

```bash
bazel build --config=docs //docs:build_docs      # writes bazel-bin/docs/site.tar
bazel run --config=docs //docs:serve_docs        # build and serve it locally
```

`--config=docs` forwards the optional `GISCUS_*` overrides into the build. The
[release log](../release-log.md) is committed, and a workflow refreshes it on
`master` after every merge. To refresh it by hand:

```bash
bazel run //docs:update_release_log -- --recent --output "$PWD/docs/content/release-log.md"
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `bazel: command not found` after the bootstrap | Add the install directory (`~/.local/bin` when `/usr/local/bin` was not writable) to `PATH`. |
| `external repository not found` | Run `bazel clean --expunge`, then the command again. |
| A kernel run prints `error:` | Run it again with `--config=debug` to step through it; see [Debug a kernel](../debugging.md). |
