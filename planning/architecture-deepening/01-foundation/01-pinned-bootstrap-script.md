# One pinned toolchain bootstrap script for CI, the Dockerfile and the SessionStart hook

Part of [Epic E01: Foundation](00-epic.md) · ID E01-01 · Size S · Depends on: none · Behaviour change: none (same Bazel version; installs become pinned and verified)

## Goal

One idempotent script installs the toolchain at pinned, checksummed versions. Every environment that builds the repository (three GitHub workflows, the Dockerfile and dev container, the Claude Code SessionStart hook) calls it, so they all run the same Bazel the same way.

## Background

Verified at 694a8a8:

- `.bazelversion` pins Bazel 8.3.0, so the Bazel version itself is fixed. Bazelisk, which reads it, is not:
  - `ci.yml:28`, `docs.yml:39`, `vscode-extension.yml:76` and `Dockerfile:24` download `bazelisk/releases/latest/...`, without a checksum.
  - `.claude/hooks/session-start.sh:17` pins bazelisk v1.27.0, also without a checksum.
- `uv` is installed from `https://astral.sh/uv/install.sh` (unversioned) in `ci.yml:21-24`, `docs.yml:32-35`, `vscode-extension.yml:68-72` and twice in the Dockerfile (`Dockerfile:28, 51`). It is needed today only because `tools/python_deps.bzl` runs `uv pip compile` at fetch time; E01-02 removes that need.
- The Dockerfile adds `~/.cargo/bin` to `PATH` (`Dockerfile:29, 48`), but current `uv` installers write to `~/.local/bin`. `ci.yml:24` and `docs.yml:35` have the same stale path; `vscode-extension.yml:71-72` works around it by adding both.
- `actions/setup-python` installs 3.11 in `ci.yml:16-19` and `vscode-extension.yml:58-61` and 3.10 in `docs.yml:27-30`. Bazel uses its own hermetic 3.10 (`MODULE.bazel:10-17`); the host Python only ran the `uv` installer.
- The Dockerfile installs `build-essential`, `gcc` and `g++` (Dockerfile:7-21) for a C/C++ toolchain the repository does not use (E00-10).
- The hook carries the cloud-container specifics: `BAZELISK_BASE_URL` pointed at GitHub releases, a BCR mirror registry and `--lockfile_mode=off` in the user `~/.bazelrc` (session-start.sh:14-55). Those are environment policy, not toolchain choice, and stay in the hook.
- There is no `.dockerignore`, so `COPY . /workspace` (Dockerfile:54) copies `bazel-*` output trees and `vscode-ipu-asm/node_modules`.

## Decisions applied

- Q61(a): the dev loop and CI time come first. One pinned bootstrap shared by CI, Docker and the hook is the build/CI default accepted with Q29 (see the epic).

## Implementation

1. Add `tools/bootstrap.sh` (`set -euo pipefail`, idempotent):
   ```bash
   # Pinned toolchain. To upgrade: change the version and both checksums,
   # run the script once locally, and open a PR (CI runs it on every job).
   BAZELISK_VERSION=v1.27.0
   BAZELISK_SHA256_LINUX_AMD64=<sha256>
   BAZELISK_SHA256_LINUX_ARM64=<sha256>
   ```
   - Install bazelisk as `bazel` into `${BIN_DIR:-/usr/local/bin}` (fall back to `~/.local/bin` when not writable, as the hook does). Download from `${BAZELISK_DOWNLOAD_URL:-https://github.com/bazelbuild/bazelisk/releases/download}` and verify with `sha256sum -c`.
   - Skip the download when `bazel` on `PATH` is bazelisk at the pinned version (`bazel --version` under bazelisk prints both).
   - Leave `BAZELISK_BASE_URL` to the caller; the script honours it because bazelisk does.
   - Under GitHub Actions (`GITHUB_PATH` set), append `BIN_DIR` to `$GITHUB_PATH`.
   - Install `uv` at a pinned version only while E01-02 is not merged, behind `--with-uv`. E01-02 deletes the flag.
2. Workflows (`ci.yml`, `docs.yml`, `vscode-extension.yml`): replace the setup-python, uv and bazelisk steps with `./tools/bootstrap.sh --with-uv`. Keep `actions/setup-node` in the extension workflow.
3. `Dockerfile`: drop `build-essential`, `gcc`, `g++` and both `uv` installs. `COPY tools/bootstrap.sh /tmp/` and run it once as root with `BIN_DIR=/usr/local/bin`. Fix `PATH` (no `~/.cargo/bin`).
4. `.dockerignore`: `bazel-*`, `**/node_modules`, `**/__pycache__`, `*.vsix`, `site/`, `.git` (the dev container bind-mounts the workspace, `devcontainer.json:20-21`).
5. `.claude/hooks/session-start.sh`: keep the proxy, registry and `lockfile_mode` lines; replace its own bazelisk download (session-start.sh:17, 27-32) with `BIN_DIR="$BIN_DIR" "$PROJECT_DIR/tools/bootstrap.sh"`.
6. `devcontainer.json:23`: drop `uv --version` from `postCreateCommand` once E01-02 lands (note it in that PR).

## Tests

- [ ] `shellcheck tools/bootstrap.sh` is clean.
- [ ] Running the script twice: the second run downloads nothing and exits 0.
- [ ] A wrong checksum makes the script fail before anything is installed (test by editing the constant locally; record in the PR).
- [ ] `docker build .` succeeds and `bazel version` in the container reports 8.3.0.
- [ ] Each workflow's log shows bazelisk v1.27.0.
- [ ] A fresh Claude Code cloud session reaches "Session start hook complete." and can run `bazel test //src/tools/ipu-as-py:test_assemble`.

## Acceptance Criteria

- [ ] `grep -rn "releases/latest" .github Dockerfile .claude tools` returns nothing.
- [ ] `grep -rn "cargo/bin" .github Dockerfile` returns nothing.
- [ ] Bazelisk is installed in exactly one place (`tools/bootstrap.sh`).
- [ ] No workflow sets up a host Python for Bazel's sake.

## Risks

- Pinned versions age. The header comment gives the upgrade procedure. Dependabot for GitHub Actions versions is optional and not part of this issue.
