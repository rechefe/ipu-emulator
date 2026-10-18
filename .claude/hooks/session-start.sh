#!/bin/bash
# SessionStart hook: prepare a Claude Code on the web container so that
# `bazel build` / `bazel test` work out of the box.
set -euo pipefail

# Only needed in the remote (web) environment. Local checkouts use the
# devcontainer image, which already ships bazelisk.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# The sandbox egress policy allows github.com but not releases.bazel.build or
# bcr.bazel.build, so the Bazel binary and the module registry are pointed at
# the equivalent GitHub-hosted sources (bootstrap.sh already fetches bazelisk
# from GitHub releases).
export BAZELISK_BASE_URL="https://github.com/bazelbuild/bazel/releases/download"

if [ -w /usr/local/bin ]; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.local/bin"
fi
mkdir -p "$BIN_DIR"

# Pinned, checksummed bazelisk from the shared bootstrap script (idempotent).
BIN_DIR="$BIN_DIR" "$PROJECT_DIR/tools/bootstrap.sh"
export PATH="$BIN_DIR:$PATH"

# bazelisk keys its download cache by base URL, so persist the override in the
# user-level bazeliskrc too — a shell that does not inherit the exported
# variable would otherwise fall back to the blocked releases.bazel.build.
if ! grep -q "^BAZELISK_BASE_URL=" "$HOME/.bazeliskrc" 2>/dev/null; then
  echo "BAZELISK_BASE_URL=$BAZELISK_BASE_URL" >> "$HOME/.bazeliskrc"
fi

# Environment-specific Bazel flags live in the user bazelrc so the repository's
# own .bazelrc (used by CI and by developer machines) stays untouched.
#   --registry      : bcr.bazel.build is blocked; use the GitHub mirror.
#   --lockfile_mode : the mirror URL differs from the one recorded in
#                     MODULE.bazel.lock, so leave the committed lockfile alone.
#   --nobuild_runfile_links : do not materialise runfiles trees at build time.
#                     Each kernel test target links large runfiles, and
#                     `//...` alone fills the session's disk allowance.
BAZELRC_MARKER="# claude-code-on-the-web session-start hook"
if ! grep -qF "$BAZELRC_MARKER" "$HOME/.bazelrc" 2>/dev/null; then
  cat >> "$HOME/.bazelrc" << RC

$BAZELRC_MARKER
common --registry=https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/
common --lockfile_mode=off
RC
fi
if ! grep -q "^build --nobuild_runfile_links" "$HOME/.bazelrc" 2>/dev/null; then
  echo "build --nobuild_runfile_links" >> "$HOME/.bazelrc"
fi

# Bazel actions run repository shell scripts; keep them executable.
find "$PROJECT_DIR" -type f -name "*.sh" -exec chmod +x {} + 2>/dev/null || true

# Warm the container: download Bazel itself plus every external dependency
# (module registry entries and pip wheels) while the image is still being built.
# Fetch only; a full build here costs tens of GB of disk before work starts.
cd "$PROJECT_DIR"
bazel version
bazel fetch --config=ci //... || true

# Persist the settings the rest of the session needs.
if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -q "^export BAZELISK_BASE_URL=" "$CLAUDE_ENV_FILE" 2>/dev/null; then
  {
    echo "export PATH=\"$BIN_DIR:\$PATH\""
    echo "export BAZELISK_BASE_URL=\"$BAZELISK_BASE_URL\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "Session start hook complete."
