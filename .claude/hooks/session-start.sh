#!/bin/bash
# SessionStart hook: prepare a Claude Code on the web container so that
# `bazel build` / `bazel test` work out of the box.
set -euo pipefail

# Only needed in the remote (web) environment. Local checkouts use the
# devcontainer image, which already ships bazelisk and uv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# The sandbox egress policy allows github.com but not releases.bazel.build or
# bcr.bazel.build, so bazelisk, the Bazel binary and the module registry are all
# pointed at the equivalent GitHub-hosted sources.
BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64"
export BAZELISK_BASE_URL="https://github.com/bazelbuild/bazel/releases/download"

if [ -w /usr/local/bin ]; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.local/bin"
fi
mkdir -p "$BIN_DIR"

# Idempotent: skip the download when a working bazel is already on PATH.
if ! command -v bazel >/dev/null 2>&1; then
  echo "Installing bazelisk into $BIN_DIR ..."
  curl -fsSL -o "$BIN_DIR/bazel" "$BAZELISK_URL"
  chmod +x "$BIN_DIR/bazel"
fi
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
BAZELRC_MARKER="# claude-code-on-the-web session-start hook"
if ! grep -qF "$BAZELRC_MARKER" "$HOME/.bazelrc" 2>/dev/null; then
  cat >> "$HOME/.bazelrc" << RC

$BAZELRC_MARKER
common --registry=https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/
common --lockfile_mode=off
RC
fi

# Bazel actions run repository shell scripts; keep them executable.
find "$PROJECT_DIR" -type f -name "*.sh" -exec chmod +x {} + 2>/dev/null || true

# Warm the container: download Bazel itself plus every external dependency
# (module registry entries and pip wheels) while the image is still being built.
cd "$PROJECT_DIR"
bazel version
bazel build //... --build_tag_filters=-docs || true

# Persist the settings the rest of the session needs.
if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -q "^export BAZELISK_BASE_URL=" "$CLAUDE_ENV_FILE" 2>/dev/null; then
  {
    echo "export PATH=\"$BIN_DIR:\$PATH\""
    echo "export BAZELISK_BASE_URL=\"$BAZELISK_BASE_URL\""
  } >> "$CLAUDE_ENV_FILE"
fi

echo "Session start hook complete."
