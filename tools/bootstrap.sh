#!/usr/bin/env bash
# Install the pinned build toolchain. Idempotent: a second run downloads nothing.
#
# Called by every environment that builds this repository: the GitHub workflows,
# the Dockerfile (dev container) and the Claude Code SessionStart hook. Bazel
# itself is pinned by .bazelversion; this script pins bazelisk, which reads it.
#
# Usage: tools/bootstrap.sh [--with-uv]
#
#   --with-uv  Also install uv. Needed only while tools/python_deps.bzl runs
#              `uv pip compile` at fetch time (removed by E01-02).
#
# Environment:
#   BIN_DIR                 Install directory. Default /usr/local/bin, or
#                           ~/.local/bin when /usr/local/bin is not writable.
#   BAZELISK_DOWNLOAD_URL   Base URL of the bazelisk releases.
#   UV_DOWNLOAD_URL         Base URL of the uv releases.
#   BAZELISK_BASE_URL       Not read here; bazelisk honours it when it fetches
#                           Bazel, so callers that need a mirror export it.
#   GITHUB_PATH             When set (GitHub Actions), BIN_DIR is appended to it.
set -euo pipefail

# Pinned toolchain. To upgrade: change the version and both checksums,
# run the script once locally, and open a PR (CI runs it on every job).
BAZELISK_VERSION=v1.27.0
BAZELISK_SHA256_LINUX_AMD64=e1508323f347ad1465a887bc5d2bfb91cffc232d11e8e997b623227c6b32fb76
BAZELISK_SHA256_LINUX_ARM64=bb608519a440d45d10304eb684a73a2b6bb7699c5b0e5434361661b25f113a5d

UV_VERSION=0.11.32
UV_SHA256_LINUX_X86_64=aab924fd522efd06f1c5f3b93a243864fc453132c94b2dc49f1371b528a4b967
UV_SHA256_LINUX_AARCH64=4d4fa08d95b06642e5800df6a22bd71455f23f988269e18da2847971d8c0bf31

BAZELISK_DOWNLOAD_URL="${BAZELISK_DOWNLOAD_URL:-https://github.com/bazelbuild/bazelisk/releases/download}"
UV_DOWNLOAD_URL="${UV_DOWNLOAD_URL:-https://github.com/astral-sh/uv/releases/download}"

log() { echo "bootstrap: $*" >&2; }
die() { log "error: $*"; exit 1; }

with_uv=false
for arg in "$@"; do
  case "$arg" in
    --with-uv) with_uv=true ;;
    -h | --help)
      sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) die "unknown argument: $arg (see --help)" ;;
  esac
done

[ "$(uname -s)" = Linux ] || die "only Linux is supported (got $(uname -s))"
case "$(uname -m)" in
  x86_64 | amd64)
    bazelisk_asset=bazelisk-linux-amd64
    bazelisk_sha256=$BAZELISK_SHA256_LINUX_AMD64
    uv_target=x86_64-unknown-linux-gnu
    uv_sha256=$UV_SHA256_LINUX_X86_64
    ;;
  aarch64 | arm64)
    bazelisk_asset=bazelisk-linux-arm64
    bazelisk_sha256=$BAZELISK_SHA256_LINUX_ARM64
    uv_target=aarch64-unknown-linux-gnu
    uv_sha256=$UV_SHA256_LINUX_AARCH64
    ;;
  *) die "unsupported architecture: $(uname -m)" ;;
esac

for tool in curl sha256sum tar install; do
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' is required"
done

if [ -n "${BIN_DIR:-}" ]; then
  mkdir -p "$BIN_DIR"
  [ -w "$BIN_DIR" ] || die "BIN_DIR=$BIN_DIR is not writable"
elif [ -w /usr/local/bin ]; then
  BIN_DIR=/usr/local/bin
else
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

# download URL FILE SHA256: fetch into the scratch directory and verify it.
# Nothing is installed unless the checksum matches.
download() {
  local url=$1 file=$2 sha256=$3
  log "downloading $url"
  curl -fsSL --retry 3 -o "$tmp_dir/$file" "$url"
  if ! (cd "$tmp_dir" && echo "$sha256  $file" | sha256sum -c --quiet - >/dev/null 2>&1); then
    die "checksum mismatch for $url (expected $sha256)"
  fi
}

# install_bin SRC NAME: replace BIN_DIR/NAME atomically, so a running binary or
# an interrupted install never leaves a truncated file behind.
install_bin() {
  install -m 0755 "$1" "$BIN_DIR/.$2.bootstrap"
  mv -f "$BIN_DIR/.$2.bootstrap" "$BIN_DIR/$2"
}

has_sha256() {
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

# --- bazelisk, installed as `bazel` ------------------------------------------
# Checked by checksum rather than `bazel --version`, which would make bazelisk
# download Bazel and does not print the bazelisk version.
on_path="$(command -v bazel 2>/dev/null || true)"
if has_sha256 "$BIN_DIR/bazel" "$bazelisk_sha256"; then
  log "bazelisk $BAZELISK_VERSION already installed at $BIN_DIR/bazel"
elif [ -n "$on_path" ] && has_sha256 "$on_path" "$bazelisk_sha256"; then
  log "bazelisk $BAZELISK_VERSION already installed at $on_path"
else
  download "$BAZELISK_DOWNLOAD_URL/$BAZELISK_VERSION/$bazelisk_asset" bazelisk "$bazelisk_sha256"
  install_bin "$tmp_dir/bazelisk" bazel
  log "installed bazelisk $BAZELISK_VERSION at $BIN_DIR/bazel"
fi

# --- uv (until E01-02) -------------------------------------------------------
if [ "$with_uv" = true ]; then
  if [ -x "$BIN_DIR/uv" ] && [ "$("$BIN_DIR/uv" --version 2>/dev/null | cut -d' ' -f2)" = "$UV_VERSION" ]; then
    log "uv $UV_VERSION already installed at $BIN_DIR/uv"
  else
    archive="uv-$uv_target.tar.gz"
    download "$UV_DOWNLOAD_URL/$UV_VERSION/$archive" "$archive" "$uv_sha256"
    tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"
    install_bin "$tmp_dir/uv-$uv_target/uv" uv
    install_bin "$tmp_dir/uv-$uv_target/uvx" uvx
    log "installed uv $UV_VERSION at $BIN_DIR/uv"
  fi
fi

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$BIN_DIR" >> "$GITHUB_PATH"
fi
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) log "note: $BIN_DIR is not on PATH; add it to use the installed tools" ;;
esac
