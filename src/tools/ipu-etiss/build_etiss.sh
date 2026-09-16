#!/usr/bin/env bash
# Build ETISS and the IPU architecture plugin from scratch.
#
#   ./src/tools/ipu-etiss/build_etiss.sh [PREFIX]
#
# PREFIX defaults to build/etiss under the repository root.  On success it
# prints the export line that points the Python backend and the parity tests at
# the runner:
#
#   export IPU_ETISS_RUN=<PREFIX>/install/bin/ipu_etiss_run
#
# Requirements: cmake >= 3.13, a C++17 compiler, and the Boost development
# packages ETISS needs:
#
#   sudo apt-get install -y cmake build-essential \
#       libboost-system-dev libboost-filesystem-dev libboost-program-options-dev
#
# ETISS is fetched at a pinned commit and patched (see patches/) so that
# instructions wider than 32 bits survive the fetch path; without that patch the
# 186-bit IPU VLIW word is silently truncated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PREFIX="${1:-${REPO_ROOT}/build/etiss}"
PATCH_DIR="${REPO_ROOT}/src/tools/ipu-etiss/patches"

# Pinned so the patch applies and behaviour is reproducible.
ETISS_REPO="https://github.com/tum-ei-eda/etiss.git"
ETISS_COMMIT="74451e05512e10c2e8574b914e7dc8a19de67975"

SRC="${PREFIX}/etiss-src"
BUILD="${PREFIX}/etiss-build"
INSTALL="${PREFIX}/install"
PLUGIN_BUILD="${PREFIX}/ipu-build"

PYTHON="${PYTHON:-python3}"
JOBS="$(nproc 2>/dev/null || echo 4)"

echo "==> ETISS source: ${SRC}"
if [ ! -d "${SRC}/.git" ]; then
    mkdir -p "$(dirname "${SRC}")"
    git clone "${ETISS_REPO}" "${SRC}"
fi
git -C "${SRC}" fetch --depth 1 origin "${ETISS_COMMIT}" 2>/dev/null || git -C "${SRC}" fetch origin
git -C "${SRC}" checkout --force "${ETISS_COMMIT}"
git -C "${SRC}" submodule update --init --depth 1 --recursive

echo "==> Applying patches"
git -C "${SRC}" checkout -- include src
for patch in "${PATCH_DIR}"/*.patch; do
    [ -e "${patch}" ] || continue
    echo "    $(basename "${patch}")"
    git -C "${SRC}" apply "${patch}"
done

echo "==> Building ETISS"
cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${INSTALL}"
cmake --build "${BUILD}" --parallel "${JOBS}"
cmake --install "${BUILD}"

echo "==> Building the IPU plugin"
cmake -S "${REPO_ROOT}/src/tools/ipu-etiss" -B "${PLUGIN_BUILD}" \
      -DETISS_ROOT="${INSTALL}" -DIPU_PYTHON="${PYTHON}"
cmake --build "${PLUGIN_BUILD}" --parallel "${JOBS}"
cmake --install "${PLUGIN_BUILD}"

# ETISS's own install rewrites plugins/list.txt, so de-duplicate after both.
sort -u "${INSTALL}/lib/plugins/list.txt" -o "${INSTALL}/lib/plugins/list.txt"

echo
echo "Done. Point the Python backend and the parity tests at the runner with:"
echo
echo "    export IPU_ETISS_RUN=${INSTALL}/bin/ipu_etiss_run"
echo
