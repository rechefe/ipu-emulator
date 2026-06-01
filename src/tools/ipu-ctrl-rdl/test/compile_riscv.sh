#!/usr/bin/env bash
# Compile the generated ipu-pac crate for bare-metal RISC-V (smoke test).
set -euo pipefail

find_lib_rs() {
  local candidate
  for candidate in \
    "${TEST_SRCDIR:-}/_main/src/tools/ipu-ctrl-rdl/lib.rs" \
    "${TEST_SRCDIR:-}/lib.rs" \
    "${RUNFILES_DIR:-}/_main/src/tools/ipu-ctrl-rdl/lib.rs" \
    "$(dirname "$0")/../lib.rs"
  do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  if [[ -n "${RUNFILES_DIR:-}" ]]; then
    find "${RUNFILES_DIR}" -path '*/ipu-ctrl-rdl/lib.rs' -print -quit
    return 0
  fi
  return 1
}

LIB_RS="$(find_lib_rs || true)"
if [[ -z "${LIB_RS}" || ! -f "${LIB_RS}" ]]; then
  echo "could not locate generated lib.rs in runfiles"
  exit 1
fi

TARGET="${RISCV_TARGET:-riscv32imac-unknown-none-elf}"

ensure_rust() {
  if command -v rustup >/dev/null 2>&1 && rustup run stable rustc --version >/dev/null 2>&1; then
    return 0
  fi
  if command -v rustc >/dev/null 2>&1 && rustc --version >/dev/null 2>&1; then
    return 0
  fi
  if command -v rustup >/dev/null 2>&1; then
    rustup default stable
    return 0
  fi
  echo "Rust toolchain not available; skipping RISC-V PAC compile smoke test"
  exit 0
}

ensure_rust

RUSTC=(rustc)
if command -v rustup >/dev/null 2>&1; then
  RUSTC=(rustup run stable rustc)
fi

if command -v rustup >/dev/null 2>&1; then
  if ! rustup target list --installed | grep -q "^${TARGET}$"; then
    rustup target add "${TARGET}"
  fi
fi

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "${OUT_DIR}"' EXIT

"${RUSTC[@]}" \
  --edition 2021 \
  --crate-type lib \
  --target "${TARGET}" \
  -C panic=abort \
  -C relocation-model=static \
  -o "${OUT_DIR}/libipu_pac.rlib" \
  "${LIB_RS}"

echo "ok: ipu_pac compiles for ${TARGET}"
