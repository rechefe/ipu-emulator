#!/usr/bin/env bash
# Regenerate tools/requirements.txt from all tool pyproject.toml files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv pip compile \
  src/tools/ipu-as-py/pyproject.toml \
  src/tools/ipu-emu-py/pyproject.toml \
  src/tools/ipu-apps/pyproject.toml \
  src/tools/ipu-common/pyproject.toml \
  src/tools/ipu-ctrl-rdl/pyproject.toml \
  --extra=docs \
  --extra=dev \
  --python-version=3.10 \
  --generate-hashes \
  -o tools/requirements.txt

python3 - <<'PY'
from pathlib import Path

path = Path("tools/requirements.txt")
lines = path.read_text().splitlines()
filtered = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("-e ") or stripped.startswith("--editable"):
        continue
    if "../ipu-" in stripped or (stripped.startswith("../") and "ipu-" in stripped):
        continue
    filtered.append(line)
path.write_text("\n".join(filtered) + "\n")
print(f"Wrote {path} ({len(filtered)} lines)")
PY
