#!/usr/bin/env bash
# Compile the emulator's hot modules with mypyc, in place, then check the build.
#
#   tools/mypyc/build.sh CHECKOUT PYTHON
#
# CHECKOUT  an ipu-emulator checkout (this branch).
# PYTHON    the interpreter the compiled emulator will run under, with mypy 1.19
#           (mypyc) installed and the emulator's runtime packages (numpy, lark,
#           ...). The extensions are only valid for this Python version.
# RUN       optional command prefix for the heavy steps, e.g.
#           RUN="nice -n 10 taskset -c 19-63,83-127".
#
# What it does:
#  1. Compiles MODULES below (ipu_emu and ipu_isa together, so calls between
#     them are native) into extension modules next to their .py files in
#     src/ipu-emu/src/ipu_emu and src/ipu-isa/src/ipu_isa. CPython prefers an
#     extension over a .py of the same name, so the normal import paths pick
#     them up:
#         PYTHONPATH=<checkout>/src/ipu-emu/src:<checkout>/src/ipu-isa/src:...
#     The build's shared runtime library (<hash>__mypyc.*.so) is copied into
#     both of those src roots, so either one on sys.path is enough.
#     The emulator's .py files import mypy_extensions (mypyc_attr), so the
#     runtime environment needs it too (it comes with mypy).
#  2. Checks the build: fingerprint.py runs matmul_384x192_x128.default and
#     fully_connected.default (e4 and dynamic) in kernelgen's fast-path mode,
#     once compiled and once on a pure copy of the same .py files; every line
#     (passed, cycles, state hash) must be equal.
#  3. Writes <checkout>/mypyc-build.json: the commit, the Python version, and
#     the sha256 of every compiled module's .py (an edited .py is shadowed by
#     its stale extension: compare against this stamp, or rebuild).
# Exits non-zero if the build or any check fails. Rebuilding removes the old
# extensions first; `tools/mypyc/build.sh --clean CHECKOUT` only removes them.
set -euo pipefail

if [[ ${1:-} == --clean ]]; then
  CLEAN_ONLY=1
  shift
fi
CHECKOUT=$(cd "${1:?usage: build.sh CHECKOUT PYTHON}" && pwd)
PY=${2:-}
RUN=${RUN:-}
EMU=$CHECKOUT/src/ipu-emu/src
ISA=$CHECKOUT/src/ipu-isa/src
WORK=$CHECKOUT/build/mypyc

MODULES=(
  ipu_emu/regfile.py ipu_emu/issue.py ipu_emu/commit.py ipu_emu/resolver.py
  ipu_emu/cycle_view.py ipu_emu/cycle.py ipu_emu/geometry.py ipu_emu/cr_contents.py
  ipu_emu/ipu.py ipu_emu/ipu_state.py ipu_emu/xmem.py ipu_emu/numerics/codecs.py
  ipu_emu/slots/__init__.py ipu_emu/slots/aaq.py ipu_emu/slots/aaq_exact.py
  ipu_emu/slots/acc.py ipu_emu/slots/brk.py ipu_emu/slots/cond.py ipu_emu/slots/load.py
  ipu_emu/slots/lr.py ipu_emu/slots/mult.py
  ipu_isa/types.py ipu_isa/cr_halves.py
)
# 23 modules (21 of ipu_emu, 2 of ipu_isa); with the shared mypyc runtime
# library that makes 24 extension files.

clean() {
  find "$EMU/ipu_emu" "$ISA/ipu_isa" -name '*.so' -delete
  find "$EMU" "$ISA" -maxdepth 1 -name '*__mypyc*.so' -delete
  rm -rf "$WORK"
}

clean
[[ -n ${CLEAN_ONLY:-} ]] && { echo "removed the mypyc extensions"; exit 0; }
[[ -n $PY ]] || { echo "usage: build.sh CHECKOUT PYTHON" >&2; exit 2; }

PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
MYPYVER=$("$PY" -c 'import mypy.version; print(mypy.version.__version__)')
echo "python $PYVER ($PY), mypy $MYPYVER"
[[ $MYPYVER == 1.19.* ]] || echo "warning: built and checked with mypy 1.19, this is $MYPYVER" >&2

# --- 1. build -----------------------------------------------------------------
# ipu_emu and ipu_isa live under two src roots; mypyc writes each extension
# next to its module relative to the build directory, so build from a directory
# holding both packages (symlinks into the checkout: the extensions land in place).
ROOT=$WORK/root
mkdir -p "$ROOT"
ln -s "$EMU/ipu_emu" "$ROOT/ipu_emu"
ln -s "$ISA/ipu_isa" "$ROOT/ipu_isa"
OTHER_SRC=$(find "$CHECKOUT/src" -mindepth 2 -maxdepth 2 -name src -type d \
  ! -path "$EMU" ! -path "$ISA" | sort | tr '\n' ':')
(
  cd "$ROOT"
  MYPYPATH="${OTHER_SRC}$ROOT" $RUN "$PY" -m mypyc --python-version "$PYVER" \
    --ignore-missing-imports --cache-dir "$WORK/mypy-cache" "${MODULES[@]}"
) > "$WORK/mypyc.log" 2>&1 || { tail -40 "$WORK/mypyc.log"; echo "mypyc failed (log: $WORK/mypyc.log)" >&2; exit 1; }

shopt -s nullglob
shared=("$ROOT"/*__mypyc*.so)
[[ ${#shared[@]} -eq 1 ]] || { echo "expected one shared mypyc library, found ${#shared[@]}" >&2; exit 1; }
cp "${shared[0]}" "$EMU/"
cp "${shared[0]}" "$ISA/"
built=0
for m in "${MODULES[@]}"; do
  pkg_root=$EMU; [[ $m == ipu_isa/* ]] && pkg_root=$ISA
  ext=("$pkg_root/${m%.py}".*.so)
  [[ ${#ext[@]} -eq 1 ]] || { echo "no extension for $m" >&2; exit 1; }
  built=$((built + 1))
done
echo "built $built extension modules (+ $(basename "${shared[0]}"))"

# --- 2. check -----------------------------------------------------------------
PURE=$WORK/pure
mkdir -p "$PURE/emu" "$PURE/isa"
"$PY" - "$EMU" "$PURE/emu" "$ISA" "$PURE/isa" <<'EOF'
import shutil, sys
ignore = shutil.ignore_patterns("*.so", "__pycache__")
for src, dst in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)
EOF
ALL_SRC=$(find "$CHECKOUT/src" -mindepth 2 -maxdepth 2 -name src -type d | sort | tr '\n' ':')
COMPILED_PATH=${ALL_SRC%:}
PURE_PATH=$(echo "$COMPILED_PATH" | sed "s#$EMU#$PURE/emu#; s#$ISA#$PURE/isa#")
SPECS=(matmul_384x192_x128:default fully_connected:default:e4 fully_connected:default:dynamic)

loaded=$(cd "$CHECKOUT" && PYTHONPATH=$COMPILED_PATH "$PY" -c '
import ipu_emu.cycle, ipu_isa.types
print(ipu_emu.cycle.__file__.endswith(".so") and ipu_isa.types.__file__.endswith(".so"))')
[[ $loaded == True ]] || { echo "the compiled modules are not the ones imported" >&2; exit 1; }

cd "$CHECKOUT"
echo "fingerprints, compiled:"
PYTHONPATH=$COMPILED_PATH $RUN "$PY" "$CHECKOUT/tools/mypyc/fingerprint.py" "${SPECS[@]}" \
  | tee "$WORK/compiled.txt"
echo "fingerprints, pure:"
PYTHONPATH=$PURE_PATH $RUN "$PY" "$CHECKOUT/tools/mypyc/fingerprint.py" "${SPECS[@]}" \
  | tee "$WORK/pure.txt"
if ! diff -u "$WORK/pure.txt" "$WORK/compiled.txt"; then
  echo "CHECK FAILED: the compiled build differs from the pure .py" >&2
  exit 1
fi
[[ $(wc -l < "$WORK/compiled.txt") -eq ${#SPECS[@]} ]] || { echo "CHECK FAILED: missing cases" >&2; exit 1; }
if grep -q "passed=False" "$WORK/compiled.txt"; then
  echo "CHECK FAILED: a case did not pass" >&2
  exit 1
fi
echo "check passed: compiled == pure for ${#SPECS[@]} cases"

# --- 3. stamp -----------------------------------------------------------------
"$PY" - "$CHECKOUT" "${MODULES[@]}" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
checkout = Path(sys.argv[1])
modules = {}
for m in sys.argv[2:]:
    rel = ("src/ipu-isa/src/" if m.startswith("ipu_isa/") else "src/ipu-emu/src/") + m
    modules[rel] = hashlib.sha256((checkout / rel).read_bytes()).hexdigest()
commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                        capture_output=True, text=True, check=True).stdout.strip()
stamp = {"commit": commit, "python": sys.version, "modules": modules}
(checkout / "mypyc-build.json").write_text(json.dumps(stamp, indent=2) + "\n")
print(f"wrote {checkout / 'mypyc-build.json'} ({len(modules)} modules, commit {commit[:12]})")
EOF
