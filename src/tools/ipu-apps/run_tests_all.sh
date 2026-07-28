#!/usr/bin/env bash
# Run every kernel test on ZDlinear (assembles first, then pytest per app).
# Usage (from src/tools/ipu-apps/):  bash run_tests_all.sh [bin_dir]
set -uo pipefail

BIN_DIR="${1:-/tmp/bins}"
bash assemble_all.sh "$BIN_DIR" >/dev/null || { echo "assembly failed"; exit 1; }

pass=0; fail=0; faillist=""
for t in test/test_*.py; do
  name=$(basename "$t" .py); name=${name#test_}
  # Env var prefix is the uppercased app name, except fully_connected -> FC.
  if [ "$name" = "fully_connected" ]; then prefix="FC"; else
    prefix=$(echo "$name" | tr '[:lower:]' '[:upper:]'); fi

  out=$(env "${prefix}_INST_BIN=$BIN_DIR/$name.bin" \
            "${prefix}_DATA_DIR=src/ipu_apps/$name/test_data_format" \
            uv run pytest "$t" -q 2>&1)
  # Require a clean summary line: a bare "N passed" with no failures or errors.
  # Matching only "passed" would let "3 passed, 1 failed" through as a pass.
  if echo "$out" | grep -qE "^[0-9]+ passed" && ! echo "$out" | grep -qE "[0-9]+ (failed|error)"; then
    echo "PASS  $name :: $(echo "$out" | grep -oE '[0-9]+ passed')"
    pass=$((pass+1))
  else
    echo "FAIL  $name"
    echo "$out" | tail -15 | sed 's/^/        /'
    fail=$((fail+1)); faillist="$faillist $name"
  fi
done

echo
echo "=== $pass passed, $fail failed ==="
[ -n "$faillist" ] && echo "failing:$faillist"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
