#!/usr/bin/env bash
# Every file this branch deletes (not renames) since it left master must be named in
# planning/architecture-deepening/DELETIONS.md: as a row's path, or under a row's
# directory (a path ending in "/"). Prints the missing ones and exits 1; prints nothing
# and exits 0 when the log covers every deletion.
# Usage: tools/check_deletion_log.sh [base-ref]   (default: origin/master)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
base=$(git merge-base "${1:-origin/master}" HEAD)
log=planning/architecture-deepening/DELETIONS.md
# The first column of each table row, without its backticks.
mapfile -t logged < <(grep -oE '^\| `[^`]+`' "$log" | sed -E 's/^\| `//; s/`$//')
missing=0
while IFS= read -r path; do
  covered=0
  for entry in "${logged[@]}"; do
    if [ "$path" = "$entry" ] || { [ "${entry%/}" != "$entry" ] && [ "${path#"$entry"}" != "$path" ]; }; then
      covered=1; break
    fi
  done
  if [ $covered = 0 ]; then echo "not in $log: $path"; missing=1; fi
done < <(git diff -M --diff-filter=D --name-only "$base" HEAD)
exit $missing
