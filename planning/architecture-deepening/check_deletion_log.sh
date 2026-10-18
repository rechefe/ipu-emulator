#!/usr/bin/env bash
# Every file this branch deletes (not renames) since it left master must be named in
# planning/architecture-deepening/DELETIONS.md: as a row's path, or under a row's
# directory (a path ending in "/"). Prints the missing ones and exits 1; prints nothing
# and exits 0 when the log covers every deletion.
# Usage: planning/architecture-deepening/check_deletion_log.sh [base-ref]   (default: origin/master)
# The rename limit is raised: the branch deletes and renames over a thousand
# files, past which git stops detecting renames and lists them as deletions.
# A file under a moved directory (DIR_MOVES, "old/=new/") that this branch also
# rewrote past git's rename threshold still exists at its new path: a rename.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
base=$(git merge-base "${1:-origin/master}" HEAD)
log=planning/architecture-deepening/DELETIONS.md
# The first column of each table row, without its backticks.
mapfile -t logged < <(grep -oE '^\| `[^`]+`' "$log" | sed -E 's/^\| `//; s/`$//')
DIR_MOVES=(
  "src/tools/ipu-as-py/=src/ipu-as/"
  "src/tools/ipu-emu-py/=src/ipu-emu/"
  "src/tools/=src/"
)
missing=0
while IFS= read -r path; do
  covered=0
  for move in "${DIR_MOVES[@]}"; do
    old=${move%%=*}; new=${move#*=}
    if [ "${path#"$old"}" != "$path" ] && git cat-file -e "HEAD:$new${path#"$old"}" 2>/dev/null; then
      covered=1; break
    fi
  done
  if [ $covered = 1 ]; then continue; fi
  for entry in "${logged[@]}"; do
    if [ "$path" = "$entry" ] || { [ "${entry%/}" != "$entry" ] && [ "${path#"$entry"}" != "$path" ]; }; then
      covered=1; break
    fi
  done
  if [ $covered = 0 ]; then echo "not in $log: $path"; missing=1; fi
done < <(git -c diff.renameLimit=32767 diff -M --diff-filter=D --name-only "$base" HEAD)
exit $missing
