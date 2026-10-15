# Register reads keep constant-ones provenance

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-24 · Size S · Depends on: E04-04, E04-11 · Behaviour change: yes (statistics only: MAC-identity and alias counts can rise; architectural state unchanged)

## Goal

A register read never changes what the profiler knows. Remove the `forget_provenance` calls that E04-04 kept to stay output-identical.

## Background

- Before E04-04, read-only uses of `RegFile.raw()` dropped constant-ones provenance: `r_mask` (ipu.py:909) and the snapshot's `r_acc`/`mult_res` (ipu.py:1421, 1448, 1476, 1603, 1668, 1701). Once dropped on the snapshot, `copy_into` keeps filtering provenance for those names, so a constant-ones R_ACC is never detected as an identity operand again.
- E04-04 replaced those reads with `view()` plus an explicit `forget_provenance(name)` marked `# E04-24`.

## Implementation

1. Delete every `forget_provenance` call marked `# E04-24`, and the method if unused.
2. Run the full suite with statistics output and list every changed counter (kernel, counter, before, after) in the PR.

## Tests

- [ ] New regression test: load constant ONES into R_ACC, run ACC.ADD, and show the identity is detected in the following cycle (fails before this change).
- [ ] New: a prototyping run does not pin R_MASK (observable: provenance survives the mask shortcut).
- [ ] `test_stats.py`, `test_alias_profile.py`, `test_alias_tracking.py` updated only where the PR's table shows a justified change.
- [ ] E02 kernel output digests unchanged (architectural state is not affected).

## Acceptance Criteria

- [ ] No `forget_provenance` call remains on a read path.
- [ ] Every statistics change is listed and explained in the PR.
