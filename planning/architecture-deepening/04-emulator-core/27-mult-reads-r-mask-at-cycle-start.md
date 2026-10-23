# MULT reads R_MASK at cycle start

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-27 · Size S · Depends on: E04-12, E03 (timing contract) · Behaviour change: yes (a same-word LD.MASK is no longer visible to MULT)

## Goal

Apply Q20 and Q32 rule 2: MULT reads R0, R1, R_CYCLIC and R_MASK at cycle start. R_MASK is the only one of the four that MULT reads post-load today.

## Background

- MULT reads R0/R1 and R_CYCLIC from the snapshot (issue #157; for example ipu.py:1235, 1363, 1392).
- The mask path reads the live register: `_effective_mult_mask` (ipu.py:937) and the prototyping all-ones shortcut `self.state.regfile.raw("r_mask")[slot:slot + 16]` (ipu.py:909). So a same-word LD.MASK (formerly `LDR_MULT_MASK_REG`) is visible to MULT, while a same-word LD.MULT is not.
- Plan 01's scan found **0** bundles that co-issue LD.MASK with MULT, so no kernel output changes.

## Implementation

1. In `ipu_isa`, declare MULT's implicit read of `r_mask` as snapshot. After E04-12, `_effective_mult_mask` reads through `CycleView`.
2. Make sure the prototyping shortcut reads the same view (it goes through `CycleView` after E04-12; confirm with the equivalence test).
3. Rerun `tools/audit_cross_slot_hazards.py` and put the output (expected: 0 bundles) in the PR.
4. Update the stage-control/MULT docs that describe R_MASK timing.

## Tests

- [ ] A word with LD.MASK and MULT uses the previous R_MASK; the next word sees the new one.
- [ ] Same result in faithful and prototyping modes (`--define=ipu_proto=1`).
- [ ] E02 timing characterization row updated; E02 kernel output digests unchanged.

## Acceptance Criteria

- [ ] All four MULT data inputs (R0, R1, R_CYCLIC, R_MASK) are declared and read at cycle start.
