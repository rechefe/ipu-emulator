# ACC.RESHAPE reads this cycle's MULT_RES

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-26 · Size S · Depends on: E04-12, E03 (timing contract) · Behaviour change: yes (ACC.RESHAPE consumes the MULT result of its own word instead of the previous word's)

## Goal

Apply Q19 and Q32 rule 3: MULT_RES produced by MULT is seen by ACC, AGG and RESHAPE in the same cycle. RESHAPE is the only ACC-stage op that reads the previous cycle's MULT_RES today.

## Background

- ACC.RESHAPE reads `self._snapshot_buffer.raw("mult_res")` (ipu.py:1603); every other ACC/AGG op reads the live MULT_RES.
- `docs/content/specs/stage-accumulator.md` contradicts itself: :87 says ACC consumes "this cycle's multiply result"; :186 says RESHAPE uses the "pre-instruction MULT_RES snapshot", with a rationale ("scattered elements do not chain") that does not hold, because source (MULT_RES) and destination (R_ACC) are different registers.
- `test_reshape_mult_res_reads_from_snapshot` (test_execute.py:1886) pins the snapshot read.
- Plan 01's scan found **0** bundles that co-issue MULT with ACC.RESHAPE, so no kernel output changes.

## Implementation

1. In `ipu_isa`, change ACC.RESHAPE's implicit read of `mult_res` to forwarded. After E04-12 this is the only code change.
2. Rewrite `test_reshape_mult_res_reads_from_snapshot` as `test_reshape_reads_same_cycle_mult_res`.
3. Fix `stage-accumulator.md:184-190` and the pseudocode at :368 (`// pre-instruction snapshot`) to match :87, and drop the rationale.
4. Rerun `tools/audit_cross_slot_hazards.py` (E04-25) and put the output (expected: 0 bundles) in the PR.

## Tests

- [ ] A word with `MULT.*` and `ACC.RESHAPE` scatters the product of the same word.
- [ ] E02 timing characterization row updated; the generated conformance test follows the spec.
- [ ] E02 kernel output digests unchanged.

## Acceptance Criteria

- [ ] All ACC-stage consumers of MULT_RES read the same-cycle value.
- [ ] `stage-accumulator.md` no longer contradicts itself.
