# AGG `dest_slot` and RESHAPE `source`/`dest` read post-LR-write

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-25 · Size S · Depends on: E04-12, E03 (timing contract declared in `ipu_isa`) · Behaviour change: yes (AGG.* `dest_slot` and ACC.RESHAPE `source`/`dest` see same-cycle LR writes; 3 `attn_v_*` kernels are fixed to keep their outputs)

## Goal

Apply Q32 rule 1: only the Control stage (LR, COND) reads LRs at cycle start; every other stage reads post-LR-write. Today two ACC-stage operands read the snapshot and contradict the RTL control spec (`docs/content/specs/stage-control.md` §5, §7.2).

## Background

- AGG.SUM, AGG.SUM.FIRST, AGG.MAX and AGG.MAX.FIRST declare `dest_slot` with `read: snapshot` in the spec; the emulator follows it.
- ACC.RESHAPE reads the LRD pairs `source` and `dest` from the snapshot (`_get_lrd_bytes(source, self._snapshot_buffer)`, ipu.py:1589-1590).
- Plan 01's bundle scan found 6 bundles in 3 kernels that co-issue `INC agg_slot 1` with an AGG on the same LR and rely on the snapshot. They are in:
  - `src/tools/ipu-apps/src/ipu_apps/kernels/attention/attn_v_16x60/attn_v_16x60.asm` (bundle at lines 129-134; the header comment at :59 states the snapshot dependency)
  - `.../attention/attn_v_256x36/attn_v_256x36.asm` (bundles at :128-132, :144-148, :165-169, :182-186)
  - `.../attention/attn_v_64x48/attn_v_64x48.asm` (bundle at :126-131)
  
  Re-run the scan on the current tree before editing; the count is approximate because the sources are Jinja templates.
- `docs/content/specs/stage-accumulator.md` documents the snapshot reads in its union layout and instruction reference (:123, :132-133, :308, :363-364).
- No kernel co-issues an LR write with ACC.RESHAPE on its `source`/`dest` pair (re-check with the same scan).

## Implementation

1. In `ipu_isa`, change the declared view of AGG.* `dest_slot` and ACC.RESHAPE `source`/`dest` to post-LR-write ("forwarded"). After E04-12 that is the whole emulator change.
2. Fix the 3 kernels so their outputs stay the same: either initialise `agg_slot` one lower so the forwarded value equals the old snapshot value, or move the `INC` to the next bundle if that costs no cycle. Choose per kernel and state the cycle-count effect.
3. Add the bundle scan as `tools/audit_cross_slot_hazards.py` (plan 01) so later flips can rerun it, and list its output in the PR.
4. Update the `attn_v_*` header comments, `stage-accumulator.md` (:123, :132-133, :308, :363-364: "snapshot value" → "post-LR-write value") and the generated forwarding table (E03 docs generation).

## Tests

- [ ] E02 timing characterization rows for AGG `dest_slot` and RESHAPE `source`/`dest` updated to forwarded; the generated conformance test (E04-12) follows the spec.
- [ ] The 3 `attn_v_*` kernel tests pass with unchanged outputs; cycle counts reported.
- [ ] All other E02 kernel output digests unchanged.
- [ ] Conformance traces (E04-31, if landed) regenerated and the diff reviewed.

## Acceptance Criteria

- [ ] No ACC-stage operand reads LRs at cycle start.
- [ ] `stage-control.md` §5/§7.2, `stage-accumulator.md` and the ISA forwarding table agree.
- [ ] The audit script is committed and its output is in the PR.
