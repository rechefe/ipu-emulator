# Bug: wide-mode R0/R1 mult read uses snapshot, not live

## Background
A same-cycle `LDR_MULT_REG` into R0/R1 followed by a mult that consumes it
(`MULT.RC.VV`, `MULT.VE`, `MULT.EE`) returns stale data in **wide-vector mode**
only. Normal mode reads R0/R1 **live**, so the same-cycle load is visible (the
conv apps rely on this). The two modes disagree; **live is correct**.

Cause: in wide mode the mult handlers read lanes from
`_debug_mult_stage_vectors_snap` (the start-of-cycle snapshot), while
`LDR_MULT_REG` writes `_debug_mult_stage_vectors` (live). See
`ipu.py:_resolve_operand` (MultStageReg, wide branch returns the index) and the
mult handlers' `_debug_ra_lane_vals`, which reads `..._snap`. The r_cyclic path
is unaffected (read live in both modes).

Impact: forces a wasted separate-cycle load before every mult in wide-mode apps
(e.g. `softmax_rows` loads x in its own cycle in all 4 passes).

## Change
Make wide-mode mult-stage reads observe the **live** `_debug_mult_stage_vectors`
written this cycle, matching normal mode. Simplest fix: have `_debug_ra_lane_vals`
read live staging instead of the `_snap` copy (or drop the R0/R1 snapshotting in
`execute_vliw_cycle` for the mult-stage vectors). Keep r_cyclic semantics as-is.

## Checklist
- [ ] `_debug_ra_lane_vals` (and any other `_debug_mult_stage_vectors_snap`
      reader) reads live staging, so same-cycle `LDR_MULT_REG` → mult is visible.
- [ ] Add wide-mode test: `LDR_MULT_REG r0` + same-cycle `MULT.RC.VV`/`MULT.VE`
      sees the freshly loaded data (currently returns 0).
- [ ] Confirm normal-mode behavior unchanged; `bazel test //...` green.
- [ ] Refactor `softmax_rows.asm` to fuse load+mult (remove the separate-cycle
      loads in all 4 passes); re-run `test_softmax_rows`.
