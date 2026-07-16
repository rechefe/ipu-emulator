# Full CTRL-stage integration, app-level parity suite, sign-off

Part of the [CTRL-stage RTL epic](00-epic.md). Depends on #3, #4, #5, #6.

## Goal

Wire `ctrl_lr_regfile`/`ctrl_lr_alu`/`ctrl_cr_regfile`/`ctrl_fetch`/
`ctrl_cond`/`ctrl_dispatch`/`ctrl_apb_slave` into the top-level `ctrl_stage.sv`
matching the boundary in `stage-control.md` §2/§3, and prove it end-to-end by
running a real assembled program through both the RTL simulation and the
Python emulator and diffing full execution traces, not just per-instruction
unit results.

## Integration test approach

- Pick an existing `ipu-apps` program (e.g. `fully_connected`,
  `src/tools/ipu-apps/src/ipu_apps/fully_connected/`) and assemble it the
  normal way (`bazel run //src/tools/ipu-as-py:ipu-as`).
- Run it once through `ipu_emu` directly (however #2's harness already does
  this per-instruction, extended here to a full program) and capture, per
  cycle: PC, the four dispatch-bus contents, `xmem_read_addr`/`xmem_read_en`,
  and the full LR file.
- Run the same assembled binary through the cocotb testbench driving
  `ctrl_stage.sv`, capturing the same per-cycle signals.
- Diff cycle-for-cycle. Any divergence is either a real RTL bug or a
  remaining spec/emulator gap #1 didn't catch — both are epic-blocking
  findings, not test flakiness to retry past.
- Given #1's finding that IMEM is unbanked/flat in the emulator
  (`INST_MEM_SIZE = 1024`) versus banked in the RTL (`IMEM_DEPTH = 256`,
  `PC_W = 7`), the chosen test program must fit within 128 entries (one
  active bank) so both models can run it without invoking banking/APB
  behavior that only the RTL implements — this bounds which existing
  `ipu-apps` programs are eligible; check program length before picking one.

## Sign-off checklist

- [ ] `bazel test //src/hw/ipu-ctrl-rtl/...` runs the full parity suite
      (#3–#6's per-block tests plus this issue's program-level test) in CI.
- [ ] At least one full `ipu-apps` program produces byte-identical dispatch
      traces and final LR/CR state between RTL and emulator.
- [ ] `stage-control.md` is re-read end to end against the finished RTL and
      any remaining prose/implementation gaps are fixed (this is the second,
      final pass beyond #1's up-front one — implementation always surfaces
      a few things planning missed).
- [ ] The epic's "Known spec/emulator discrepancies" list (in `00-epic.md`)
      is updated to reflect what was actually resolved vs. deliberately
      deferred (e.g. APB/banking parity, which #6 explicitly couldn't test
      against the emulator).

## Acceptance Criteria

- [ ] `ctrl_stage.sv` integrates every sub-block behind the exact port list
      in `stage-control.md` §3.1/§3.2.
- [ ] Program-level parity test passes for at least one real application.
- [ ] All epic-level acceptance criteria in `00-epic.md` are met or have an
      explicit, recorded reason they're deferred.
