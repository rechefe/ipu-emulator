# End-to-end firmware-driven app run, parity tests, and docs

Part of the [RISC-V host integration epic](00-epic.md). Depends on #2–#5.

## Goal

Tie everything together: a **Rust firmware** image running on the emulated
RISC-V core (#4/#5) configures the IPU, streams an assembled program into
instruction memory, starts it, polls for halt, reads/writes the PC, and can
reset and re-run — all through the SystemRDL control block (#1/#2). Prove parity
with the existing direct-Python path and document the workflow.

## Implementation

- Pick an existing app (e.g. `fully_connected`) as the reference workload.
- Author firmware (using `ipu-rt` from #5) that reproduces that app's setup
  (CRs, dtype, dstructure) and program load, then `start()` + `wait_until_halted()`.
- Add an integration harness/test that:
  1. Assembles the app program (existing Bazel `assemble_*` target).
  2. Boots the Unicorn core with the firmware and the program/inputs available.
  3. Runs to completion via the host path.
  4. Asserts the resulting `IpuState`/XMEM output equals the direct-Python run.
- Exercise the full control surface in tests: start, halt/step, PC read+write,
  reprogram IMEM, and reset-preserving-IMEM (re-run with new inputs).

## Docs

- Add a user-facing page (or expand the spec) showing the host-driven workflow
  end to end, including how to build firmware and read back results.
- Link it from the docs nav alongside the design spec.

## Tests

- [ ] End-to-end host-path run matches the direct-Python result for the chosen app.
- [ ] Halt/step/PC-R/W/reset are each exercised and asserted.
- [ ] Reset-preserving-IMEM allows a second run with fresh inputs, no re-stream.
- [ ] `bazel test //...` is green, including the new integration target.

## Acceptance Criteria

- [ ] A Rust-firmware-driven run of a real app produces identical results to the
      direct-Python path.
- [ ] All five required execution controls are demonstrated end to end.
- [ ] Documentation describes the complete host-driven workflow.
