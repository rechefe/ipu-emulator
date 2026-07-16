# APB slave register block (host config + bank swap)

Part of the [CTRL-stage RTL epic](00-epic.md). Depends on #4. Coordinates
with, but does not block on, the
[RISC-V host integration epic](../riscv-host-integration/00-epic.md).

## Goal

Implement `ctrl_apb_slave.sv`: the APB slave described in
`stage-control.md` §8 — inactive-bank IMEM/CR writes, the IMEM and CR
bank-swap registers, and read-only status — without inventing a second,
disconnected register-map source of truth alongside the one the host
integration epic is already building.

## Why this coordinates with the other epic instead of duplicating it

The [RISC-V host integration epic](../riscv-host-integration/00-epic.md)'s
[Issue 1](../riscv-host-integration/01-systemrdl-regfile.md) is already
scoped to author `ipu_ctrl.rdl` in SystemRDL as *the* single source of truth
for this exact register surface, generating Python, Rust, and docs
consumers from it. This RTL should be a **fourth** consumer of the same
`.rdl`, not a hand-written register map that happens to describe the same
bits. Concretely:

- If host-integration Issue 1 has landed by the time this issue starts,
  generate the APB decode/field logic from `ipu_ctrl.rdl` (PeakRDL has a
  SystemVerilog exporter; evaluate it the same way Issue 1 evaluated the
  Python/Rust exporters).
- If it hasn't landed yet, this issue is **not blocked** — write the APB
  slave directly against the register map documented in
  `riscv-host-integration.md` §4 (same table both epics already point at),
  but flag the hand-written fields for a follow-up swap to the generated
  source once `ipu_ctrl.rdl` exists, so the two never permanently diverge.
- Either way, do not let this issue's register offsets/field layouts drift
  from `riscv-host-integration.md` §4 — that table and `stage-control.md`
  §8 are describing the same hardware from two documents; if this issue
  finds a discrepancy between them, fix the docs (likely a #1-style
  reconciliation, but scoped to the APB surface specifically) rather than
  picking one silently.

## Scope specific to CTRL (not the full host-integration register set)

`stage-control.md` §8 lists what CTRL's APB slave itself needs to expose —
this is narrower than the full host-control block in
`riscv-host-integration.md` §4 (which also covers `DTYPE`, `ELU_ALPHA`, and
execution-flow `CTRL`/`STATUS`/`PC`/`CYCLES` registers that live in other
stages or the steppable-engine layer, not CTRL):

- Inactive IMEM bank — every decoded VLIW-word entry writable while
  inactive.
- Inactive CR bank — `CR0`/`CR1` addresses return `apb_pslverr` on write
  (see #4's note on reconciling this with the emulator's current
  silent-drop behavior).
- IMEM bank-swap register — new active bank takes effect on the next CTRL
  fetch cycle.
- CR bank-swap register — analogous.
- Optional status (active-bank IDs, last-error flags).
- `apb_pready`/`apb_pslverr` handshake, gated only by the slave's own
  write-port throughput (§8: host APB traffic never contends with the
  IPU's own active-bank reads, by construction, since the host only ever
  touches the inactive bank).

## No emulator oracle yet (per #1, gap #3)

Unlike #3/#4/#5, this block currently has **no Python ground truth** to
parity-test against — the emulator's `inst_mem`/CR file are flat and
unbanked with no APB model. Do not simulate a fake oracle; instead:

- Test this block against the **spec directly** (bank-swap timing,
  read-only enforcement, `pready`/`pslverr` handshake correctness) with
  cocotb testbenches that don't reference `ipu_emu` at all.
- Once the host-integration epic's emulator MMIO model
  ([its Issue 2](../riscv-host-integration/02-emulator-mmio-model.md)) lands,
  revisit this issue (or file a fast follow) to add real parity tests
  against it — note that as a known gap here rather than silently skipping
  it.

## Tests

- [ ] Inactive-bank IMEM/CR writes land correctly; active-bank reads are
      unaffected mid-write.
- [ ] `CR0`/`CR1` writes are rejected per #4's decided behavior.
- [ ] Bank-swap register write flips the active bank exactly on the next
      fetch cycle, not combinationally mid-cycle.
- [ ] `apb_pready` never stalls for IPU-internal reasons (only for the APB
      slave's own write-port throughput).
- [ ] Out-of-range `apb_paddr` returns `apb_pslverr`.

## Acceptance Criteria

- [ ] APB slave matches `stage-control.md` §8 and is field-compatible with
      `riscv-host-integration.md` §4's CTRL-owned subset (IMEM/CR
      banks + swap regs), sharing a register-map source once
      host-integration Issue 1 lands.
- [ ] Bank-swap and read-only enforcement tested directly against the spec.
- [ ] The "no emulator oracle yet" gap is recorded, not silently absent.
