# Follow-up: RISC-V host + IPU in one ETISS virtual platform

Part of the [ETISS integration epic](00-epic.md). Depends on #6 and on the
SystemRDL register block from
[`planning/riscv-host-integration/01-systemrdl-regfile.md`](../riscv-host-integration/01-systemrdl-regfile.md).

## Goal

Replace the Unicorn-based host from the
[RISC-V Host Integration](../riscv-host-integration/00-epic.md) plan with
ETISS's own `RV32IMACFD` core, so host firmware and the IPU run in **one**
ETISS virtual platform sharing a memory system.

## Implementation

- Build ETISS with `ArchImpl/RV32IMACFD` enabled (currently optional in #2).
- Runner v2 (`ipu_vp`): two `CPUCore`s (`core0` = `RV32IMACFD`, `core1` = `IPU`)
  over one `SimpleMemSystem`; host RAM, IPU IMEM, IPU XMEM, and the control
  block in one address map.
- Control block as an ETISS `MemMappedPeriph` (`etiss::plugin::SelectiveSysWrapper`)
  whose `read32/write32` implement the SystemRDL semantics (`CTRL.START/HALT/
  STEP/RESET`, `PC`, `STATUS`, `CR[n]`, `DTYPE`, …) by driving the IPU core:
  `START` runs the IPU core to `CPUFINISHED` inside the write callback
  (deterministic, same as the Unicorn design's §10), `STEP` executes one
  instruction, `RESET` calls `resetCPU` preserving IMEM.
- The IMEM window becomes a normal memory segment the host writes and the IPU
  fetches from (halt-gated as the spec requires).
- Rust `no_std` firmware and the generated driver from issues 5/6 of the host
  plan are reused unchanged; only the load path differs (ELF via
  `SimpleMemSystem::load_elf`).

## Docs

- Update `docs/content/specs/riscv-host-integration.md` §7 to describe the
  ETISS core instead of Unicorn, and reference this issue.

## Tests

- [ ] Firmware stub configures CRs, loads a program, starts the IPU, polls
      `STATUS`, and the end-to-end result equals the direct `run_test` result.
- [ ] The fully-connected kernel end-to-end through firmware.

## Acceptance Criteria

- [ ] Single ETISS VP runs host firmware and the IPU with results identical to
      the Python path.
- [ ] Unicorn is no longer a dependency of the host plan.
