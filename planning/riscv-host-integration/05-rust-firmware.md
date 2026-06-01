# Rust `no_std` firmware + generated driver for the IPU host

Part of the [RISC-V host integration epic](00-epic.md). Depends on #1, #4.

## Goal

Write the on-device host code in **Rust** (`no_std`): a generated register-access
crate, a small HAL, and an example firmware that drives the IPU entirely through
the MMIO control block, then build it for a bare-metal RISC-V target and load it
into the Unicorn core (#4).

## Crates

- **`ipu-pac`** — generated from `ipu_ctrl.rdl` (#1): typed, volatile register
  access (fields + enums). Build output, not hand-edited.
- **`ipu-rt`** — thin hand-written HAL over `ipu-pac` exposing:
  - `configure(dtype, dstructure, crs)`
  - `load_program(words)` (streams via the IMEM port)
  - `start()`, `wait_until_halted()`
  - `read_pc()`, `write_pc(addr)`
  - `reset()`
- **`firmware`** — example `no_std` binary whose `main` mirrors an app flow:
  configure → load program → start → wait → done.

## Implementation

- Add `rules_rust` to `MODULE.bazel`; register a bare-metal RISC-V toolchain
  (e.g. `riscv32imac-unknown-none-elf`).
- Provide a linker script / memory layout matching the host RAM base from #4;
  emit a **flat binary** artifact for loading into Unicorn.
- Use `core`-only Rust (no `std`), a minimal `_start`/reset handler, and a panic
  handler.
- Wire firmware build outputs so the integration harness (#6) can consume the
  flat binary as a Bazel data dependency.

## Docs

- Document how to build the firmware and the driver API in the spec; add a short
  "writing host firmware" snippet.

## Tests

- [ ] `ipu-pac` + `ipu-rt` + `firmware` compile for the bare-metal target under
      Bazel.
- [ ] The produced flat binary loads and runs in the Unicorn core (smoke test
      from #4 harness): firmware writes reach the control model.

## Acceptance Criteria

- [ ] Rust (not C) firmware + driver, generated register access from the single
      SystemRDL source.
- [ ] Bare-metal RISC-V build wired into Bazel producing a loadable flat binary.
- [ ] Firmware drives configure/load/start/wait/reset over MMIO.
