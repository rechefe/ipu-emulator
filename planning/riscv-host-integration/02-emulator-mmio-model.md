# Emulator MMIO control-register model mapped onto `IpuState`

Part of the [RISC-V host integration epic](00-epic.md). Depends on #1.

## Goal

Implement a Python **control-register model** that interprets reads/writes to the
generated register block (#1) and applies them to an `IpuState`, covering both
configuration and execution control. This is the bridge target that the Unicorn
MMIO callbacks (#4) will call.

## Implementation

Add a module (e.g. `ipu_emu/host_ctrl.py`) exposing an `IpuHostController` that
wraps an `IpuState` (and the steppable engine from #3) and offers:

- `read(offset) -> int` and `write(offset, value)` decoded against the generated
  register map (no magic numbers — import offsets/masks from the #1 artifacts).
- Side effects per register:
  - **Config** — `DTYPE` → `state.dtype`; `DSTRUCTURE` → `state.set_cr_dstructure`;
    `ELU_ALPHA` → `state.set_activation_alphas`; `CR[n]` → `regfile.set_cr`
    (rejecting `CR0`/`CR1`, aliasing `CR15` to `DSTRUCTURE`, raising `ERROR`).
  - **Execution** — `CTRL.START/HALT/STEP/RESET/CONTINUE` drive the engine (#3);
    `PC` read/write maps to `state.program_counter` (writes honored only while
    halted); `STATUS`/`CYCLES` reflect engine state.
  - **IMEM** — `IMEM_ADDR`/`IMEM_WDATA`/`IMEM_CTRL` accept raw encoded 32-bit
    sub-words, assemble full VLIW words, decode via `decode_instruction_word`, and
    write into `inst_mem`. Read-back via `IMEM_RDATA`.

### Reset semantics

`CTRL.RESET` must reset **everything except `inst_mem`**: re-init the register
file (`CR0=0`, `CR1=1`, default dstructure), clear `R_ACC`/`R`/AAQ/post-AAQ and
XMEM, set `PC=0`, clear stats/cycles, and preserve the loaded program + length.

## Docs

- Document the register semantics and reset behavior in
  [`specs/riscv-host-integration.md`](../../docs/content/specs/riscv-host-integration.md)
  (already drafted) — keep it in sync with the final field set.

## Tests

- [ ] Writing `DTYPE`/`DSTRUCTURE`/`ELU_ALPHA`/`CR[n]` produces the same
      `IpuState` as the equivalent direct-Python setup.
- [ ] Writing to `CR0`/`CR1` sets `STATUS.ERROR` and leaves them unchanged.
- [ ] Streaming an assembled program through the IMEM port yields `inst_mem`
      identical to `load_program_from_binary`.
- [ ] `CTRL.RESET` clears state but preserves `inst_mem`; a re-run reproduces
      results without re-streaming the program.
- [ ] `PC` read/write round-trips; writes while `RUNNING` are rejected.

## Acceptance Criteria

- [ ] All config + execution controls reachable purely through `read`/`write`.
- [ ] Offsets/masks imported from #1 artifacts (no duplication).
- [ ] Reset preserves only instruction memory.
- [ ] Unit tests pass under `bazel test`.
