# Refactor the run loop into a host-drivable, steppable execution engine

Part of the [RISC-V host integration epic](00-epic.md). Can proceed in parallel with #2.

## Goal

Decouple IPU execution from the monolithic `run_until_complete()` loop so the
host-control model (#2) can **start, halt, single-step, and reset** the IPU and
**read/write the PC**, while the existing direct-Python path keeps working
unchanged.

## Background

Today `emulator.run_until_complete()` owns the loop and the halt condition
(`PC >= INST_MEM_SIZE`), and `run_with_debug()` layers breakpoints on top. There
is no externally drivable "advance one cycle and tell me the status" primitive.

## Implementation

- Introduce an engine abstraction (e.g. `ipu_emu/engine.py`) over `IpuState`
  with:
  - `step() -> RunStatus` — execute exactly one VLIW cycle; report
    `RUNNING` / `HALTED` / `BREAK`.
  - `run(max_cycles) -> RunStatus` — loop `step()` until halt/break/limit
    (this is what `run_until_complete`/`run_with_debug` become, preserving
    current behavior).
  - `halt()` — request a cooperative stop at the next instruction boundary.
  - `reset(preserve_imem=True)` — soft reset (see #2 reset semantics).
  - `get_pc()` / `set_pc(value)` — PC read/write (set honored only when halted).
  - status surface: `running`, `halted`, `break_hit`, `cycles`, `error`.
- Re-express `run_until_complete()` / `run_with_debug()` in terms of the engine
  so behavior and tests are unchanged.
- Fold `RunStats` cycle counting and `is_halted` into the engine status.

## Docs

- Note the engine API in the spec and (briefly) in `debugging.md` since
  single-step mirrors the debug-CLI `step`.

## Tests

- [ ] `run()` reproduces `run_until_complete` results across existing programs.
- [ ] `step()` advances exactly one cycle and updates PC/cycles/status.
- [ ] `halt()` stops at an instruction boundary; `run()` can resume.
- [ ] `reset(preserve_imem=True)` clears state but keeps `inst_mem`.
- [ ] `set_pc` while running is rejected; while halted it takes effect.
- [ ] No regressions: `bazel test //src/tools/ipu-emu-py/...` and app tests pass.

## Acceptance Criteria

- [ ] A steppable, host-drivable engine exists with start/halt/step/reset/PC R-W.
- [ ] Existing run/debug APIs are thin wrappers over it (no behavior change).
- [ ] Full test suite passes.
