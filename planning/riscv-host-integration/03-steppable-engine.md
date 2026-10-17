# Refactor the run loop into a host-drivable, steppable execution engine

Part of the [RISC-V host integration epic](00-epic.md). Can proceed in parallel with #2.
Depends on [E04-07](../architecture-deepening/04-emulator-core/07-cycle-module.md) (the `Cycle` module, landed).

## Goal

Decouple IPU execution from the monolithic `run_until_complete()` loop so the
host-control model (#2) can **start, halt, single-step, and reset** the IPU and
**read/write the PC**, while the existing direct-Python path keeps working
unchanged.

## Background

Today `emulator.run_until_complete()` owns the loop and the halt condition
(`PC >= INST_MEM_SIZE`), and `run_with_debug()` layers breakpoints on top. There
is no externally drivable "advance one cycle and tell me the status" primitive.

What one VLIW word does is already one module: `ipu_emu.cycle.Cycle`
(E04-07). It owns BREAK evaluation, the register snapshot, the slot order, the
LR conflict check, the idle-word path and the observer hooks, and knows nothing
about run control. Its interface is `step(*, honour_break=True) -> CycleOutcome`
(`RETIRED` or `BREAK`), `evaluate_break()` and `retire()` (the break-then-resume
pair a debugger needs) and `reset()`. The engine builds on it and must not
call `Ipu.execute_vliw_cycle` / `check_break` / `execute_vliw_cycle_skip_break`:
those are transitional delegates of `Cycle`.

## Implementation

- Introduce an engine abstraction (e.g. `ipu_emu/engine.py`) over `IpuState`
  with:
  - `step() -> RunStatus` — execute exactly one VLIW cycle; report
    `RUNNING` / `HALTED` / `BREAK`. `Engine.step()` wraps `Cycle.step()`:
    it is `Cycle.step(honour_break=...)` plus run-control bookkeeping (cycle
    count, status flags, halt check).
  - `run(max_cycles) -> RunStatus` — loop `step()` until halt/break/limit
    (this is what `run_until_complete`/`run_with_debug` become, preserving
    current behavior).
  - `halt()` — request a cooperative stop at the next instruction boundary.
  - `reset(preserve_imem=True)` — soft reset (see #2 reset semantics). Calls
    `Cycle.reset()`, which drops cached schedules and the snapshot source.
  - `get_pc()` / `set_pc(value)` — PC read/write (set honored only when halted).
  - status surface: `running`, `halted`, `break_hit`, `cycles`, `error`.
- Re-express `run_until_complete()` / `run_with_debug()` in terms of the engine
  so behavior and tests are unchanged.
- Fold `RunStats` cycle counting and `is_halted` into the engine status. The
  engine owns `ipu_emu.run_clock.RunClock` (E04-21): the run's cycle count,
  the F-9 cycle limit and the one write of `total_cycles`.

- Consumer: the stable reference-model API, `ipu_emu.reference` (E04-30),
  runs on `RunClock` and `Cycle` today. When the engine lands, its
  `ReferenceModel.step`/`run`/`status` delegate to `Engine` without changing
  their signatures (`API_VERSION` stays `1.0.0`; the contract test pins them).

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
