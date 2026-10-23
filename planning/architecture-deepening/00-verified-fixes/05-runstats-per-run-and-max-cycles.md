# Run statistics: reset per run, keep `total_cycles` on error, one `max_cycles` default

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-05 · Size S · Depends on: none · Behaviour change: yes (stats describe one run instead of accumulating; `run_until_complete` / `run_with_debug` default limit rises from 100 000 to 1 000 000 cycles)

## Goal

`state.stats` always describes exactly one run, its `total_cycles` is set even when the run raises, and every run entry point uses one named `max_cycles` default.

## Background

Verified at 694a8a8:

- `IpuState.__init__` creates `self.stats = RunStats()` once (ipu_state.py:81). No run entry point resets it.
- `run_until_complete` (emulator.py:63-86) overwrites `state.stats.total_cycles = cycles` at the end, while every other counter (`mult_active_cycles`, `acc_active_cycles`, `xmem_reads`, `mult_lane_ops`, `alias_hits`, ...) keeps accumulating. Running a second program on the same state therefore reports the counters of both runs over the cycles of the last one. `mult_utilization` (stats.py:54-58) can exceed 1.0.
- That happens today. The three `depthwise_conv_stride2_*` harnesses run stage 2 on stage 1's state (`depthwise_conv_stride2_16/app.py:186-192`, `_128/app.py:206`, `_narrow/app.py:237`) and return `cycles1 + cycles2`, while `state.stats.total_cycles` holds only stage 2's cycles.
- `run_until_complete` sets `total_cycles` only after the loop. When the cycle limit or an `EmulatorError` raises, `total_cycles` stays at its old value. `run_with_debug` (emulator.py:89-164) does set it in a `finally` block (emulator.py:161-163). The two runners disagree.
- `max_cycles` defaults differ by entry point: 100 000 in `run_until_complete` (emulator.py:63) and `run_with_debug` (emulator.py:92); 1 000 000 in `run_test` (emulator.py:277), `IpuApp.run` (kernel_registry/base.py:71) and `KernelCase.max_cycles` (kernel_registry/cases.py:32). The same program can pass through one API and hit the limit through another. Per-kernel overrides (for example `conv_universal_wide384/app.py:329`, the `cases.py` values) are deliberate and stay.

## Decisions applied

- L3, L4 (accepted via Q29): `max_cycles` default 1 000 000; stats reset per run.
- Q52, Q59: the cycle limit becomes fault F-9 in E04. This issue does not change the exception type (`RuntimeError`). E04-21 converts it and should shrink to that conversion once this lands.

## Implementation

1. `stats.py`: add `RunStats.reset()`, which zeroes every counter and clears `alias_hits` but keeps the `alias_profile` attachment (an `AliasProfile` is attached at state construction and must survive). Add `RunStats.__add__` that sums counters and `alias_hits` and keeps the left operand's `alias_profile`.
2. `emulator.py`:
   - `DEFAULT_MAX_CYCLES = 1_000_000` at module level, exported from `ipu_emu`.
   - `run_until_complete` and `run_with_debug` call `state.stats.reset()` before the first cycle and set `state.stats.total_cycles = cycles` in a `finally` block.
   - `run_until_complete`, `run_with_debug` and `run_test` default `max_cycles` to `DEFAULT_MAX_CYCLES`.
3. `kernel_registry/base.py:71` and `kernel_registry/cases.py:32`: default to `ipu_emu.emulator.DEFAULT_MAX_CYCLES`.
4. The three `depthwise_conv_stride2_*` harnesses: keep a copy of the stage-1 stats, and after stage 2 set `state.stats = stage1_stats + state.stats`, so the stats cover the whole kernel and `total_cycles == cycles1 + cycles2`.
5. Update the docstrings of both runners: "stats describe this run only".

## Tests

- [ ] `test_stats.py`: run a small program twice on one state; the second run's stats equal the first run's (not double), and `mult_utilization <= 1.0`.
- [ ] `test_stats.py`: a program that exceeds `max_cycles=10` raises and leaves `state.stats.total_cycles == 10` in `run_until_complete` (it was stale before) and in `run_with_debug`.
- [ ] `test_stats.py`: `reset()` keeps an attached `alias_profile`; `__add__` sums every counter field (iterate `dataclasses.fields` so a new counter cannot be missed).
- [ ] Each `depthwise_conv_stride2_*` test asserts `state.stats.total_cycles == cycles`.
- [ ] `bazel test //...` is green. Kernel `results.md` files for the stride2 kernels may change; regenerate them in the same PR if their numbers change.

## Acceptance Criteria

- [ ] No run entry point can report stats from more than one run unless a caller adds them explicitly.
- [ ] `total_cycles` is correct after a normal halt, a cycle-limit error and an emulator error, in both runners.
- [ ] `grep -rn "max_cycles: int = [0-9]" src/tools/*/src` shows only `DEFAULT_MAX_CYCLES` uses outside per-kernel overrides.
- [ ] The PR is labelled as a behaviour change and states both changes (stats scope, default limit).

## Risks

- A test that relied on accumulated stats across two runs. At HEAD only the stride2 harnesses chain runs and read stats; check `state=state` callers (27 hits under `ipu-apps`) for later `stats` reads before merging.
- The run-control helper in E04-21 and the steppable engine ([`03-steppable-engine.md`](../../riscv-host-integration/03-steppable-engine.md)) absorb cycle counting later. Keep the reset in one place (a small `_begin_run(state)` helper) so they can take it over.
