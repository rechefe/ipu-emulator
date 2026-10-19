# Cycle limit as F-9 (default 1M) and stats reset per run

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-21 · Size S · Depends on: E04-19 · Behaviour change: yes (default limit 100 000 → 1 000 000 cycles; exceeding it is fault F-9 instead of `RuntimeError`; `RunStats` reset at the start of each run)

## Goal

- The cycle limit is the hardware fault F-9 with a programmable limit (Q52, Q59), default 1 000 000 cycles (L3). A limit of 0 disables the check (faults.md).
- Each run starts from fresh statistics (L4). `total_cycles` is written the same way by every runner.
- Cycle counting and stats bookkeeping live in one small run-control helper that the [steppable engine](../../riscv-host-integration/03-steppable-engine.md) absorbs.

## Background

- `run_until_complete(state, max_cycles=100_000)` raises `RuntimeError` past the limit (emulator.py:63-86) and writes `total_cycles` only after a clean finish (emulator.py:85).
- `run_with_debug(..., max_cycles=100_000)` raises the same `RuntimeError` (emulator.py:122) and writes `total_cycles` in a `finally` (emulator.py:161-163).
- `RunStats` is never reset: running twice on one state accumulates counters.
- Apps pass their own `max_cycles` in some places; grep shows the call sites.

## Implementation

1. `IpuState.cycle_limit: int = 1_000_000` (0 = no limit). `DEFAULT_CYCLE_LIMIT` lives next to it.
2. New `ipu_emu/run_control.py`:
   ```python
   class RunControl:
       def __init__(self, state: IpuState, cycle: Cycle): ...
       def begin_run(self, *, reset_stats: bool = True) -> None
       def count_cycle(self) -> None          # raises F-9 via the fault path when over the limit
       def end_run(self) -> None              # writes total_cycles; called from finally
   ```
   `run_until_complete` and `run_with_debug` use it. Their `max_cycles` argument sets `state.cycle_limit` for the run.
3. F-9 goes through the E04-19 fault path (halt record with F-9 and the PC).
4. Note in the steppable-engine issue that `Engine` owns `RunControl` once it lands.

## Tests

- [ ] A tight loop with `cycle_limit=10` faults with F-9 at the expected PC after 10 cycles.
- [ ] `cycle_limit=0` runs past 1M cycles (use a short loop and a counter observer that stops the test, not real time).
- [ ] Running a program twice on one state gives identical `RunStats`; `reset_stats=False` accumulates.
- [ ] `total_cycles` equals the cycle count in both runners, including when a run ends in a fault.
- [ ] Update tests that expect `RuntimeError` for the limit.

## Acceptance Criteria

- [ ] One default limit constant used by every runner and the apps' harness.
- [ ] No `RuntimeError` for the cycle limit remains.
- [ ] `RunStats` totals are written in exactly one place.
