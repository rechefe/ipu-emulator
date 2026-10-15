# Per-word execution plans cached by PC; prototyping as an adapter

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-08 · Size M · Depends on: E04-07 · Behaviour change: none

## Goal

- The faithful path stops re-decoding each word every cycle. `Cycle` caches the `Schedule` for each PC and drops it when that IMEM entry is written (Q72: faster without changing results).
- Prototyping mode becomes an **adapter** on the single cycle (D7): it only changes how the schedule is built (idle NOP slots skipped). Its results are identical to the faithful path, **including partial state when a fault interrupts a cycle** (decision on plan-01 Q5, option a).

## Background

- Only prototyping caches today: `_cycle_plan` (ipu.py:2139-2194) stores `(dict(inst), plan)` per PC and compares a field copy of the word on every cycle (ipu.py:2148-2152) to notice debugger edits.
- The faithful path rebuilds operand dicts from the word on every dispatch (`dispatch_instruction`, ipu.py:1905; `_dispatch_lr_slots`, ipu.py:1176).
- `_execute_cycle_fast` (ipu.py:2228-2255) resolves and executes LR sub-slots one at a time (`_run_plan` per sub-slot, ipu.py:2245-2246). The faithful path resolves all LR sub-slots before running any (ipu.py:1186-1206). They agree only because every LR-slot read is a snapshot read. If a sub-slot read raised, prototyping would already have written the earlier sub-slot. This is unreachable today but undeclared and untested.
- `state.inst_mem` is a plain list (ipu_state.py:84). Writers: `load_program` (emulator.py:45-54), debugger edits in `debug_cli`/`debug_tui`, and tests.

## Implementation

1. Make IMEM an object with a write hook:
   ```python
   class InstMem(Sequence[dict[str, int] | None]):
       def __setitem__(self, pc: int, word: dict[str, int] | None) -> None   # bumps version[pc]
       def version(self, pc: int) -> int
       def add_write_observer(self, fn: Callable[[int], None]) -> Callable[[], None]
   ```
   Words stored in IMEM are frozen (`MappingProxyType`), so in-place edits of a fetched dict are impossible; debuggers write a new word through `__setitem__`. Grep the debuggers for in-place edits and convert them in this PR.
2. `Cycle` keeps `dict[int, Schedule]` and drops an entry from the IMEM write observer. `Cycle.reset()` clears it. The cache is used in both modes.
3. `schedule_word(inst, *, skip_idle_nops: bool)`: prototyping passes `True`. Skipped NOP slots that bump a counter (`mult_idle_cycles`) keep their place as counter-only entries, as `_cycle_plan` does today (ipu.py:2180-2194).
4. LR sub-slots resolve all operands before any LR write in both modes. Delete `_execute_cycle_fast` and `_cycle_plan`; prototyping runs `Cycle.retire()` with its schedule.
5. The profiler no longer forces the faithful path (ipu.py:2283-2287); it is a `CycleObserver` in both modes. A profiled prototyping run reports the slots the schedule issued; skipped NOP slots are reported as idle, matching what the faithful run records. Verify with the equivalence test.

## Tests

- [ ] New: writing IMEM at PC p invalidates only p's schedule; `Cycle.reset()` clears all.
- [ ] New `test_lr_subslot_exception_atomicity`: inject a resolver error in LR sub-slot 2 and show neither mode wrote LR sub-slot 1's target (both modes, compared in lockstep).
- [ ] New: a profiled prototyping run and a profiled faithful run give identical alias profiles and stats.
- [ ] Output-identical proof: `test_prototyping_equivalence.py` (per-cycle lockstep of registers, XMEM and stats) under `--define=ipu_proto=1`, including `test_an_lr_conflict_still_raises` (test_prototyping_equivalence.py:317); E02 kernel output digests in both modes; debugger tests that edit IMEM at a break.
- [ ] E04-02 benchmark before/after for both modes.

## Acceptance Criteria

- [ ] One schedule builder, one cache, used by both modes.
- [ ] `_execute_cycle_fast`, `_cycle_plan`, `_run_plan` and `_SLOTS_WITH_ACTIVE_NOP`'s duplicate logic are gone.
- [ ] Prototyping differs from faithful only in `skip_idle_nops=True` (and, after E04-10, the resolver adapter).
- [ ] Faithful-path cycles/s improves or stays within 5 %; number recorded.

## Risks

- A debugger that mutates a fetched word dict in place would bypass invalidation. Frozen words turn that into an immediate `TypeError` instead of a stale cache.
