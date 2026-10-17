# Extract the single `Cycle` module

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-07 · Size M · Depends on: E04-06 · Behaviour change: none

## Goal

Give one VLIW word exactly one implementation: a `Cycle` module with a small interface (`evaluate_break`, `retire`, `step`, `reset`). It owns the snapshot, the slot order, the idle-word path, the LR conflict check and the observer hooks. It knows nothing about run control.

**This issue is the prerequisite of [`planning/riscv-host-integration/03-steppable-engine.md`](../../riscv-host-integration/03-steppable-engine.md).** That engine's `step()` becomes `Cycle.step(honour_break=...)` plus run-control bookkeeping. Landing the engine first would freeze today's three-method calling convention into a public API.

## Background

- The idle-word path (`inst is None`) is written twice: ipu.py:2271-2279 and ipu.py:2318-2326.
- The LR conflict check is written twice: `_dispatch_lr_slots` (ipu.py:1176-1206) and `_cycle_plan` (ipu.py:2161-2178).
- Profiler `begin`/`end` guards appear four times (ipu.py:2273-2278, 2295-2307, 2319-2324, 2334-2346).
- The break-then-resume protocol ("evaluate BREAK before side effects; take the snapshot again; run everything except BREAK") exists only as a calling pattern: `check_break` (ipu.py:2125-2131), `execute_vliw_cycle` (ipu.py:2257-2309), `execute_vliw_cycle_skip_break` (ipu.py:2311-2346), used by `emulator.py:63-169` and wrapped again by `execute.py:97-121`.
- Tests monkeypatch `Ipu.execute_vliw_cycle` (ipu-apps/test/test_debug_launch.py:175, 192, 269) and `Ipu.dispatch_instruction` (ipu-apps/test/mobilevit/fixture_packed_l4_measure.py:51-74). Both seams must keep working until their callers migrate (E04-11, E08).

## Implementation

New module `ipu_emu/cycle.py`:

```python
class CycleOutcome(Enum):
    RETIRED = 0
    BREAK = 1            # BreakResult stays as an alias until E08 migrates the debuggers

@dataclass(frozen=True)
class Schedule:                          # pure function of one instruction word
    break_op: IssuedOp | None
    lr_ops: tuple[IssuedOp, ...]
    slot_ops: tuple[IssuedOp, ...]       # in SLOT_EXECUTION_ORDER
    lr_conflict: tuple[int, ...] | None  # duplicate LR targets, checked once

def schedule_word(inst: Mapping[str, int]) -> Schedule
def decode_word(inst: Mapping[str, int]) -> tuple[IssuedOp, ...]   # public; replaces _SLOT_PLANS imports

class CycleObserver(Protocol):
    def begin(self, pc: int) -> None: ...
    def end(self) -> None: ...

class Cycle:
    def __init__(self, state: IpuState, ipu: Ipu, observer: CycleObserver | None = None): ...
    def evaluate_break(self) -> CycleOutcome   # snapshot + BREAK slot only; no side effects
    def retire(self) -> None                   # fresh snapshot, then every non-break slot in order
    def step(self, *, honour_break: bool = True) -> CycleOutcome
    def reset(self) -> None                    # drops cached schedules and the snapshot source
```

Steps:

1. Move the idle-word path, the LR conflict check, and the profiler guards into `Cycle`. `schedule_word` is the one builder; the faithful path calls it without a cache in this issue (E04-08 adds the cache).
2. `Ipu.execute_vliw_cycle`, `check_break` and `execute_vliw_cycle_skip_break` become one-line delegates to `Cycle`, which keeps the monkeypatch seams.
3. `emulator.run_until_complete` and `run_with_debug` call `Cycle.step` / `evaluate_break` + `retire`. `execute.py`'s legacy wrappers delegate to `Cycle` (they are deleted in E04-30 when the reference API replaces them).
4. Prototyping keeps its own `_execute_cycle_fast` in this issue, called from `Cycle` behind the existing flag. E04-08 turns it into an adapter.
5. `retire()` takes the snapshot again on purpose: a debugger edit made while stopped at a break is visible to the resumed word. That is today's behaviour; pin it.

## Tests

- [ ] New `test_cycle.py`: `step()` equals `evaluate_break()` followed by `retire()`; `evaluate_break()` has no side effects (registers, XMEM, PC, stats unchanged); an idle word bumps `mult_idle_cycles` once and advances the PC; with `honour_break=False` the word retires on BREAK; `reset()` empties the schedule cache; an edit between `evaluate_break()` and `retire()` is visible.
- [ ] Output-identical proof: E02 timing characterization tests and kernel output digests unchanged; `test_execute.py`, `test_debug_cli.py`, `test_debug_tui.py`, `ipu-apps/test/test_debug_launch.py` pass; `test_prototyping_equivalence.py` passes under `--define=ipu_proto=1`.
- [ ] E04-02 benchmark before/after in both modes; regression above 5 % blocks the merge.

## Acceptance Criteria

- [ ] One idle-word path, one LR conflict check, one profiler begin/end pair.
- [ ] `Ipu.execute_vliw_cycle*` and `check_break` are delegates of one line each.
- [ ] `Cycle` has no knowledge of cycle limits, stats totals or debug callbacks.
- [ ] The steppable-engine issue is updated to say `Engine.step()` wraps `Cycle.step()`.
