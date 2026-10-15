# Precise faults: detect before any slot writes

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-20 · Size M · Depends on: E04-12, E04-19 · Behaviour change: yes (a faulting word leaves no architectural effect; today earlier slots of the word have already written)

## Goal

Faults are precise (IPU-FLT-3): the faulting word has no architectural effect, and all older words have retired. When one word raises several faults, the lowest number is recorded (IPU-FLT-8). The emulator matches IPU-FLT-2: every fault is detected from values CTRL or the cache agent holds before the word is accepted.

## Background

- Faults are raised from inside handlers today. By the time the load slot raises an XMEM range error, the LR sub-slots of the same word have written their LRs (slot order `lr` → `load` → ..., ipu.py:2297-2304). An F-5 raised by ACC.RESHAPE (ipu.py:1598, 1607) comes after LOAD and MULT have written.
- `test_prototyping_equivalence.py` pins that both modes stop at the same point with the same partial state after an exception (for example `test_an_lr_conflict_still_raises`, test_prototyping_equivalence.py:317). For faults, those expectations change to "no state change". The AaQ activation overflow case (test_prototyping_equivalence.py:268-288) is not a fault and is out of scope (Q35).
- Every fault F-1 to F-7 depends only on the word's encoding and on LR/CR values. LR values seen by non-Control stages are post-LR-write (Q32), so the checker must compute the LR results first.

## Implementation

1. Split each fault check out of its handler into a pure predicate attached to the op's plan: `check(encoded, operands, view) -> FaultId | None`. Static checks (undefined opcode, reserved encodings that do not depend on registers) are evaluated once per word and cached in the `Schedule` (E04-08).
2. `Cycle.retire()` runs in two phases:
   - **check**: compute LR results into a scratch copy of the 16 LRs (LR handlers are pure functions of the snapshot), build the forwarded view from it, evaluate every op's predicate, and collect all faults;
   - **commit**: if no fault, run the slots as today; otherwise record the lowest-numbered fault in `HaltRecord` and write nothing.
3. Delete the raises inside handlers that the predicates replace. Handlers assume legal operands.
4. Keep the check phase cheap: words with no dynamic predicate skip it (a flag on the cached schedule).

## Tests

- [ ] For each fault F-1 to F-6: a word that also writes an LR, loads, and multiplies leaves registers, XMEM and PC exactly as before the word (compare with a snapshot).
- [ ] A word with F-3 and F-5 together records F-3.
- [ ] `test_prototyping_equivalence.py` updated: both modes leave identical, unchanged state on a fault.
- [ ] E02 kernel output digests unchanged.
- [ ] E04-02 benchmark: cost of the check phase recorded (target within 5 %).

## Acceptance Criteria

- [ ] No handler raises `EmulatorFault`.
- [ ] A faulting word changes no register, XMEM byte, PC or statistic other than the halt record and the fault flags.
- [ ] Lowest fault number wins when several apply.
