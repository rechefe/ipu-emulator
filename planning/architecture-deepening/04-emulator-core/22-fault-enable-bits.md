# Per-fault enable bits and sticky fault flags

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-22 · Size S · Depends on: E04-19 · Behaviour change: yes (faults can be disabled; a disabled fault sets a sticky flag instead of halting)

## Goal

Each fault has an enable bit (IPU-FLT-5, Q52). Set: the fault halts and records code and PC. Clear: the fault only sets a sticky flag. All enable bits are set after reset (IPU-FLT-6).

## Background

- `docs/content/requirements/faults.md` defines the policy rows IPU-FLT-4 to IPU-FLT-6.
- What a word does when its fault's enable bit is clear is open: [OQ-2](../../../docs/content/requirements/open-questions.md). The only proposal on record: for F-1, the slot with the reserved encoding executes as NOP.
- The host register map reserves fault control registers (`docs/content/requirements/host-registers.md`); the MMIO model in `riscv-host-integration/02-emulator-mmio-model.md` will map them onto `IpuState`.

## Implementation

1. `IpuState.faults: FaultControl`:
   ```python
   @dataclass
   class FaultControl:
       enable: dict[FaultId, bool]        # all True after reset
       sticky: set[FaultId]               # set when a fault fires, enabled or not
       def reset(self) -> None
   ```
2. In `Cycle`'s check phase (E04-20): an enabled fault halts as today; a disabled fault adds to `sticky` and the word continues.
3. Behaviour of the word with a disabled fault, per OQ-2:
   - F-1: the slot executes as NOP (the recorded proposal).
   - F-9: the run continues past the limit.
   - F-2 to F-6: **Open question (OQ-2)**. Until answered, disabling these raises `ConfigurationError` at setup ("behaviour not defined"), so no undefined behaviour is modelled by accident.
4. Expose enable/sticky through the reference API (E04-30) and in the commit record (E04-29).

## Tests

- [ ] After reset every fault is enabled and `sticky` is empty.
- [ ] F-1 disabled: the slot runs as NOP, the other slots retire, `sticky == {F_1}`.
- [ ] F-9 disabled: the run passes the limit and sets the flag.
- [ ] Disabling F-2..F-6 raises `ConfigurationError` until OQ-2 is answered.
- [ ] Enabled faults behave exactly as in E04-20.

## Acceptance Criteria

- [ ] Fault enable and sticky state are part of `IpuState` and survive `step()`.
- [ ] OQ-2 cases that are not decided are refused at setup, not guessed.
