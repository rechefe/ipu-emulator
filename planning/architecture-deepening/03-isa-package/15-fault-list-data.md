# Fault list F-1 to F-9 and arithmetic events as data

Part of [Epic E03: ISA package](00-epic.md) · ID E03-15 · Size S · Depends on: E03-07, E03-09 · Behaviour change: none (declares data; the emulator fault model is E04-19 to E04-23) · `*_WIDTH` localparams: unchanged

## Goal

The proposed fault list in the requirements (F-1 to F-9) and the arithmetic events become data in `ipu_isa` (Q52, Q59, Q53), with stable IDs, so the emulator (E04-19), the commit-record schema (E04-29), the docs and a future RTL testbench all use the same numbers. F-1's reserved encodings are derived from the ISA instead of listed by hand.

## Background

- `docs/content/requirements/faults.md:25-37` lists F-1 to F-9 (illegal state, where detected, proposed behaviour), owner Yuval Harary, status Proposed. `:39-48` lists F-1's reserved encodings per slot by hand ("One opcode of the 3-bit field" for COND, "Seven opcodes of the 4-bit field" for LR, and so on).
- Policy: each fault has an enable bit, all set after reset (IPU-FLT-5, IPU-FLT-6); when one word raises several faults, the lowest number is recorded (IPU-FLT-8).
- `faults.md:52-55`: accumulator overflow, NaN results and quantization saturation are arithmetic events, not faults; they never halt and each has a sticky flag (decision Q53).
- Decision Q59: F-8 (word stall limit) is listed as "not modelled" (the emulator has no stall model). There are two error kinds: hardware faults F-1 to F-9 and configuration errors at setup time.
- The emulator today raises untyped errors for several of these states: LR write conflict (`RuntimeError`, `ipu_emu/ipu.py:1203`), cycle limit (`emulator.py:76,122`), XMEM out of range (`xmem.py:60,73`), `LDR_CYCLIC_MULT_REG` index (`R_CYCLIC_VALID_INDICES`, `ipu.py:110`), RESHAPE mask (`ipu.py:1597-1601`).

## Implementation

1. `ipu_isa/faults.py`, authored as data:
   ```python
   class FaultId(IntEnum): F1 = 1; F2; F3; F4; F5; F6; F7; F8; F9      # value = priority (IPU-FLT-8: lowest wins)

   class Detector(StrEnum): CTRL = "ctrl"; CACHE_AGENT = "cache_agent"

   @dataclass(frozen=True, slots=True)
   class FaultInfo:
       id: FaultId
       summary: str                 # the "illegal state" text from faults.md
       detected_in: Detector
       modelled: bool               # False only for F8 (Q59)
       enabled_at_reset: bool       # True for all (IPU-FLT-6)

   class ArithmeticEvent(StrEnum): ACC_OVERFLOW; NAN_RESULT; QUANT_SATURATION
   FAULTS: Mapping[FaultId, FaultInfo]
   ```
2. `def reserved_encodings(isa) -> Mapping[Slot, ReservedEncodings]`: for each HARDWARE slot, the unused opcode values (from the instruction count and opcode width) and the reserved operand values (enum fields with fewer members than their width allows: `MultStageReg` values 2-3, `ElementsInRow`/`HorizontalStride`/`VerticalStride` value 3, unnamed `ActivationFn` codes, `INCR_MOD_POW2` `k` outside 1 to 9). This is F-1, derived.
3. Generate the F-1 reserved-encoding table in `faults.md` from `reserved_encodings` (a marked generated block), or, if the requirements page must stay hand-written, add a test that the hand-written table states the same counts. Open question below.
4. Add a docs consistency test: the fault IDs and summaries in `faults.md`'s fault-list table equal `FAULTS`.

## Tests

- [ ] `ipu-isa/test/test_faults.py`: 9 faults, priority equals the number, only F8 is not modelled, all enabled at reset.
- [ ] `reserved_encodings` for today's ISA: COND 1 opcode, LR 7, MULT 2, ACC 3 reserved (matching `faults.md:41-48`); LOAD/MULT `MultStageReg` 2 values.
- [ ] The docs consistency test passes, and fails on a planted wording change in `FAULTS`.

## Acceptance Criteria

- [ ] Fault IDs, detectors, "not modelled" status and reset enables are data in `ipu_isa`.
- [ ] F-1's reserved encodings are derived from the ISA.
- [ ] Arithmetic events are an enum next to the faults.

## Open question

- Should the F-1 table in `docs/content/requirements/faults.md` be generated (the requirements pages are hand-written today), or only checked by a test? The fault list itself is "Proposed" (OQ-1); this issue mirrors it and does not change its content.
