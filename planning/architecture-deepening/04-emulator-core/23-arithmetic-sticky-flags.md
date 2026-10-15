# Accumulator overflow and NaN: IEEE result plus sticky flag

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-23 · Size S · Depends on: E04-16, E04-19 · Behaviour change: yes (an FP accumulator overflow stores ±inf and sets a flag instead of raising `OverflowError`; integer wrap and NaN results set flags)

## Goal

Arithmetic events are not faults and never halt (Q53, faults.md "Arithmetic events"). An accumulator overflow or a NaN result produces the IEEE result and sets a per-event sticky flag.

## Background

- FP accumulator stores go through `struct.pack_into("<f", ...)`, which raises `OverflowError: float too large to pack with f format` for a finite value above the float32 maximum. Two tests pin that for ACC.ADD: `TestNarrowStoreOverflow` (test_execute.py:3116-3144) and `TestWideVectorStoreOverflow` (test_wide_vector_debug.py:622-653). The activation overflow test at test_prototyping_equivalence.py:281 is AaQ and stays as it is (Q35).
- After E04-13/E04-16 that behaviour sits in one place, `LaneCodec.store_acc`.
- Integer accumulation wraps to int32 silently (`ipu_add`, ipu_math.py:197-207).
- faults.md reserves "one sticky flag for each event": accumulator overflow, NaN result, quantization saturation. Quantization saturation is AaQ and out of scope (Q35).

## Implementation

1. `IpuState.arith_flags: set[ArithEvent]` with `ArithEvent.ACC_OVERFLOW` and `ArithEvent.ACC_NAN` (names from `ipu_isa` if E03 declares them, otherwise defined here and moved). Cleared at reset, never by a run.
2. `LaneCodec.store_acc` returns the events it saw:
   - FP formats: store the IEEE float32 result (±inf on overflow, NaN propagated) using numpy float32 casting; report `ACC_OVERFLOW` if any lane overflowed from finite inputs, `ACC_NAN` if any lane is NaN.
   - INT formats: keep the wrap; report `ACC_OVERFLOW` when any lane wrapped.
3. ACC and AGG handlers add the events to `arith_flags`. The commit record (E04-29) lists events per cycle.
4. Do not set a quantization-saturation flag; record in E09 open questions that the AaQ placeholder does not model it.
5. Update the two pinned ACC tests to assert the inf result and the flag.

## Open question

- Do MULT products that overflow float32 (possible for E7 operands) count as an accumulator event? faults.md places events "in ACC and AAQ". Proposal: flag only in ACC/AGG; MULT stores the IEEE result without a flag. The user decides.

## Tests

- [ ] FP32 ACC.ADD overflow: lane holds +inf, other lanes correct, `ACC_OVERFLOW` set, no exception.
- [ ] INT8 accumulate wrap at 2^31: wrapped value, `ACC_OVERFLOW` set.
- [ ] A NaN input propagates and sets `ACC_NAN`.
- [ ] Flags are sticky across cycles and cleared only by reset.
- [ ] E02 kernel output digests unchanged (no kernel overflows today).

## Acceptance Criteria

- [ ] No arithmetic event raises or halts.
- [ ] The two pinned ACC overflow tests assert the new behaviour; the AaQ overflow test is unchanged.
