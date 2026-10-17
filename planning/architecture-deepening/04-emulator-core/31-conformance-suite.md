# Conformance suite stored as data

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-31 · Size M · Depends on: E04-30 · Behaviour change: none

## Goal

Store the expected behaviour of the ISA as data files (Q60): programs plus expected traces. Any model, the Python emulator, the RTL under cocotb or a future ETISS model (Q56), can run the same cases and compare traces record by record. The Python runner is one consumer of the data.

## Background

- E02 adds timing characterization tests and kernel output digests, written as Python tests.
- E04-12 generates timing probe programs from the declared contract.
- E04-29 defines the trace format; E04-30 provides `ReferenceModel.run(trace=...)`.

## Implementation

1. A directory of cases, one folder per case:
   ```
   conformance/
     <case>/
       program.asm          # source, for reading
       program.bin          # assembled words (what models load)
       setup.json           # format, CR values, XMEM regions (hex), fault enables
       expected.trace.jsonl # E04-29 trace
   ```
2. Initial coverage:
   - every HARDWARE-tagged instruction (Q17) in INT8 and E4, plus FP32 where the instruction is format-dependent;
   - the timing probes from E04-12 (one per declared implicit read and operand view);
   - each modelled fault (F-1 to F-6, F-9) and HALT;
   - arithmetic events (E04-23).
3. Generator: `bazel run //...:regen_conformance` assembles the programs, runs `ReferenceModel` and writes `expected.trace.jsonl`. Regeneration diffs are reviewed like goldens.
4. Runner: `bazel test //...:test_conformance` loads each case into `ReferenceModel`, runs it, and compares traces record by record, reporting the first differing field.
5. Usage note for RTL testbenches: how to load `setup.json` and compare commit records (content for the E09 page).

## Tests

- [ ] `test_conformance` passes on the generated data.
- [ ] A deliberately corrupted expected trace fails with a message naming the cycle and field.
- [ ] Every HARDWARE-tagged instruction appears in at least one case (coverage check against `ipu_isa`).
- [ ] The suite runs in under 30 s.

## Acceptance Criteria

- [ ] The cases contain no Python; a non-Python consumer needs only the schema and the files.
- [ ] Each later behaviour change (E04-25 to E04-28, if landing after this issue) regenerates and reviews the affected traces.

## Open question

- Where does the conformance data live: inside the `ipu_isa` package directory (it is ISA contract data) or a top-level `conformance/` directory? It must not depend on `ipu_emu` either way.
