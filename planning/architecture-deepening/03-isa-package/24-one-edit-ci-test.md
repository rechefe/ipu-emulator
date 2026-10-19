# "One edit" CI test with a dummy instruction

Part of [Epic E03: ISA package](00-epic.md) · ID E03-24 · Size M · Depends on: E03-11, E03-14, E03-15, E03-17, E03-18 · Behaviour change: none (test only) · `*_WIDTH` localparams: unchanged (the test renders a modified ISA in memory; the real package is not touched)

## Goal

Turn the guiding goal (D2) into a test (Q16): adding an instruction takes one entry in the ISA package, one execute handler and one test, and everything else (opcode, layout, encoding, assembler, disassembly, docs, editor data, SV package, timing checks, F-1 reserved encodings) follows with no other edit. The test adds a dummy instruction and checks every derived artifact. Where a manual edit is still needed today, the test lists it explicitly, so the list can only shrink.

## Background

- CLAUDE.md "Adding an Instruction" lists 5 steps; `docs/content/adding-instruction.md` describes the same process. Nothing checks that the steps are sufficient.
- After E03-09 to E03-18, `load_isa`, `compute_layout`, the codecs, the disassembler and the generators all accept an `Isa` argument, so a modified ISA can be built and rendered in-process.
- The assembler (`ipu_as`) and the emulator (`ipu_emu`) read the default `isa()` at import. E05 (assembler on the slot codec) and E04 (handler modules per slot, E04-14) finish making them fully derived.
- The LR slot uses 9 of 16 opcodes (4-bit field) and is padded to 20 bits (`union_layout.py:51-53`); a two-register instruction fits without changing any width.

## Implementation

1. The dummy instruction, as a dict literal in the test: `TESTNEG` in the LR slot, operands `dest` (`LrIdx`, `write: true`) and `src` (`LrIdx`, `read: cycle_start`), handler `execute_testneg` computing `LR[dest] = -LR[src]` (32-bit wrap), doc with an example `TESTNEG LR1 LR0;;`.
2. **Part A, in-process** (`ipu-isa/test/test_one_edit.py` plus a generator-side test in the package that owns the generators): build `load_isa(spec_with_dummy, ...)` and assert, with no other change:
   - opcode 9 in the LR slot, every existing opcode unchanged, every slot width and `word_bits()` unchanged;
   - slot codec encode/decode round trip and `disassemble` text `TESTNEG LR1 LR0`;
   - derived syntax `TESTNEG dest src`;
   - timing validator accepts it (Control-slot read at cycle start, declared write);
   - `reserved_encodings` for LR drops from 7 to 6;
   - rendered SV package contains `LR_INST_OPCODE_TESTNEG = 4'd9` and union member `testneg`, and every `*_WIDTH` equals the E02-03 table;
   - generated instruction reference and `isa.json` data contain `TESTNEG`.
3. **Part B, end to end in a subprocess**: copy the `ipu_isa`, `ipu_as`, `ipu_emu` (and `ipu_gen`) sources to a temp dir; apply exactly two edits, inserting the dict into `instruction_spec.py` and the handler into the handler location (`Ipu` in `ipu.py` today; the LR handler module after E04-14); then assemble `SET LR0 CR2 ;; TESTNEG LR1 LR0 ;; HALT ;;` with `CR2 = 5` and run it in the emulator; assert `LR1 == -5`.
4. `KNOWN_MANUAL_EDITS` in the test: a list of `(file, reason)` for any extra edit Part B needs today to pass (expected empty after E03-11; if E04/E05 work is still pending, list it with the issue that removes it). The test applies those edits too and fails if a listed edit is no longer needed, so the list cannot go stale.
5. Link the test from CLAUDE.md's "Adding an Instruction" checklist and from `adding-instruction.md` as the proof the checklist is complete.

## Tests

- [ ] Part A and Part B pass.
- [ ] Removing the handler edit from Part B fails with the import-time "names handler ... which does not exist" error (`ipu_emu/ipu.py:471-478`), proving the test exercises the real path.
- [ ] The test runs in under 60 s and is in the default tier.

## Acceptance Criteria

- [ ] A CI test proves that one ISA entry plus one handler (plus its test) is enough to add an instruction.
- [ ] Any remaining manual step is listed by file and owning issue, and the list is empty when E03, E04-14 and E05-02 have landed.
