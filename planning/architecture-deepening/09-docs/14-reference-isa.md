# Reference: ISA pages generated from `ipu_isa`

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-14 · Size M · Depends on: E09-04, E09-06, E03 (`ipu_isa` package, typed view, tags, element formats, derived syntax, mnemonic rename), E08 (`ipu_gen` docs generator) · Behaviour change: none

## Goal

The ISA reference (instructions, operand types, registers, assembly syntax,
instruction format, configuration registers) is generated from `ipu_isa`,
follows the CLAUDE.md conventions, and is the only place these facts are
written.

## Background

- Five ISA pages are generated today by `ipu_as/gen_docs.py`
  (`docs/BUILD.bazel:15-67`): `instructions.md`, `operand-types.md`,
  `assembly-syntax.md`, `programmer-guide.md`, `instruction-format.md`.
- They break the CLAUDE.md conventions (checked in a local build of
  `//docs:generate_mkdocs_md`):
  - `assembly-syntax.md:19-23` writes registers in lower case
    (`LDR_MULT_REG r0 lr0 cr0`), and `:27` states the opposite convention
    ("Operand tokens use lower case (`lr0`, `r0`, `cr0`)").
  - Syntax lines use commas, for example `BEQ reg1, reg2, label`
    (`instructions.md:976`); 34 `syntax=` strings in `instruction_spec.py`
    contain ", ". Q15 derives syntax from mnemonic and operands (E03), which
    removes the hand-written strings.
  - Descriptions give sizes in bytes ("256 bytes", `instructions.md:1449`)
    where the convention is elements.
- `ipu-configuration.md` (154 lines, hand-written) documents dstructure fields
  (`valid_elements`, `partition`, `pad_mode`), dtype selection and CR
  constants. Part of it duplicates the instruction reference.
- E03 adds HARDWARE/EMULATOR/ASSEMBLER tags (Q17), element formats (Q47), the
  timing contract (Q32) and the fault list F-1 to F-9 (Q52) to `ipu_isa`.
- E08 moves the generators into `ipu_gen`.
- `history_link.py` maps generated pages to generators (E09-04).

## Implementation

This issue defines what the generated pages show and checks the result. The
generator code changes happen in E08's `ipu_gen`; if E08 has already landed, make
them there in this issue.

1. Pages, all under `reference/isa/`:
   - **Instructions**: one section per slot; per instruction the mnemonic,
     tag (HARDWARE/EMULATOR/ASSEMBLER), derived syntax with space-separated,
     lower-case placeholders, operands with types, semantics, pseudocode with
     sizes in elements, timing-contract reads (which operands are read at
     cycle start), and the faults it can raise (F-IDs).
   - **Operand types**, **Registers** (from the register definitions in
     `ipu_isa`, upper-case names), **Assembly syntax**, **Instruction format**
     (bit layout, HARDWARE slots first, then EMULATOR slots, Q41).
   - **Configuration registers**: the dstructure CR fields and the CR scalar
     format rule (link E09-19). Fold the reference parts of
     `ipu-configuration.md` here; move the dtype or format selection text to
     E09-19. Delete `ipu-configuration.md`.
   - `programmer-guide.md`: keep its content where it is reference; move
     explanation to E09-17 or E09-18. Open question: confirm with the user
     what the programmer's guide is for after the restructure.
2. **Convention test** in the generator's test target: generated markdown has
   no lower-case register tokens (`\b(r|lr|cr|lrd)\d+\b` outside identifiers),
   no comma-separated operands in syntax lines, and no "bytes" in pseudocode
   size comments.
3. Run E09-01's asm-block test over the generated pages as well, once they
   pass: a generated example that does not assemble is a generator bug.
4. Update the `generated_pages` map (E09-04) with the new paths and the
   `ipu_gen` module.
5. `SKILL.md` and `CLAUDE.md` link these pages instead of listing operand types
   or slots.

## Tests

- [ ] The convention test passes and fails on a planted `lr0` or comma.
- [ ] Every `asm` example on the generated pages assembles.
- [ ] `bazel build //docs:build_docs` shows no new warnings.

## Acceptance Criteria

- [ ] The ISA reference is generated only, with no hand-written copy of any
      table in `docs/content`.
- [ ] The pages follow the CLAUDE.md naming, separator and element-unit
      conventions, under test.
- [ ] Every instruction shows its tag and the faults it can raise.
- [ ] `ipu-configuration.md` is gone; its content lives in the generated
      configuration-register page and in E09-19.
