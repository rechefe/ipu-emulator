# Fix drift in CLAUDE.md, SKILL.md and the adding-instruction walkthrough

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-02 · Size S · Depends on: E09-01 · Behaviour change: none

## Goal

The instructions an agent or a new contributor follows first are correct today.
This is a fact-fix pass, not a rewrite: the full rewrites happen in E09-07
(SKILL.md) and E09-11 (adding an instruction) after E03 and E04 land.

## Background

Verified at `694a8a8`:

- `SKILL.md:9`: names the project "ipu-c-samples". `SKILL.md:11` says
  "there is no C code" because of that old name.
- `SKILL.md:108`: lists `ReshapeMaskImmediate`. The operand type is
  `LrOrReshapeMaskImmediate` (`instruction_spec.py:34`, `:944`).
- `SKILL.md:118-124`: the "add an instruction" example gives `dest` (an
  `LrIdx` destination) a `"read": "snapshot"` key. A destination is a raw index
  and takes no `read` key; with `read` it receives the register value.
- `SKILL.md:177`: `ipu-as -- assemble --define ROWS=8 prog.asm.j2`. The CLI
  has no `--define` and no positional input (`cli.py:13-21`). `--define`
  arrives with E05.
- `SKILL.md:217-227`: the debugging section shows only the deprecated
  `run_with_debug` line debugger. `README.md:44-60` and `debugging.md:5-43`
  document `bazel run --config=debug`.
- `docs/content/adding-instruction.md`:
  - `:180` tells the reader to create `test_<slot>_instructions.py`. No such
    files exist; emulator instruction tests live in
    `src/tools/ipu-emu-py/test/test_execute.py`.
  - `:209` calls `state.regfile.set_pc(0)`. `RegFile` has no `set_pc`.
  - `:137` says `"read": "snapshot"` register operands receive a `bytearray`.
    In wide-vector mode a `MultStageReg` operand receives the index
    (`ipu.py:821-832`).
  - `:230` pipes text to `assemble --format hex -`. There is no `hex` format,
    and `assemble` has no stdin input. `ipu-as -- check --input -` reads stdin
    (`cli.py:62-83`).
  - It has no step for the SystemVerilog package check that `CLAUDE.md`
    ("Generated SystemVerilog Package") requires.
  - Its examples use lower-case register names (`r0 r1`) and commas (`BGT
    reg1, reg2, label`), against the CLAUDE.md conventions.

## Implementation

1. `SKILL.md`:
   - Rename the project to "ipu-emulator" and delete the "no C code" sentence.
   - Replace the operand-type list with a link to the generated operand-types
     page (`operand-types.md`). A hand-kept list drifts; the generated page
     cannot.
   - Remove `"read"` from the `dest` operand in the example.
   - Delete the `--define` line. Note that template defines come with E05.
   - Replace the debugging section with `bazel run --config=debug <target>`
     and a link to `debugging.md`.
2. `adding-instruction.md`:
   - Point the test step at `test_execute.py` and the `_run()` helper.
   - Replace the hand-built instruction dict and `set_pc` with `_run("...;;")`,
     which assembles and runs real source.
   - Replace step 4's pipe with:
     `echo "MY_NEW_INSTRUCTION R0 R1 R0;;" | bazel run //src/tools/ipu-as-py:ipu-as -- check --input -`.
   - Qualify the `bytearray` claim: narrow mode only; wide mode passes the
     index. (E04's per-slot handlers remove the difference; E09-11 then
     rewrites the sentence.)
   - Add a step "Check the generated SystemVerilog package" that runs the two
     commands from `CLAUDE.md` and links to that section instead of copying
     its rules.
   - Upper-case register names and space-separated operands in all examples.
3. `CLAUDE.md`: no change beyond E09-01's `--format bin`. Edit with
   `mattpocock-skills:writing-for-agents` if anything else needs touching.
4. Remove any `xfail` entries E09-01 added for these lines.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks //docs:test_doc_commands` passes
      with no `xfail` entries.
- [ ] Manually follow `adding-instruction.md` end to end with a throwaway
      instruction on a scratch branch: every command runs and `bazel test //...`
      passes. Record the result in the PR.

## Acceptance Criteria

- [ ] No mention of "ipu-c-samples", `set_pc`, `--format hex`,
      `ReshapeMaskImmediate` or `test_<slot>_instructions.py` remains
      (`grep -rn` over `SKILL.md`, `CLAUDE.md`, `README.md`, `docs/content`).
- [ ] The walkthrough includes the SystemVerilog package step.
- [ ] Examples follow the CLAUDE.md naming and separator conventions.
