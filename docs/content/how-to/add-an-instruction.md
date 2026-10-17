# Add an instruction

An instruction takes three edits: its entry in the ISA package, its handler in
the emulator, and a test. Everything else is derived from the entry: the
opcode, the instruction word layout, the assembler's parser and encoder, the
disassembler, the syntax line, the generated docs, the editor's hover and
completion, and the SystemVerilog package.

The "one edit" test proves it. It adds a dummy instruction, `TESTNEG`, to copies
of the sources with only the first two edits, then assembles and runs a program
that uses it (`src/tools/ipu-apps/test/test_one_edit_end_to_end.py`). The examples
below are that test's own entry and handler.

## Before you start

Decide three things:

- **The slot** the instruction issues in: `lr`, `load`, `mult`, `acc`, `aaq`,
  `store`, `acc_store`, `cond` or `break`. The [Instructions](../instructions.md)
  reference lists what each slot holds today.
- **The tag**, which the instruction takes from its slot:
    - **HARDWARE** (every slot but `acc_store` and `break`) is in the IPU, so it
      is in the hardware word and the SystemVerilog package.
    - **EMULATOR** (`acc_store`, `break`) exists only in the emulator. The
      assembler rejects it under `--target hardware`.
    - **ASSEMBLER** is a pseudo-instruction: the assembler expands it into a
      real instruction, and it has no opcode. See
      [Pseudo-instructions](#pseudo-instructions).
- **The mnemonic**, under the standard in the [Glossary](../reference/glossary.md):
  control-stage instructions have plain names (`ADD`, `BEQ`), datapath ones
  `UNIT.OP[.VARIANT]` (`LD.MULT`, `ACC.ADD.FIRST`), and none has an underscore.

## Step 1: the ISA entry

Add the entry to the slot's dictionary in `INSTRUCTION_SPEC`, in
`src/tools/ipu-isa/src/ipu_isa/instruction_spec.py`. This is `TESTNEG` at the
end of the `"lr"` dictionary:

```python
--8<-- "src/tools/ipu-isa/test/one_edit.py:23:39"
```

- **`operands`**: the name and type of each operand, in assembly order. A
  register the instruction reads has `"read"`: `"cycle_start"` (its value when
  the word began) or `"forwarded"` (after earlier slots of the same word wrote
  it). The register it writes has `"write": True`. The types are on the
  [Operand types](../operand-types.md) page.
- **`implicit_reads`**, **`implicit_writes`** (optional): every register the
  handler reads or writes that no operand names, such as
  `{"register": "mult_res", "view": "forwarded"}`.
- **`doc`**: the title, summary, one line per operand starting with its name,
  the operation, and an example that assembles. There is no syntax line: it is
  derived from the mnemonic and the operand names (`TESTNEG dest src`).
- **`execute_fn`**: the handler's name (step 2).

The opcode is the entry's position in its slot, so append the entry to keep
every other opcode where it is; never assign one. The entry is checked when
`ipu_isa` is imported: an unknown key, a misnamed operand line or a read the
[timing model](../explanation/timing-model.md) does not allow raises `SpecError`
with the entry's path, such as `lr.TESTNEG.operands[src].read`.

## Step 2: the handler

Add `execute_<name>` to the module of the instruction's slot, under
`src/tools/ipu-emu-py/src/ipu_emu/slots/` (`lr.py` for `lr`; `store.py` holds
`store` and `acc_store`; `brk.py` is `break`).
This is `TESTNEG`'s, in `lr.py`:

```python
--8<-- "src/tools/ipu-isa/test/one_edit.py:44:46"
```

- The first parameter is the engine, `ipu`. The rest are keyword-only (the
  `*`), and their names are exactly the operand names of the entry. The emulator binds every entry's
  `execute_fn` when it is imported and names any handler that is missing or
  whose parameters differ.
- An operand with `"read"` arrives resolved: an LR or CR as its 32-bit value,
  a mult-stage register as its bytes, which the handler decodes. One without,
  such as the register written, an immediate or a label, arrives as its index
  or value.
- A vector handler reads its lanes through the active lane codec (`ipu._codec`,
  from `ipu_emu.numerics`), computes on them, and writes the result back through
  the same codec. It has no branch on the element format. `execute_mult_rc_vv`
  in `mult.py` is a short example.
- A slot module never imports another one. A helper two slots need goes in
  `ipu_emu/geometry.py` or on `CycleView`; one only this slot uses stays a
  private function in the module.

## Step 3: the test

Add a test with the shared emulator test helpers in
`src/tools/ipu-emu-py/test/emu_support.py`, next to the tests of the same slot
in `test_execute.py`. `run` assembles a program, runs it to `HALT` and returns
the final state; `make_state` stops before running, so the test can write XMEM
first. A test of `ADD`, an existing `lr` instruction:

```python
--8<-- "docs/examples/test_add_an_instruction.py:test-import"
--8<-- "docs/examples/test_add_an_instruction.py:test"
```

## Step 4: check

```bash
bazel test //...
```

Some tests pin today's ISA, and fail by design once it has one more
instruction. Update them; the failing test names what it pins.

- **Baselines** regenerate: run `tools/update_baselines.sh`. For an `lr`
  instruction it rewrites nine files, among them the generated pages, the
  editor's data, the decoded field maps and the SystemVerilog golden. Commit
  them with the reason, as the [Baselines](../code-conventions.md#baselines)
  rule asks.
- **Facts written by hand** are edited:
    - the record of every instruction, `src/tools/ipu-isa/test/fixtures/spec_snapshot.json`;
    - the reserved encodings: the per-slot counts in
      `src/tools/ipu-isa/test/test_faults.py`, and the F-1 table in
      [Faults](../requirements/faults.md);
    - for an `lr` instruction, the "one edit" tests' own count of `lr`
      instructions (`test_one_edit.py` in ipu-isa, `test_one_edit_generators.py`
      in ipu-gen).
- An instruction whose opcode does not fit in its slot's opcode field (a slot of
  16 instructions getting a 17th) widens the slot, and with it the instruction
  word. More tests then fail, those that pin widths and the word layout. For a
  HARDWARE slot it is a breaking change to the wire format (`CLAUDE.md`,
  below). The list above is for an instruction that fits.

For a HARDWARE instruction, check the generated SystemVerilog package: a
downstream RTL repository vendors it, and its widths are wire format. Follow
the [Generated SystemVerilog Package](https://github.com/rechefe/ipu-emulator/blob/master/CLAUDE.md#generated-systemverilog-package)
section of `CLAUDE.md`.

For an EMULATOR instruction, check that a program using it is rejected for the
hardware:

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --target hardware --input "$PWD/prog.asm" --output "$PWD/prog.bin"
```

## What you do not edit

No code has a list of instructions to extend. The opcode, the assembler's
grammar and encoder, the disassembler, the generated pages and the editor's
data all follow from the entry. If the code needed an edit beyond the entry and
the handler, the "one edit" test would list it in `KNOWN_MANUAL_EDITS`. That
list is empty. Only the tests and pages that pin today's ISA change, as in
step 4.

A **new operand type** is the exception: it needs an entry in
`src/tools/ipu-isa/src/ipu_isa/operand_types.py`, a token class in the
assembler's `OPERAND_TYPE_MAP` (`src/tools/ipu-as-py/src/ipu_as/inst.py`). The
emulator picks how a read operand resolves from the register banks its domain
names (`_READ_BY_BANKS` in `src/tools/ipu-emu-py/src/ipu_emu/ipu.py`); only a
type over a new bank needs a read code in `ipu_emu/resolver.py`.
Its `field_suffix` and `sv_typedef` are wire format.

## Pseudo-instructions

When an existing instruction already does the work under another name or
operand order, add a pseudo-instruction instead. `BGT.S a b label` is
`BLT.S b a label`, for example. It needs only an entry in
`PSEUDO_INSTRUCTION_SPEC`, in the same file:

- `operands`, as for an instruction;
- `expands_to`: the target's `slot`, its `instruction`, and its `args` in
  order, each one of the pseudo's operand names or a fixed register such as
  `"CR0"`;
- `doc`, as for an instruction.

It has no `execute_fn`, no opcode and no handler, since the emulator never sees
it. The assembler picks a pseudo-instruction by name and operand count, so one
may share a name with an instruction of another arity. Its test checks it
assembles to the same words as its expansion (in
`src/tools/ipu-as-py/test/test_assemble.py`):

```python
--8<-- "docs/examples/test_add_an_instruction.py:pseudo-import"
--8<-- "docs/examples/test_add_an_instruction.py:pseudo"
```
