# Adding Instructions

This guide shows how to add a new instruction to the IPU assembler and emulator.

## Overview

The IPU uses a **single source of truth** for all instruction definitions: `INSTRUCTION_SPEC` in `src/tools/ipu-isa/src/ipu_isa/instruction_spec.py`.

Adding an instruction takes three edits, then two checks:

1. **Add to instruction_spec** — define the instruction with operands and documentation (instruction **mnemonic keys** are **upper case**; each operand’s `"name"` is **lower case**)
2. **Implement execution handler** — add `execute_<name>` to `ipu_emu/slots/<slot>.py`, the emulator's handler module for the instruction's slot
3. **Add tests** — verify the instruction works correctly
4. **Verify integration** — the assembler accepts it and `bazel test //...` passes
5. **Check the SystemVerilog package** — its widths are wire format

The assembler accepts mnemonics in any case; written examples and the spec use upper case by convention.

No manual opcode management needed — opcodes are automatically derived from instruction position.

A **new operand type** needs one entry in the operand-type registry, `src/tools/ipu-isa/src/ipu_isa/operand_types.py` (kind, width, value domain, readability, wire names and documentation), and one token class in the assembler's `OPERAND_TYPE_MAP` (`src/tools/ipu-as-py/src/ipu_as/inst.py`), plus, if it is readable, a read code in the emulator's operand resolver (`src/tools/ipu-emu-py/src/ipu_emu/resolver.py`). Everything else (the valid-type set, union widths, decoded field names, SV typedefs, the generated [Operand types](operand-types.md) page) is derived from the registry. A missing registry entry or token class, or a token class whose name, width, value range or members disagree with the registry, fails at import naming the type. Its `field_suffix` and `sv_typedef` are wire format: changing one is a breaking change.

## Step 1: Add Instruction to `instruction_spec.py`

Open `src/tools/ipu-isa/src/ipu_isa/instruction_spec.py` and find the `INSTRUCTION_SPEC` dictionary. Locate the slot type where your instruction belongs (e.g., `"mult"`, `"acc"`, `"load"`, `"store"`, `"acc_store"`, `"lr"`, `"cond"`, `"break"`).

Add your instruction to the slot's dictionary (**instruction keys are upper-case mnemonics**, e.g. `MULT.EE`, `ST.ACC`; operand `"name"` fields stay **lower case**):

```python
INSTRUCTION_SPEC = {
    "mult": {
        # ... existing mult instructions ...

        "MY_NEW_INSTRUCTION": {
            "operands": [
                {"name": "dest", "type": "MultStageReg", "write": True},
                {"name": "src_a", "type": "MultStageReg", "read": "cycle_start"},
                {"name": "src_b", "type": "MultStageReg", "read": "cycle_start"},
            ],
            "implicit_reads": [{"register": "r_mask", "view": "forwarded"}],
            "doc": InstructionDoc(
                title="My New Operation",
                summary="Performs a custom operation on two source registers.",
                operands=[
                    "dest: Destination mult-stage register (R0 or R1)",
                    "src_a: First source mult-stage register",
                    "src_b: Second source mult-stage register",
                ],
                operation=(
                    "for i in [0, R_REG_SIZE):\n"
                    "    dest[i] = custom_operation(src_a[i], src_b[i])"
                ),
                example="MY_NEW_INSTRUCTION R0 R1 R0;;",
            ),
            "execute_fn": "execute_my_new_instruction",
        },
    },
    # ... other slot types ...
}
```

The spec is checked strictly when `ipu_isa` is imported (`validate_instruction_spec`):

- An instruction has the keys `operands`, `doc` and `execute_fn`, and optionally `tag`, `implicit_reads` and `implicit_writes`; an operand has `name`, `type` and optionally `read` or `write`. Any other key is an error, so a typo such as `"raed"` fails at import instead of being ignored. The allowed keys are listed in `src/tools/ipu-isa/src/ipu_isa/schema.py`; a new key goes there first.
- Operand names are unique within the instruction and match `[a-z][a-z0-9_]*`.
- `read` and `write` are allowed only on register operands (`LrIdx`, `CrIdx`, `LcrIdx`, `MultStageReg`, `LrdIdx`), and not both on one operand.
- `implicit_reads` / `implicit_writes` entries name a register of `REGISTER_DEFINITIONS`, optionally the operand that selects its index (`via_operand`), and for a read its `view`.
- Each `doc.operands` line starts with its operand's name, in operand order, and `doc.example` has no comma.
- Mnemonics are unique within the slot ignoring case, and each slot has exactly one `NOP`.
- A pseudo-instruction has exactly `operands`, `expands_to` and `doc`, and its `expands_to` has exactly `slot`, `instruction` and `args`. Each argument is one of the pseudo's own operand names or a register valid for the target operand's type (`CR0` for an `LcrIdx`).

A failure raises `SpecError` with the path of the entry, for example `mult.MY_NEW_INSTRUCTION.operands[src_a].read`.

`load_isa` then checks the declared register accesses against the timing contract in `src/tools/ipu-isa/src/ipu_isa/timing.py` (ADR 0002):

- A `forwarded` read is allowed only when every slot that writes the register runs earlier in `SLOT_EXECUTION_ORDER`; a register written by the same slot is read at `cycle_start`.
- CR reads, and every read in a Control slot (`lr`, `cond`, `break`), are at `cycle_start`.
- Every `lr`-slot instruction except `NOP` marks the LR it writes with `"write": True` (the emulator's LR write-conflict check reads it).
- A read whose view differs from `contract_view` must be listed in `KNOWN_DEVIATIONS`, with the E04 issue that fixes it. A new instruction should follow the contract and need no row.

**Key fields:**

- **`operands`**: List of operand definitions, each with:
  - `name`: Meaningful name for the operand (used in handler signature)
  - `type`: Operand type — one of:
    - `"MultStageReg"` — `R0` or `R1` (2-bit encoding in the VLIW word)
    - `"LrIdx"` — `LR0`–`LR15`
    - `"CrIdx"` — `CR0`–`CR15`
    - `"LcrIdx"` — `LR0`–`LR15` or `CR0`–`CR15`
    - `"LrModPow2KImmediate"` — exponent **k** for `INC.MODP2` (semantic **k ∈ [1, 9]**)
    - `"MultMaskOffsetImmediate"` — mult mask slot index **0**–**7**
    - `"BreakImmediate"` — 16-bit break condition
    - `"Label"` — branch target label
  - `read` (optional): `"cycle_start"` or `"forwarded"` — marks source registers whose values are auto-resolved:
    - `"cycle_start"` — the value the register held when the instruction word began
    - `"forwarded"` — the value after the writes of earlier slots of the same word
    - Omit for destination registers or indices passed as raw values
  - `write` (optional): `True` on the destination register operand; the handler receives its raw index

- **`implicit_reads`** / **`implicit_writes`** (optional): every register the handler reads or writes that is not a `read` / `write` operand, as `{"register": "mult_res", "view": "forwarded"}` (`via_operand` names the operand that selects the register's index, for example `"dest_slot"`; writes have no `view`). The generated [Assembly syntax](assembly-syntax.md#execution-order-and-timing) page lists them for every instruction.

- **`doc`**: `InstructionDoc` with:
  - `title` — Short human-readable name
  - `summary` — What the instruction does
  - `operands` — Description of each operand (type and meaning), each line starting with the operand's name
  - `operation` — Pseudo-code showing the operation
  - `example` — Code example that assembles: operands separated by spaces, no commas

  There is no syntax line to write: the generated docs and editor hover derive
  it from the mnemonic and the operand names (`MY_NEW_INSTRUCTION dest src_a src_b`).

- **`execute_fn`**: Name of the handler function that executes this instruction, in the slot's handler module `ipu_emu/slots/<slot>.py` (e.g., `"execute_my_new_instruction"`)

**Opcode assignment:**

Opcodes are **automatically derived from position** within the slot's instruction list. The first instruction gets opcode 0, the second gets opcode 1, etc. If you want to preserve existing opcodes, append new instructions to the end. To change opcodes, reorder instructions (both assembler and emulator will update automatically).

## Step 2: Implement Execution Handler

Each slot has one handler module in `src/tools/ipu-emu-py/src/ipu_emu/slots/`: `lr.py`, `load.py`, `mult.py`, `acc.py` (ACC, AGG and RESHAPE), `aaq.py`, `store.py` (the STORE and ACC_STORE slots), `cond.py` and `brk.py` (the BREAK slot; `break` is a Python keyword). `ipu_emu.slots.HANDLER_MODULES` maps each slot to its module.

Add your handler to the module of your instruction's slot, here `mult.py`:

```python
def execute_my_new_instruction(ipu: Ipu, *, dest: int, src_a: bytearray, src_b: bytearray) -> None:
    """Execute MY_NEW_INSTRUCTION: performs custom operation.

    Args:
        ipu: The engine (state, register views, issuer).
        dest: Destination register index (raw MultStageRegField value)
        src_a: Source A register bytes (auto-resolved, cycle start)
        src_b: Source B register bytes (auto-resolved, cycle start)
    """
    result = bytearray(R_REG_SIZE)

    for i in range(R_REG_SIZE):
        # Perform your custom operation
        result[i] = (src_a[i] + src_b[i]) & 0xFF  # Example: element-wise add

    # Write result to destination register
    reg_name, elem_idx = _MULT_STAGE_MAP[dest]
    ipu.state.regfile.set_register_bytes(reg_name, elem_idx, result)
```

When the emulator is imported, `ipu_emu.ipu` binds every instruction's `execute_fn` from its slot's module. It raises `RuntimeError` if the function is missing, or if its parameters are not the engine followed by the operand names, keyword-only (`test_slot_modules` checks every instruction).

A slot module never imports another slot module. A helper two slots need goes in `ipu_emu/geometry.py` (lane formats, row addresses, LRD bytes) or on `CycleView` (register reads); a helper only one slot uses stays a private function in that module.

**Key points:**

- **Function signature**: `def execute_<instruction_name>(ipu, *, <operand_names>) -> None:`
  - Use **keyword-only arguments** (note the `*` before operand names)
  - Parameter names **must match** the `name` fields in `instruction_spec`
  - Parameter order doesn't matter (they're passed as kwargs)

- **Auto-resolved operands** (those with `"read"` field):
  - Register operands with `"read": "cycle_start"` receive `bytearray` values (register contents) in narrow mode. In wide-vector mode a `MultStageReg` operand receives the register index instead (`0` for `R0`, `1` for `R1`), and the handler reads the wide lanes itself
  - LR/CR operands with `"read"` receive `int` values (32-bit register value)
  - The dispatcher reads them at cycle start or forwarded, as the `read` value says

- **Raw operands** (no `"read"` field):
  - Destination registers receive the raw index as `int`
  - Immediates receive the literal value as `int`
  - Labels receive the resolved address as `int`

- **Common patterns**:
  - Use `_MULT_STAGE_MAP[dest]` to get `(reg_name, elem_idx)` from a MultStageReg index
  - Use `ipu.state.regfile.set_register_bytes()` to write to a register
  - Use `ipu.state.dtype` to get the current data type
  - Read registers through `ipu._view` (`CycleView`), which serves the snapshot or the live value as the instruction declares
  - Use `ipu_mult()`, `ipu_add()` from `ipu_emu.ipu_math` for typed arithmetic

**Available helpers:**

```python
# Register reads, in the view the instruction declares (ipu._view is a CycleView)
ipu._view.lr(idx, "operand") -> int            # Read LR register
ipu._view.read("r_acc") -> memoryview          # Read a register's bytes

# Register writes (the live register file)
ipu.state.regfile.set_lr(idx, value: int)      # Write LR register
ipu.state.regfile.set_r_acc_bytes(data: bytearray)  # Write accumulator

# Register metadata
reg_name, elem_idx = _MULT_STAGE_MAP[idx]      # Resolve MultStageReg index
R_REG_SIZE                                      # Register size in bytes (128)
R_ACC_SIZE                                      # Accumulator size (512)

# State access
ipu.state.xmem.read_address(addr, size) -> bytearray
ipu.state.xmem.write_address(addr, data: bytearray)

# Math operations
from ipu_emu.ipu_math import ipu_mult, ipu_add, DType
ipu_mult(a: int, b: int, dtype: DType) -> int
ipu_add(a: int, b: int, dtype: DType) -> int
```

## Step 3: Add Tests

Emulator instruction tests live in `src/tools/ipu-emu-py/test/test_execute.py`. Its helpers assemble real source, so a test needs no hand-built instruction word:

- `_run(asm_code, cr=...)` assembles, loads, runs to `HALT` and returns the final `IpuState`.
- `_make_state(asm_code, cr=...)` assembles and loads without running, so you can write input data to XMEM first; then call `run_until_complete(state)`.

Add a test next to the tests for the same slot:

```python
def test_my_new_instruction():
    """MY_NEW_INSTRUCTION adds R0 and R1 element-wise into R0."""
    # CR8 = row 32 (byte 0x1000), CR9 = row 33 (byte 0x1080).
    state = _make_state("""\
SET LR0 CR8;;
SET LR1 CR9;;
LD.MULT R0 LR0 CR0;;
LD.MULT R1 LR1 CR0;;
MY_NEW_INSTRUCTION R0 R0 R1;;
HALT;;
""", cr={8: 32, 9: 33})
    state.xmem.write_address(0x1000, bytes(range(128)))
    state.xmem.write_address(0x1080, bytes([1] * 128))
    run_until_complete(state)

    expected = bytearray((i + 1) & 0xFF for i in range(128))
    assert state.regfile.get_r(0) == expected
```

Run tests:

```bash
bazel test //src/tools/ipu-emu-py:test_execute
```

## Step 4: Verify Integration

After adding your instruction, verify it works in both assembler and emulator:

```bash
# Verify assembler accepts the instruction (check reads stdin with --input -)
echo "MY_NEW_INSTRUCTION R0 R1 R0;;" | bazel run //src/tools/ipu-as-py:ipu-as -- check --input -

# Run all tests
bazel test //...
```

## Step 5: Check the Generated SystemVerilog Package

A downstream RTL repository vendors the generated `ipu_instr_pkg.sv`, so a new instruction changes its wire format. Run the two commands from the [CLAUDE.md "Generated SystemVerilog Package" section](https://github.com/rechefe/ipu-emulator/blob/master/CLAUDE.md#generated-systemverilog-package) and follow its rules for comparing the `localparam` widths against the base commit:

```bash
bazel test //src/tools/ipu-gen:test_sv_package_compiles
bazel build //docs:generate_instruction_format_artifacts
```

## Adding a Pseudo-Instruction

Some assembly mnemonics don't need any new hardware behavior — they're just a
more convenient way of writing an existing instruction. For example, `BGT.S
reg1 reg2 label` is exactly `BLT.S reg2 reg1 label` with the operands
swapped: there's no new opcode, no new emulator behavior, just a friendlier
name. These are **pseudo-instructions**, declared in
`PSEUDO_INSTRUCTION_SPEC` (in the same `instruction_spec.py` file as
`INSTRUCTION_SPEC`).

Pseudo-instructions are expanded by the assembler at compile time, before a
compound instruction is encoded. They:

- **Never get an opcode** — they don't occupy a slot, so they're never
  assigned one
- **Never need an `execute_fn`** — the emulator never sees them; only the
  real instruction they expand into appears in the binary
- **Cost nothing at runtime** — the expansion happens once, in the assembler

### Declaring a pseudo-instruction

Add an entry to `PSEUDO_INSTRUCTION_SPEC`:

```python
PSEUDO_INSTRUCTION_SPEC: dict[str, dict] = {
    "BGT.S": {
        "operands": [
            {"name": "reg1", "type": "LcrIdx"},
            {"name": "reg2", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BLT.S",
            "args": ["reg2", "reg1", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Greater Than (pseudo)",
            summary="Branch if first register is greater than second.",
            operands=[
                "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                "label: Branch target label",
            ],
            operation="if (reg1 > reg2) PC = label",
            example="BGT.S LR0 LR1 bigger;;",
        ),
    },
}
```

**Key fields:**

- **`operands`**: the pseudo-instruction's own operand list, exactly like a
  real instruction's `operands` (same `name`/`type` rules).
- **`expands_to`**: how to rewrite the pseudo into a real instruction:
  - `slot` — the real instruction's slot (e.g. `"cond"`)
  - `instruction` — the real instruction's name in `INSTRUCTION_SPEC`
  - `args` — the real instruction's operands, in order. Each entry is
    either one of this pseudo's own operand names (substituted with
    whatever the caller wrote) or a literal register name such as `"CR0"`
    (useful when a pseudo-instruction hard-codes a register — e.g. `BZ reg
    label` expands to `BEQ reg CR0 label`, assuming `CR0` always holds
    zero).
- **`doc`**: an `InstructionDoc`, same as for real instructions — this is
  what generates the pseudo-instruction's entry in the programmer-facing
  guide.
- **No `execute_fn`** — adding one is a validation error
  (`validate_pseudo_instruction_spec` rejects it), since pseudo-instructions
  never reach the emulator.

### Resolution and disambiguation

A pseudo-instruction is matched by **name + operand count**
(`isa().pseudo(mnemonic, arity)` in `ipu_isa.view`). This means a pseudo-instruction could share a
name with a real instruction of a *different* arity without ambiguity — the
assembler picks whichever matches the operand count the caller wrote.

If a name matches a pseudo-instruction but the operand count doesn't match
any definition (and no real instruction shares that name either), the
assembler raises a clear error naming the expected operands instead of a
generic "instruction not found" message.

### Testing a pseudo-instruction

Verify that the pseudo-instruction assembles to the **exact same binary** as
its hand-written real-instruction expansion:

```python
from ipu_as.lark_tree import assemble

def test_bgt_matches_hand_written_blt():
    assert assemble("BGT.S LR0 LR1 +1;;") == assemble("BLT.S LR1 LR0 +1;;")
```

See `src/tools/ipu-as-py/test/test_assemble.py` for more examples, including
tests for arity disambiguation and misuse errors.

## Summary

1. **Add to `instruction_spec.py`** — define operands, docs, and `execute_fn`
2. **Implement `execute_<name>` in `ipu_emu/slots/<slot>.py`** — write the execution logic
3. **Add tests** — verify correctness in `test_execute.py`
4. **Check the SystemVerilog package** — its widths are wire format
5. **No opcode management** — opcodes are derived from position automatically

The instruction will automatically work in:
- Assembler (syntax highlighting, parsing, binary encoding)
- Emulator (execution, operand resolution, register updates)
- Documentation generation (help text, reference docs)

All from the single source of truth in `instruction_spec.py`.

The "one edit" tests check that this list is complete. `//src/tools/ipu-apps:test_one_edit_end_to_end` copies the sources, adds a dummy `TESTNEG` instruction with exactly one spec entry and one handler, then assembles and runs a program that uses it. `//src/tools/ipu-isa:test_one_edit` and `//src/tools/ipu-gen:test_one_edit_generators` check the derived opcode, layout, codec, disassembly, timing check, reserved encodings, SV package, instruction reference and editor data. Any further edit an instruction still needs is listed, with the issue that removes it, in `KNOWN_MANUAL_EDITS` in the end-to-end test; the list is empty.