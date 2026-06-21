# IPU Emulator & Assembler — AI Assistant Skill Reference

This file gives Claude Code the context needed to work effectively in this repository.

---

## Project Purpose

**ipu-c-samples** is a Python-based IPU (Inference Processing Unit) emulator and assembler toolchain. It lets you write assembly programs for a custom VLIW ISA, then assemble and emulate them in software.

Despite the repo name, there is no C code — it is pure Python with a Bazel build system.

---

## Repository Layout

```
src/tools/
├── ipu-common/src/ipu_common/    # Shared definitions (single source of truth)
│   ├── instruction_spec.py       # ALL instruction definitions live here
│   ├── registers.py              # ALL register definitions live here
│   ├── acc_stride_enums.py       # Stride control enums
│   └── types.py                  # RegDtype, RegKind, RegDescriptor
├── ipu-as-py/src/ipu_as/         # Assembler
│   ├── compound_inst.py          # VLIW encoding/decoding
│   ├── inst.py                   # Instruction classes per slot
│   ├── lark_tree.py              # Lark parser
│   └── opcodes.py                # Dynamic opcode generation from spec
├── ipu-emu-py/src/ipu_emu/       # Emulator
│   ├── ipu.py                    # execute_* handlers for every instruction
│   ├── ipu_state.py              # Top-level state (registers, memory, PC)
│   ├── execute.py                # VLIW decode + dispatch
│   ├── regfile.py                # Register file
│   ├── xmem.py                   # 2 MB external memory
│   ├── ipu_math.py               # Typed math (INT8, FP8 E1-E7)
│   └── debug_cli.py              # Interactive debugger
└── ipu-apps/src/ipu_apps/        # Sample applications
    └── fully_connected/          # FC neural network layer example
docs/                             # MkDocs (config + content/ page sources)
```

---

## Architecture: Single Source of Truth

**`instruction_spec.py` is the master file.** Adding an entry there automatically:
- Gives the assembler a new mnemonic (opcode derived from dict position — never assign manually)
- Lets the emulator dispatch to the corresponding `execute_*` method

Do not hardcode opcodes anywhere. Do not duplicate instruction metadata.

Same principle applies to **`registers.py`** for register definitions.

### Naming in prose, docs, and examples

- **Instruction mnemonics** are written in **upper case** everywhere in this repository (Markdown, comments, `InstructionDoc.syntax` / examples, and the keys in `INSTRUCTION_SPEC`). The assembler still accepts upper or lower case in source files.
- **Hardware register names** (`R0`, `R1`, `LR0`–`LR15`, `CR0`–`CR15`) are written in **upper case** in all docs and examples.
- **Abstract operand placeholder tokens** (`dest`, `src_a`, `offset`) in syntax templates are written in **lower case**.

---

## ISA Concepts

### VLIW / Compound Instructions

Every cycle executes one **compound instruction** — multiple independent slots in parallel:

```asm
LDR_MULT_REG R0, LR0, CR0; MULT.RC.VV LR1, R0, 0, LR3; ACC; ADD LR0, LR0, 1; BNE LR0, LR1, next;;
```

- Binary layout (MSB → LSB): `cond`, `lr` (×3), `load`, `mult`, `acc`, `aaq`, `store`, `acc_store`, `break`
- Separated by `;`, terminated by `;;`
- Missing slots → NOP inserted automatically
- Slots see a register **snapshot** at cycle start (read-before-write semantics)

### Register File

| Register | Size | Purpose |
|----------|------|---------|
| `R0`, `R1` | 128 bytes (128×INT8) | Multiply-stage inputs/outputs |
| `R_CYCLIC` | 512 bytes | Cyclic-access variant of `R0` |
| `R_MASK` | 128 bytes | Bit mask register |
| `R_ACC` | 512 bytes (128×INT32) | Accumulator |
| `LR0`–`LR15` | 32-bit each | Loop/scalar registers |
| `CR0`–`CR15` | 32-bit, read-only | Configuration (base addresses, params) |

`CR15` is reserved for the dstructure configuration register (`valid_elements` and `partition`). Data type selection lives on `IpuState.dtype` in the Python emulator.

### Instruction Slots

| Slot | Instructions | Purpose |
|------|-------------|---------|
| LOAD | `LDR_MULT_REG`, `LDR_CYCLIC_MULT_REG`, `LDR_MULT_MASK_REG` | First-stage memory loads (feeds multiply) |
| STORE | `STR_POST_AAQ_REG` | Last-stage memory store (drains `POST_AAQ_REG`) |
| ACC_STORE | `STR_ACC_REG` | **Simulation-only** — store `R_ACC` to external memory |
| MULT | `MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE` | 8-bit vector multiply |
| ACC | `ACC`, `ACC.FIRST`, `ACC.STRIDE`, `AGG.SUM`, `AGG.SUM.FIRST`, `AGG.MAX`, `AGG.MAX.FIRST` | Accumulate into `R_ACC`; in-place aggregation (sum/max) writing a single slot of `R_ACC` |
| AAQ | `AAQ`, `ACTIVATE` | **`ACTIVATE`** reads **`R_ACC`** and writes activated **32b** lanes into **`POST_AAQ_REG`** (512 B staging). **`AAQ`** (INT8) quantizes wide lanes in **`POST_AAQ_REG`** into the leading **128 B**; **`STR_POST_AAQ_REG`** stores the full **512 B** register to XMEM. See `docs/content/building-applications.md#activations-emulator`. |
| LR (×3) | `SET`, `ADD`, `SUB`, `INCR_MOD_POW2`, `INC`, `DEC` | Scalar loop register ops (`SET` copies from a **`CR`** register; `INC`/`DEC` read-modify-write with union-derived immediate) |
| COND | `BEQ`, `BNE`, `BLT`, `BGE`, `BR`, `BKPT` | Branches. `BGT`, `BLE`, `BZ`, `BNZ`, `B` are pseudo-instructions (assembler-expanded, no opcode) — see `PSEUDO_INSTRUCTION_SPEC` in `instruction_spec.py` |
| BREAK | `BREAK`, `BREAK.IFEQ` | Debug breakpoints |

### Operand Types (defined in instruction_spec)

`MultStageReg`, `LrIdx`, `CrIdx`, `LcrIdx`, `LrIncDecImmediate`, `FullXmemRow`, `ActivationFn`, `ElementsInRow`, `HorizontalStride`, `VerticalStride`, `LrModPow2KImmediate`, `MultMaskOffsetImmediate`, `BreakImmediate`, `Label`

---

## How to Add a New Instruction

1. **Define it in `instruction_spec.py`:**
   ```python
   "MY_INST": {
       "operands": [
           {"name": "dest", "type": "LrIdx", "read": "snapshot"},
           {"name": "src",  "type": "MultStageReg", "read": "snapshot"},
       ],
       "doc": InstructionDoc(summary="...", ...),
       "execute_fn": "execute_my_inst",
   }
   ```

2. **Implement the handler in `ipu.py`:**
   ```python
   def execute_my_inst(self, *, dest: int, src: bytearray) -> None:
       # dest is resolved to the LR value; src is the register bytes
       ...
   ```
   - Use `*,` (keyword-only args)
   - Argument names **must** match `operands[*]["name"]` exactly
   - The framework resolves operand types and passes values automatically

3. **Write tests in `ipu-emu-py/test/`** using the `_run()` helper.

4. **Run:** `bazel test //...`

---

## Testing Patterns

```python
def _run(asm_code: str) -> IpuState:
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    load_program(state, decoded)
    run_until_complete(state)
    return state

def test_my_instruction():
    state = _run("MY_INST LR0 R0;;")
    assert state.regfile.get_lr(0) == expected
```

Bazel test targets:
```bash
bazel test //src/tools/ipu-emu-py:test_execute
bazel test //src/tools/ipu-as-py:test_assemble
bazel test //src/tools/ipu-apps:test_fully_connected
bazel test //...
```

---

## Assembler Usage

```bash
# Assemble a program
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input prog.asm --output prog.bin

# With Jinja2 template variables
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --define ROWS=8 prog.asm.j2
```

Assembly files support full **Jinja2 preprocessing** (variables, loops, macros, conditionals).

---

## Application Pattern

```python
class MyApp(IpuApp):
    def setup(self, state: IpuState) -> None:
        load_binary_to_xmem(state, self.inputs_path, base_addr=0x0000)
        state.regfile.set_cr(0, 0x0000)

    def teardown(self, state: IpuState) -> None:
        dump_xmem_to_binary(state, self.output_path, base_addr=0x40000)
```

Each app lives under `ipu-apps/src/ipu_apps/<name>/` and has:
- `<name>.asm` — Assembly program
- `__init__.py` — `IpuApp` subclass
- `__main__.py` — Debug runner
- Tests in `ipu-apps/test/`

---

## Debugging

Add `BREAK;;` to assembly, then run with the debug callback:

```python
from ipu_emu.debug_cli import debug_prompt
run_with_debug(state, debug_prompt)
```

Interactive commands: `continue`, `step`, `get lr0`, `set lr0 100`, `save state.json`

---

## Key Things to Remember

- **Never assign opcodes manually** — they are derived from position in `instruction_spec.py`
- **Never duplicate instruction metadata** — assembler and emulator both read `instruction_spec.py`
- **Operand names in `execute_*` must exactly match** the names in `instruction_spec.py`
- **Read-before-write**: slots with `"read": "snapshot"` see pre-cycle register values
- **`CR15`** is reserved for dstructure configuration — never use it for application data
- The build system is **Bazel** — use `bazel build/test/run`, not pip/python directly
- Data types (`DType.INT8`, `DType.E4`, `DType.E5`, etc.) affect how register bytes are interpreted in math ops
