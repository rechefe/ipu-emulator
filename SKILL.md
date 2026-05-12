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
│   ├── acc_agg_enums.py          # Aggregation mode / post-function enums
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
│   ├── ipu_math.py               # Typed math (INT8, FP8_E4M3, FP8_E5M2)
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
- **Operand and field names** (`dest`, `src_a`, register tokens like `lr0` / `r0`) are written in **lower case**.

---

## ISA Concepts

### VLIW / Compound Instructions

Every cycle executes one **compound instruction** — multiple independent slots in parallel:

```asm
LDR_MULT_REG r0 lr0 cr0; MULT.EE r0 lr1 0 lr3; ACC; ADD lr0 lr0 1; BNE lr0 lr1 next;;
```

- Slots: `break`, `xmem`, `mult`, `acc`, `aaq`, `lr` (×3), `cond`
- Separated by `;`, terminated by `;;`
- Missing slots → NOP inserted automatically
- Slots see a register **snapshot** at cycle start (read-before-write semantics)

### Register File

| Register | Size | Purpose |
|----------|------|---------|
| `r0`, `r1` | 128 bytes (128×INT8) | Multiply-stage inputs/outputs |
| `r_cyclic` | 512 bytes | Cyclic-access variant of r0 |
| `r_mask` | 128 bytes | Bit mask register |
| `r_acc` | 512 bytes (128×INT32) | Accumulator |
| `aaq0`–`aaq3` | 32-bit each | Activation & Quantization results |
| `lr0`–`lr15` | 32-bit each | Loop/scalar registers |
| `cr0`–`cr15` | 32-bit, read-only | Configuration (base addresses, params) |

`cr15` is reserved for data type selection (INT8 / FP8_E4M3 / FP8_E5M2).

### Instruction Slots

| Slot | Instructions | Purpose |
|------|-------------|---------|
| XMEM | `LDR_MULT_REG`, `STR_ACC_REG`, `LDR_CYCLIC_MULT_REG`, … | Memory load/store |
| MULT | `MULT.EE`, `MULT.VE.CYCLIC`, `MULT.VE.PADDED`, `MULT.VE.CR`, `MULT.VE.AAQ`, … | 8-bit vector multiply |
| ACC | `ACC`, `ACC.STRIDE`, `ACC.MAX`, `ACC.MAX.FIRST`, `RESET_ACC` | Accumulate into r_acc |
| AAQ | `AGG` / `AGG.FIRST` (sum/max + post-fn + `valid_elements` mask), `AAQ`, `ACTIVATE` | Aggregate / quantize `r_acc`; `ACTIVATE` applies keyword activations (see `docs/content/specs/stage-aaq.md` for α configuration) |
| LR (×3) | `SET`, `ADD`, `SUB`, `INCR_MOD_POW2` | Scalar loop register ops |
| COND | `BEQ`, `BNE`, `BLT`, `BNZ`, `BZ`, `B`, `BR`, `BKPT` | Branches |
| BREAK | `BREAK`, `BREAK.IFEQ` | Debug breakpoints |

### Operand Types (defined in instruction_spec)

`MultStageReg`, `LrIdx`, `CrIdx`, `LcrIdx`, `AddSubSrcB`, `AaqRegIdx`, `AggMode`, `PostFn`, `ActivationFn`, `ElementsInRow`, `HorizontalStride`, `VerticalStride`, `Immediate`, `MultMaskOffsetImmediate`, `Label`

---

## How to Add a New Instruction

1. **Define it in `instruction_spec.py`:**
   ```python
   "MY_INST": {
       "operands": [
           {"name": "dest", "type": "AaqRegIdx"},
           {"name": "src",  "type": "MultStageReg", "read": "snapshot"},
       ],
       "doc": InstructionDoc(summary="...", ...),
       "execute_fn": "execute_my_inst",
   }
   ```

2. **Implement the handler in `ipu.py`:**
   ```python
   def execute_my_inst(self, *, dest: int, src: bytearray) -> None:
       # dest is resolved to an index; src is the register bytes
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
    state = _run("MY_INST aaq0 r0;;")
    assert state.regfile.get_aaq(0) == expected
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
- **`cr15`** is reserved — never use it for application data
- The build system is **Bazel** — use `bazel build/test/run`, not pip/python directly
- Data types (INT8, FP8_E4M3, FP8_E5M2) affect how register bytes are interpreted in math ops
