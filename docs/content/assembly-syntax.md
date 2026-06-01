# Assembly Language Syntax

The IPU assembler is a Python-based tool that turns assembly source into binary VLIW instructions for the emulator. Instruction names, operands, and encoding are driven by `instruction_spec.py` in `ipu_common`; the [Instruction reference](instructions.md) and [Operand types](operand-types.md) are generated from that same toolchain.

## Features

- **Jinja2 preprocessing** — templates and macros for generated assembly
- **Label resolution** — forward and backward references
- **VLIW compounds** — multiple pipeline slots per cycle (`XMEM`, `MULT`, `ACC`, `AAQ`, `LR`×3, `COND`, `BREAK`)
- **Syntax validation** — Lark-based parser with detailed errors

## Basic Syntax

Assembly is line-oriented. One **compound instruction** may contain several **slot** instructions separated by `;`, terminated by `;;`.

```asm
# Comments start with # or //
label:                          # Labels end with a colon
    LDR_MULT_REG r0 lr0 cr0;     # XMEM: load into mult stage r0
    MULT.EE r0 lr1 0 lr3;        # MULT: element-wise multiply
    ACC;                         # ACC: accumulate
    ADD lr0 lr0 1;               # LR: bump address (increment via ADD)
    BNE lr0 lr1 next;            # COND: branch
    ;;
```

By convention, **instruction mnemonics are written in upper case** in documentation and examples (e.g. `MULT.EE`, `LDR_MULT_REG`). **Operand tokens** use **lower case** (`lr0`, `r0`, `cr0`). The assembler accepts **any case** for mnemonics and register tokens.

## Compound instructions

A full IPU instruction is one **compound** word: several slots execute in parallel in a single cycle.

**Pattern:**

```asm
xmem_inst; mult_inst; acc_inst; aaq_inst; lr_inst_a; lr_inst_b; lr_inst_c; cond_inst; break_inst;;
```

**Rules:**

- Each slot appears a fixed number of times (see `SLOT_COUNT` in `instruction_spec.py`); unused slots are filled with that slot’s **NOP** by the assembler.
- Slot order in the binary word is defined by the toolchain (`break`, `xmem`, `mult`, `acc`, `aaq`, then three `lr` sub-slots, then `cond`).
- The emulator runs **BREAK** first (may halt), then resolves **LR** sub-instructions, then **XMEM**, **MULT**, **ACC**, **AAQ**, **COND** in one cycle (see `execute_vliw_cycle` in `ipu.py`).

**Example (parallel slots):**

```asm
LDR_MULT_REG r0 lr0 cr0; MULT.EE r0 lr1 0 lr3; ACC; ADD lr0 lr0 1; BNE lr0 lr1 loop;;
```

## Register names

The mult-stage and scalar register **tokens** below are derived from `REGISTER_DEFINITIONS` in `ipu_common.registers` (same source as the assembler).

| Kind | Assembler tokens | Notes |
|------|------------------|--------|
| Mult stage | `r0`, `r1` | 128-byte vectors; 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | `lr0`–`lr15` | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | `cr0`–`cr15` | **Read-only** in assembly; initialized by the harness. **`cr0`** / **`cr1`** are often 0 and 1. See [CrIdx](operand-types.md#cridx). |
| AAQ | `aaq0`, `aaq1`, `aaq2`, `aaq3` | Activation / quantization registers. See [AaqRegIdx](operand-types.md#aaqregidx). |

The **`mem_bypass`** vector may still exist in the **emulator regfile** for debugging, but it is **not** a valid mult-stage assembly operand.

**LR slot:** **`SET`** copies from a **`cr`** register (see [CrIdx](operand-types.md#cridx)); **`ADD`**/**`SUB`** accept a small unsigned immediate on **`src_b`** only; **`INCR_MOD_POW2`** uses a dedicated **k** immediate (see [LrModPow2KImmediate](operand-types.md#lrmodpow2kimmediate)).

## Labels and branches

```asm
start:
    SET lr0 cr0;;
loop:
    ADD lr0 lr0 1;;
    BNE lr0 lr1 loop;;
    BKPT;;
```

Relative labels such as `B +5` / `B -2` are supported where the grammar accepts a **label** token (see cond-slot instructions in the reference).

## Jinja2 preprocessing

The assembler runs Jinja2 on the source when `{{`, `{%`, and `{#` appear.

```jinja2
{% set off_reg = 6 %}
SET lr0 cr{{ off_reg }};;
LDR_MULT_REG r0 lr0 cr0;;
```

The harness must initialize **`cr6`** (here: the memory offset) before this program runs.

## Running the assembler

**Command line (Bazel):**

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input prog.asm --output prog.bin
```

**In Bazel build rules:** use the project’s `ipu_asm` rule (see [Building Applications](building-applications.md)).

## Further reading

- [Operand types](operand-types.md) — field types used in `instruction_spec`
- [Instruction reference](instructions.md) — per-opcode documentation
- [Adding instructions](adding-instruction.md)
- [Building applications](building-applications.md)
