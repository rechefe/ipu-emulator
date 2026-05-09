# Assembly Language Syntax

The IPU assembler is a Python-based tool that turns assembly source into binary VLIW instructions for the emulator. Instruction names, operands, and encoding are driven by `instruction_spec.py` in `ipu_common`; the [Instruction Reference](instructions.md) is generated from that same specification.

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
    ldr_mult_reg r0 lr0 cr0;     # XMEM: load into mult stage R0
    mult.ee r0 lr1 lr2 lr3;      # MULT: element-wise multiply
    acc;                         # ACC: accumulate
    incr lr0 1;                  # LR: bump address
    bne lr0 lr1 next;            # COND: branch
    ;;
```

Use **lower-case** mnemonics and register names in source (e.g. `mult.ee`, `lr0`, `r0`). Generated documentation uses **upper-case** for readability (e.g. `MULT.EE`, `LR0`, `R0`).

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
ldr_mult_reg r0 lr0 cr0; mult.ee r0 lr1 lr2 lr3; acc; incr lr0 128; bne lr0 lr1 loop;;
```

## Register names

| Kind | Assembler | Notes |
|------|-----------|--------|
| Mult stage | `r0`, `r1` | 128-byte vectors; operands for `MULT.EE` are **`r0`/`r1` only** (no `mem_bypass` on that opcode). |
| Mult stage (loads) | `r0`, `r1`, `mem_bypass` | Destination of `ldr_mult_reg` only; `mem_bypass` is a load target, not an `MULT.EE` operand. |
| Cyclic / mask | `RC`, `RM` | Documented as cyclic and mask register file in the instruction reference; addressed via offsets from `LR` operands. |
| Loop / scalar | `lr0`–`lr15` | General-purpose; **read/write**. |
| Constant | `cr0`–`cr15` | **Read-only** in assembly; values come from the emulator harness or reset state. **`cr0` is 0, `cr1` is 1** where used as constants. |
| AAQ | `aaq0`–`aaq3` | Activation/quantization registers. |

**Immediates (LR slot):** decimal, hex (`0x…`), binary (`0b…`); 16-bit signed range where applicable.

## Labels and branches

```asm
start:
    set lr0 0
loop:
    incr lr0 1
    bne lr0 lr1 loop
    bkpt
```

Relative labels such as `b +5` / `b -2` are supported where the grammar accepts a **label** token (see cond-slot instructions in the reference).

## Jinja2 preprocessing

The assembler runs Jinja2 on the source when `{{`, `{%`, or `{#` appear.

```jinja2
{% set base = 0x1000 %}
set lr0 {{ base }};
ldr_mult_reg r0 lr0 cr0;;
```

## Running the assembler

**Command line (Bazel):**

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input prog.asm --output prog.bin
```

**In Bazel build rules:** use the project’s `ipu_asm` rule (see [Building Applications](building-applications.md)).

## Further reading

- [Instruction Reference](instructions.md) — auto-generated from `instruction_spec.py`
- [Adding Instructions](adding-instruction.md)
- [Building Applications](building-applications.md)
