#!/usr/bin/env python3
"""Generate markdown documentation for IPU assembly (instructions, operand types, syntax page)."""

from __future__ import annotations

import sys
from pathlib import Path

from ipu_common.instruction_spec import (
    PSEUDO_INSTRUCTION_SPEC,
    VALID_OPERAND_TYPES,
    SLOT_COUNT,
    SLOT_METADATA,
    COMPOUND_LAYOUT_SLOT_ORDER,
)
from ipu_common.registers import create_assembler_reg_enums

# Long-form reference for each operand type string in instruction_spec (single source: VALID_OPERAND_TYPES).
OPERAND_TYPE_DETAILS: dict[str, str] = {
    "MultStageReg": (
        "Multiply-stage field in the VLIW encoding. Assembly accepts **`r0`** and **`r1`** only; "
        "the field is **two bits** wide (encoding `2` is reserved). Used as the destination of "
        "`LDR_MULT_REG` and as the **`ra`** operand of `MULT.RC.VV`."
    ),
    "LrIdx": (
        "Loop register index: resolves to **`lr0`** … **`lr15`**. Often used for addresses, strides, "
        "and control values. When marked `read: live` in the spec, the emulator reads the **current** "
        "LR value after earlier slots in the same cycle."
    ),
    "CrIdx": (
        "Constant-register index: **`cr0`** … **`cr14`**. CRs are **read-only** in assembly; the "
        "harness initializes them (e.g. base pointers, strides). `cr15` is reserved for dstructure "
        "configuration and is not a valid ISA operand. **`SET`** in the LR slot "
        "copies the full **32-bit** CR value into an LR."
    ),
    "LcrIdx": (
        "LR **or** CR index in one field: lower indices map to **`lr0`–`lr15`**, higher indices to "
        "`**cr0`–`cr14`** in the usual combined ordering used by the assembler. `cr15` is reserved "
        "and is not a valid operand. Used as **`src_b`** on **`ADD`**/**`SUB`**, as **`step`** on "
        "**`INCR_MOD_POW2`**, and as **`src`** on **`MULT.RC.VE`** (an LR's stored value selects an "
        "element from `R0`/`R1`; a CR's low byte is the scalar directly)."
    ),
    "LrIncDecImmediate": (
        "Unsigned immediate for **`INC`** / **`DEC`** in the LR slot. The bit width **W** is not "
        "hardcoded — it is derived from the LR slot union layout so the total slot width stays "
        "constant. Valid range: **`0`** to **`2^W − 1`**."
    ),
    "ElementsInRow": (
        "ACC-slot immediate: elements per row for **`ACC.STRIDE`**. Valid values: **`16`**, **`32`**, "
        "**`64`** (minimum is 16; encoded 0→16, 1→32, 2→64). See `acc_stride_enums` in `ipu_common`."
    ),
    "HorizontalStride": (
        "ACC-slot immediate: horizontal stride mode for **`ACC.STRIDE`**. Valid values: **`off`**, "
        "**`on`**, **`on_inv`** (2-bit encoded enum; **`reserved3`** is reserved). Expand padding "
        "is fixed hardware behaviour and is not programmable. See `acc_stride_enums` in `ipu_common`."
    ),
    "VerticalStride": (
        "ACC-slot immediate: **vertical stride** bit pattern for `ACC.STRIDE` (see "
        "`acc_stride_enums`)."
    ),
    "ActivationFn": (
        "AAQ-slot keyword on **`ACTIVATE`**: one of **identity**, **relu**, **relu6**, "
        "**sigmoid**, **tanh**, **gelu**, **softplus**, **elu**, **exp2** "
        "(see ``ACTIVATION_FN_NAMES`` in ``ipu_common.activations``). Emulator-only calibration (including α) "
        "and how **`POST_AAQ_REG`** (interim **512 B**) and **`STR_POST_AAQ_REG`** (store to XMEM) "
        "are described in **Building Applications** "
        "(`docs/content/building-applications.md#activations-emulator`)."
    ),
    "LrModPow2KImmediate": (
        "Four-bit immediate for **`INCR_MOD_POW2`**: encodes exponent **k** with semantic "
        "**k ∈ [1, 9]** as **(k − 1)** in the word."
    ),
    "MultMaskOffsetImmediate": (
        "Unsigned **3-bit** immediate on multiply instructions: **`mask_offset`** selects slot "
        "**`0`**–**`7`**, each a **128-bit** region of **`R_MASK`** (eight mask slots total). "
        "**`mask_shift`** remains an **`LrIdx`**."
    ),
    "BreakImmediate": "16-bit value for **`BREAK`** / breakpoint slot conditions.",
    "DstructureCrIdx": (
        "Constant-register index: **`cr0`** … **`cr15`**, selecting which CR supplies the "
        "**valid element mask** / dstructure configuration (`valid_elements`, `partition`) for "
        "`AGG.SUM`, `AGG.SUM.FIRST`, `AGG.MAX`, `AGG.MAX.FIRST`, `AAQ`, `ACTIVATE`, and the "
        "masking multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, "
        "`MULT.EE`). Unlike **`CrIdx`**, **`cr15`** is allowed here — it's the conventional "
        "dstructure register — but the operand is always mandatory; there is no implicit "
        "fallback to `cr15` when it is omitted."
    ),
    "Label": (
        "Branch target: a symbolic **`label`** or a relative offset accepted by the cond slot "
        "(e.g. `loop`, `+3`)."
    ),
}


def _operand_slug(typ: str) -> str:
    return typ.lower().replace("_", "-")


def _check_operand_docs_complete() -> None:
    missing = VALID_OPERAND_TYPES - set(OPERAND_TYPE_DETAILS)
    if missing:
        raise RuntimeError(
            "OPERAND_TYPE_DETAILS is missing entries for: " + ", ".join(sorted(missing))
        )


def generate_operand_types_md(output_path: Path) -> None:
    """Write operand-types.md (linked from the instruction reference)."""
    _check_operand_docs_complete()
    lines: list[str] = [
        "# Operand types",
        "",
        "This page is **generated** by `ipu_as.gen_docs` from `VALID_OPERAND_TYPES` in "
        "`instruction_spec.py` and the descriptions below. For encodings per instruction, see the "
        "[Instruction reference](instructions.md).",
        "",
    ]
    for typ in sorted(VALID_OPERAND_TYPES):
        slug = _operand_slug(typ)
        lines.append(f"## `{typ}` {{: #{slug} }}")
        lines.append("")
        lines.append(OPERAND_TYPE_DETAILS[typ])
        lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"Generated operand types at {output_path}")


def generate_assembly_syntax_md(output_path: Path) -> None:
    """Write assembly-syntax.md (register table from ipu_common register enums)."""
    enums = create_assembler_reg_enums()
    mult_vals = ", ".join(f"`{v}`" for v in enums.get("MultStageRegField", []))
    lr_vals = enums.get("LrRegField", [])
    cr_vals = enums.get("CrRegField", [])
    lr_span = f"`{lr_vals[0]}`–`{lr_vals[-1]}`" if lr_vals else ""
    cr_span = f"`{cr_vals[0]}`–`{cr_vals[-1]}`" if cr_vals else ""

    main = """# Assembly Language Syntax

The IPU assembler is a Python-based tool that turns assembly source into binary VLIW instructions for the emulator. Instruction names, operands, and encoding are driven by `instruction_spec.py` in `ipu_common`; the [Instruction reference](instructions.md) and [Operand types](operand-types.md) are generated from that same toolchain.

## Features

- **Jinja2 preprocessing** — templates and macros for generated assembly
- **Label resolution** — forward and backward references
- **VLIW compounds** — multiple pipeline slots per cycle (`LOAD`, `STORE`, `ACC_STORE`†, `MULT`, `ACC`, `AAQ`, `LR`×3, `COND`, `BREAK`; †simulation-only)
- **Syntax validation** — Lark-based parser with detailed errors

## Basic Syntax

Assembly is line-oriented. One **compound instruction** may contain several **slot** instructions separated by `;`, terminated by `;;`.

```asm
# Comments start with # or //
label:                          # Labels end with a colon
    LDR_MULT_REG r0 lr0 cr0;     # LOAD: load into mult stage r0
    MULT.RC.VV lr1 r0 0 lr3 cr15; # MULT: element-wise multiply
    ACC.ADD;                     # ACC: accumulate
    INC lr0 1;               # LR: bump address (increment via INC)
    BNE lr0 lr1 next;            # COND: branch
    ;;
```

By convention, **instruction mnemonics are written in upper case** in documentation and examples (e.g. `MULT.RC.VV`, `LDR_MULT_REG`). **Operand tokens** use **lower case** (`lr0`, `r0`, `cr0`). The assembler accepts **any case** for mnemonics and register tokens.

## Compound instructions

A full IPU instruction is one **compound** word: several slots execute in parallel in a single cycle.

**Pattern:**

```asm
load_inst; mult_inst; acc_inst; aaq_inst; store_inst; acc_store_inst; lr_inst_a; lr_inst_b; lr_inst_c; cond_inst; break_inst;;
```

**Rules:**

- Each slot appears a fixed number of times (see `SLOT_COUNT` in `instruction_spec.py`); unused slots are filled with that slot’s **NOP** by the assembler.
- Slot order in the binary word (MSB → LSB) is defined by the toolchain: `cond`, three `lr` sub-slots, `load`, `mult`, `acc`, `aaq`, `store`, `acc_store`, `break` (see the layout diagram on the [Instruction reference](instructions.md#compound-instruction-layout)).
- The emulator runs **BREAK** first (may halt), then resolves **LR** sub-instructions, then **LOAD**, **MULT**, **ACC**, **AAQ**, **STORE**, **ACC_STORE**, **COND** in one cycle (see `execute_vliw_cycle` in `ipu.py`). Same-cycle **load** + **store**: load resolves before store.
- The **acc_store** slot (`STR_ACC_REG`) is **simulation-only** — not implemented in real IPU hardware.

**Example (parallel slots):**

```asm
LDR_MULT_REG r0 lr0 cr0; MULT.RC.VV lr1 r0 0 lr3 cr15; ACC.ADD; INC lr0 1; BNE lr0 lr1 loop;;
```

## Register names

The mult-stage and scalar register **tokens** below are derived from `REGISTER_DEFINITIONS` in `ipu_common.registers` (same source as the assembler).

| Kind | Assembler tokens | Notes |
|------|------------------|--------|
| Mult stage | {mult_vals} | 128-byte vectors; 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | {lr_span} | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | {cr_span} | **Read-only** in assembly; initialized by the harness. **`cr0`** / **`cr1`** are often 0 and 1. See [CrIdx](operand-types.md#cridx). |

The **`mem_bypass`** vector may still exist in the **emulator regfile** for debugging, but it is **not** a valid mult-stage assembly operand.

**LR slot:** **`SET`** copies from a **`cr`** register (see [CrIdx](operand-types.md#cridx)); **`ADD`**/**`SUB`** accept a small unsigned immediate on **`src_b`** only; **`INCR_MOD_POW2`** uses a dedicated **k** immediate (see [LrModPow2KImmediate](operand-types.md#lrmodpow2kimmediate)).

## Labels and branches

```asm
start:
    SET lr0 cr0;;
loop:
    INC lr0 1;;
    BNE lr0 lr1 loop;;
    BKPT;;
```

Relative labels such as `B +5` / `B -2` are supported where the grammar accepts a **label** token (see cond-slot instructions in the reference).

""".format(
        mult_vals=mult_vals,
        lr_span=lr_span,
        cr_span=cr_span,
    )

    masking_section = """
## Masking

Multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE`) support **lane masking**: after the multiply, lanes in `MULT_RES` whose
corresponding mask bit is **1** are **active** and pass through to accumulation; lanes whose bit is
**0** are **zeroed** (deactivated). A mask of all-ones leaves every lane active (the reset default);
a mask of all-zeros deactivates every lane.

### R_MASK register

`R_MASK` is a 128-byte register packed as **8 independent 128-bit slots** (slots 0–7). Load it
from XMEM with:

```asm
LDR_MULT_MASK_REG offset, base;;
```

`offset` is an LR register holding the XMEM byte offset; `base` is a CR register holding the
base address. The full 1024-bit register is always overwritten in one cycle.

### Selecting a slot: mask_offset

Each masking multiply instruction carries a **3-bit `mask_offset` immediate** (0–7) that selects
which 128-bit slot of `R_MASK` is used for that cycle. Eight slots can co-exist in a single loaded
`R_MASK` value, so one load can serve eight different mask patterns per kernel.

### Deriving the active mask: mask_shift

Rather than using the slot mask directly, `mask_shift` selects from **7 derived masks** generated
by sequentially shifting the selected slot one bit at a time and ANDing with a partition vector
after each step. Let `M` be the 128-bit value in the selected slot. Positive and negative shifts
use **different** partition vectors:

| `mask_shift` | Derived mask |
|---|---|
| `0` | `M` (slot mask, unmodified) |
| `+1` | `(M << 1) & partition_vector` |
| `+2` | `(derived[+1] << 1) & partition_vector` |
| `+3` | `(derived[+2] << 1) & partition_vector` |
| `−1` | `(M >> 1) & inverse_partition_vector` |
| `−2` | `(derived[−1] >> 1) & inverse_partition_vector` |
| `−3` | `(derived[−2] >> 1) & inverse_partition_vector` |

`mask_shift` names an LR register. The **ctrl stage** reads that register each cycle and forwards
the value through the pipeline to the mult stage — the mult stage itself does not access LR
registers directly. LR registers are 20-bit; negative values use 20-bit two's complement. Values
outside [−3, +3] **clamp** to ±3.

### Partition vectors

Each masking multiply instruction takes a mandatory CR-index operand (`cr_idx` on `MULT.RC.VV` /
`MULT.RC.VE` / `MULT.RC.VS`; `dstructure_cr_idx` on `MULT.VE` / `MULT.EE`, since those two already
use `cr_idx` for the scalar multiplier) naming the CR register that supplies the dstructure
configuration — there is no implicit default. Both vectors are derived from that named register's
`partition` field (see [DstructureCrIdx](operand-types.md#dstructurecridx)). The valid values of P
are **{0, 2, 4, 8, 16}**.
With `partition = 0` both vectors are all-ones (no boundaries). With `partition = P` the 128 lanes
are split into `P` equal groups of `128 / P` lanes:

| Vector | 0-bit positions | Used by |
|---|---|---|
| `partition_vector` | **first** lane of each group | positive shifts (+1, +2, +3) |
| `inverse_partition_vector` | **last** lane of each group | negative shifts (−1, −2, −3) |

For `partition = 2` (groups of 64):

- `partition_vector` = `0 1`⁶³` 0 1`⁶³ — 0 at lane 0 and lane 64
- `inverse_partition_vector` = `1`⁶³` 0 1`⁶³` 0` — 0 at lane 63 and lane 127

For `partition = 4` (groups of 32):

- `partition_vector` = `0 1`³¹` 0 1`³¹` 0 1`³¹` 0 1`³¹ — 0 at lanes 0, 32, 64, 96
- `inverse_partition_vector` = `1`³¹` 0 1`³¹` 0 1`³¹` 0 1`³¹` 0` — 0 at lanes 31, 63, 95, 127

The AND at each step prevents mask bits from crossing the group boundary in either shift direction.

`CR15` remains the conventional dstructure register and is set by the host harness, but it must
still be named explicitly via the instruction's CR-index operand — there is no implicit fallback to
`CR15` (see [DstructureCrIdx](operand-types.md#dstructurecridx)).

### Example

```asm
# LR1 = XMEM byte offset of mask data; CR2 = XMEM base address
LDR_MULT_MASK_REG LR1, CR2;;

# Load a vector into R0, load cyclic data into R_CYCLIC
LDR_MULT_REG R0, LR0, CR0;;
LDR_CYCLIC_MULT_REG LR2, CR0, LR5;;

# Multiply R0 element-wise vs R_CYCLIC using slot 3, shift index in LR4, accumulate
MULT.RC.VV LR2, R0, 3, LR4, CR15; ACC.ADD;;
```

`3` is the `mask_offset` (slot 3 of `R_MASK`); `LR4` holds the `mask_shift` index; `CR15` is the
dstructure register supplying `partition` (any `CR0`–`CR15` may be named explicitly).

### Worked examples

Both examples load **all-ones** (`0xFF…FF`, 128 bits) into the selected `R_MASK` slot. At
`mask_shift = 0` every mask bit is 1, so every lane is active. Each shift step clears one
boundary bit to **0**, deactivating one lane at a time.

The tables show **active lanes** — those whose derived mask bit is **1** and therefore contribute
to accumulation.

#### Example 1 — no partitioning (`partition = 0`)

With `partition = 0` the partition vector is all-ones, so shifts slide freely across all 128 lanes.

| `mask_shift` | Active lanes (mask bit = 1) |
|:---:|---|
| `−3` | 0–124 |
| `−2` | 0–125 |
| `−1` | 0–126 |
| `0` | 0–127 *(all active)* |
| `+1` | 1–127 |
| `+2` | 2–127 |
| `+3` | 3–127 |

Positive shifts deactivate lanes from the low end (lane 0 first); negative shifts deactivate lanes
from the high end (lane 127 first). Each step removes exactly one active lane.

#### Example 2 — two partitions of 64 lanes each (`partition = 2`)

With `partition = 2` the 128 lanes are split into **group 0** (lanes 0–63) and **group 1**
(lanes 64–127). Each group evolves independently and symmetrically:

- `partition_vector` has **0** at lane 0 and lane 64 (used for positive shifts).
- `inverse_partition_vector` has **0** at lane 63 and lane 127 (used for negative shifts).

| `mask_shift` | Group 0 active (lanes 0–63) | Group 1 active (lanes 64–127) |
|:---:|---|---|
| `−3` | 0–60 | 64–124 |
| `−2` | 0–61 | 64–125 |
| `−1` | 0–62 | 64–126 |
| `0` | 0–63 *(all)* | 64–127 *(all)* |
| `+1` | 1–63 | 65–127 |
| `+2` | 2–63 | 66–127 |
| `+3` | 3–63 | 67–127 |

**Positive shifts** deactivate lanes from the **start** of each group (lane 0 and lane 64).
**Negative shifts** deactivate lanes from the **end** of each group (lane 63 and lane 127).
Each step removes exactly one lane per group — a perfectly symmetric sliding window.

"""

    jinja_tail = """## Jinja2 preprocessing

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
"""
    output_path.write_text(main + masking_section + jinja_tail)
    print(f"Generated assembly syntax page at {output_path}")


def _generate_slots_section() -> str:
    """Generate the Slots overview section from instruction_spec metadata."""
    lines: list[str] = [
        "## Slots\n",
        "A VLIW instruction word encodes one sub-instruction per slot. "
        "Slots are grouped into pipeline stages that execute sequentially within a cycle: "
        "**CTRL** (COND + LR run concurrently; LOAD address is resolved here and the data feeds MULT) "
        "→ **MULT** → **ACC** → **AAQ** → **STORE**. "
        "`ACC_STORE` and `BREAK` are simulation-only. "
        "Any omitted slot is filled with `NOP` by the assembler automatically.\n",
        "| Slot | Count | Description |",
        "|------|------:|-------------|",
    ]
    for slot in COMPOUND_LAYOUT_SLOT_ORDER:
        count = SLOT_COUNT[slot]
        meta = SLOT_METADATA.get(slot, {})
        description = meta.get("description", "")
        lines.append(f"| `{slot.upper()}` | {count} | {description} |")
    lines += [
        "",
        "### `NOP` — No Operation\n",
        "**Syntax:** `NOP`\n",
        "No operation. Every slot accepts `NOP`. "
        "The assembler fills any omitted slot with `NOP` automatically. "
        "When written explicitly in a compound instruction, `NOP` is assigned to "
        "the next unfilled slot in compound-layout order (COND → LR → LOAD → MULT → ACC → AAQ → STORE → ACC_STORE → BREAK).\n",
    ]
    return "\n".join(lines)


def generate_instruction_docs(output_path: Path) -> None:
    """Generate instruction reference documentation."""
    from ipu_as.inst import Inst
    from ipu_as.compound_inst import CompoundInst

    content = ["# IPU Assembly Instruction Reference\n"]
    content.append(
        "Per-opcode documentation is generated from `InstructionDoc` entries in "
        "`instruction_spec.py`. Operand **types** link to the shared "
        "[operand type reference](operand-types.md).\n"
    )

    content.append(_generate_slots_section())

    content.append("## Compound Instruction Layout\n")
    content.append(
        "The compound (VLIW) instruction word is shown two ways below.\n\n"
        "* **Whole-word bit layout** — every operand token in its actual bit "
        "position (MSB → LSB), coloured by the owning slot. This is what an "
        "encoded VLIW word looks like in memory.\n"
        "* **Per-slot union layout** — for each slot, the union fields packed "
        "by the layout solver, with a per-opcode grid showing which operand "
        "each opcode places in each field. Useful for seeing where cross-"
        "opcode field sharing comes from.\n"
    )
    content.append("\n### Whole-word bit layout\n")
    content.append(CompoundInst.generate_struct_layout_svg())
    content.append("\n### Per-slot union layout\n")
    content.append(CompoundInst.generate_union_layout_svg())
    content.append("\n---\n\n")

    for inst_class in Inst.get_all_instruction_classes():
        content.append(inst_class.description())
        content.append("\n---\n")

    output_path.write_text("\n".join(content))
    print(f"Generated documentation at {output_path}")


def generate_programmer_guide_md(output_path: Path) -> None:
    """Generate the programmer-facing pseudo-instruction/alias reference."""
    from ipu_as.inst import Inst

    content = ["# Programmer's Guide: Pseudo-Instructions\n"]
    content.append(
        "Pseudo-instructions are assembly mnemonics that the assembler "
        "expands into a real instruction at compile time. They never get "
        "an opcode and never appear in the binary, so they cost nothing at "
        "runtime — see [Adding a pseudo-instruction](adding-instruction.md#adding-a-pseudo-instruction) "
        "for how they're declared. This page is **generated** from "
        "`PSEUDO_INSTRUCTION_SPEC` in `instruction_spec.py`.\n"
    )

    for name, pseudo_def in PSEUDO_INSTRUCTION_SPEC.items():
        expansion = pseudo_def["expands_to"]
        lines = Inst._render_opcode_doc(name, pseudo_def["doc"], pseudo_def["operands"])
        lines.append("")
        lines.append(
            f"**Expands to:** `{expansion['instruction']}` "
            f"({expansion['slot']} slot), arguments `{', '.join(expansion['args'])}`"
        )
        content.append("\n".join(lines))
        content.append("\n---\n")

    output_path.write_text("\n".join(content))
    print(f"Generated programmer's guide at {output_path}")


def generate_all_docs(
    instructions_path: Path,
    operand_types_path: Path,
    assembly_syntax_path: Path,
    programmer_guide_path: Path,
) -> None:
    """Regenerate all MkDocs pages owned by the assembler toolchain."""
    generate_operand_types_md(operand_types_path)
    generate_assembly_syntax_md(assembly_syntax_path)
    generate_instruction_docs(instructions_path)
    generate_programmer_guide_md(programmer_guide_path)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 4:
        generate_all_docs(Path(argv[0]), Path(argv[1]), Path(argv[2]), Path(argv[3]))
    elif len(argv) == 1:
        d = Path(argv[0])
        d.mkdir(parents=True, exist_ok=True)
        generate_all_docs(
            d / "instructions.md",
            d / "operand-types.md",
            d / "assembly-syntax.md",
            d / "programmer-guide.md",
        )
    else:
        print(
            "Usage: gen_docs.py <instructions.md> <operand-types.md> <assembly-syntax.md> <programmer-guide.md>\n"
            "   or: gen_docs.py <output_directory>/",
            file=sys.stderr,
        )
        sys.exit(1)
