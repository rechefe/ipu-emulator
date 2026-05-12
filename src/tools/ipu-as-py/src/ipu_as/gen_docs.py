#!/usr/bin/env python3
"""Generate markdown documentation for IPU assembly (instructions, operand types, syntax page)."""

from __future__ import annotations

import sys
from pathlib import Path

from ipu_common.instruction_spec import VALID_OPERAND_TYPES
from ipu_common.registers import create_assembler_reg_enums

# Long-form reference for each operand type string in instruction_spec (single source: VALID_OPERAND_TYPES).
OPERAND_TYPE_DETAILS: dict[str, str] = {
    "MultStageReg": (
        "Multiply-stage field in the VLIW encoding. Assembly accepts **`r0`** and **`r1`** only; "
        "the field is **two bits** wide (encoding `2` is reserved). Used as the destination of "
        "`LDR_MULT_REG` and as the **`ra`** operand of `MULT.EE`."
    ),
    "LrIdx": (
        "Loop register index: resolves to **`lr0`** … **`lr15`**. Often used for addresses, strides, "
        "and control values. When marked `read: live` in the spec, the emulator reads the **current** "
        "LR value after earlier slots in the same cycle."
    ),
    "CrIdx": (
        "Constant-register index: **`cr0`** … **`cr15`**. CRs are **read-only** in assembly; the "
        "harness initializes them (e.g. base pointers, dtype in `cr15`)."
    ),
    "LcrIdx": (
        "LR **or** CR index in one field: lower indices map to **`lr0`–`lr15`**, higher indices to "
        "`**cr0`–`cr15`** in the usual combined ordering used by the assembler."
    ),
    "AddSubSrcB": (
        "Second source for **`ADD`** / **`SUB`** in the LR slot: **`lr0`–`lr15`**, **`cr0`–`cr15`**, "
        "or an **unsigned 5-bit immediate** (`0`–`31`). Encoded in **6 bits**: **`0`–`31`** use the "
        "same ordering as **`LcrIdx`**; **`32`–`63`** encode immediates as **`32 + imm`**."
    ),
    "AaqRegIdx": "AAQ register selector: **`aaq0`** … **`aaq3`**.",
    "ElementsInRow": (
        "ACC-slot immediate: encoded **elements-per-row** selector (see `acc_stride_enums` in "
        "`ipu_common`)."
    ),
    "HorizontalStride": (
        "ACC-slot immediate: **horizontal stride** bit pattern for `ACC.STRIDE` (see "
        "`acc_stride_enums`)."
    ),
    "VerticalStride": (
        "ACC-slot immediate: **vertical stride** bit pattern for `ACC.STRIDE` (see "
        "`acc_stride_enums`)."
    ),
    "AggMode": (
        "AAQ-slot immediate: aggregation mode for the `AGG` / `AGG.FIRST` instructions (sum / max family); see "
        "`acc_agg_enums`."
    ),
    "PostFn": (
        "AAQ-slot immediate: post-aggregation function selector (identity, inverse sqrt, etc.); "
        "see `acc_agg_enums`."
    ),
    "ActivationFn": (
        "AAQ-slot keyword on **`ACTIVATE`**: one of **identity**, **relu**, **relu6**, **leaky_relu**, "
        "**sigmoid**, **tanh**, **gelu**, **silu** (alias **swish**), **softplus**, **elu**, **prelu**, **exp2** "
        "(see ``ACTIVATION_FN_NAMES`` in ``ipu_common.activations``). Emulator-only calibration (including α) "
        "is covered in **Building Applications** (`docs/content/building-applications.md#activations-emulator`)."
    ),
    "Immediate": (
        "Signed **16-bit** immediate carried in the LR slot (sign-extended by the emulator for "
        "`SET`)."
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
    aaq_vals = ", ".join(f"`{v}`" for v in enums.get("AaqRegField", []))
    lr_span = f"`{lr_vals[0]}`–`{lr_vals[-1]}`" if lr_vals else ""
    cr_span = f"`{cr_vals[0]}`–`{cr_vals[-1]}`" if cr_vals else ""

    main = """# Assembly Language Syntax

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
| Mult stage | {mult_vals} | 128-byte vectors; 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | {lr_span} | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | {cr_span} | **Read-only** in assembly; initialized by the harness. **`cr0`** / **`cr1`** are often 0 and 1. See [CrIdx](operand-types.md#cridx). |
| AAQ | {aaq_vals} | Activation / quantization registers. See [AaqRegIdx](operand-types.md#aaqregidx). |

The **`mem_bypass`** vector may still exist in the **emulator regfile** for debugging, but it is **not** a valid mult-stage assembly operand.

**Immediates (LR slot):** decimal, hex (`0x…`), binary (`0b…`); see [Immediate](operand-types.md#immediate).

## Labels and branches

```asm
start:
    SET lr0 0;;
loop:
    ADD lr0 lr0 1;;
    BNE lr0 lr1 loop;;
    BKPT;;
```

Relative labels such as `B +5` / `B -2` are supported where the grammar accepts a **label** token (see cond-slot instructions in the reference).

## Jinja2 preprocessing

""".format(
        mult_vals=mult_vals,
        lr_span=lr_span,
        cr_span=cr_span,
        aaq_vals=aaq_vals,
    )

    jinja_tail = """The assembler runs Jinja2 on the source when `{{`, `{%`, and `{#` appear.

```jinja2
{% set base = 0x1000 %}
SET lr0 {{ base }};
LDR_MULT_REG r0 lr0 cr0;;
```

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
    output_path.write_text(main + jinja_tail)
    print(f"Generated assembly syntax page at {output_path}")


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

    content.append("## Compound Instruction Layout\n")
    svg_content = CompoundInst.generate_fields_svg()
    content.append(svg_content)
    content.append("\n---\n\n")

    for inst_class in Inst.get_all_instruction_classes():
        content.append(inst_class.description())
        content.append("\n---\n")

    output_path.write_text("\n".join(content))
    print(f"Generated documentation at {output_path}")


def generate_all_docs(
    instructions_path: Path,
    operand_types_path: Path,
    assembly_syntax_path: Path,
) -> None:
    """Regenerate all MkDocs pages owned by the assembler toolchain."""
    generate_operand_types_md(operand_types_path)
    generate_assembly_syntax_md(assembly_syntax_path)
    generate_instruction_docs(instructions_path)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 3:
        generate_all_docs(Path(argv[0]), Path(argv[1]), Path(argv[2]))
    elif len(argv) == 1:
        d = Path(argv[0])
        d.mkdir(parents=True, exist_ok=True)
        generate_all_docs(d / "instructions.md", d / "operand-types.md", d / "assembly-syntax.md")
    else:
        print(
            "Usage: gen_docs.py <instructions.md> <operand-types.md> <assembly-syntax.md>\n"
            "   or: gen_docs.py <output_directory>/",
            file=sys.stderr,
        )
        sys.exit(1)
