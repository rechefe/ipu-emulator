"""Generate the MkDocs pages for IPU assembly (E08-14).

The instruction reference, operand types, assembly syntax, programmer's guide
and instruction-format pages. ``ipu-gen docs-pages OUT_DIR`` and
``ipu-gen instruction-format-md OUT`` write them.
"""

from __future__ import annotations

from pathlib import Path

from ipu_as.inst import Inst
from ipu_gen.layout_svg import struct_layout_svg, union_layout_svg
from ipu_isa.formats import FORMATS
from ipu_isa.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    PSEUDO_INSTRUCTION_SPEC,
    SLOT_COUNT,
    SLOT_METADATA,
    VALID_OPERAND_TYPES,
)
from ipu_isa.layout import word_bits
from ipu_isa.operand_types import operand_type
from ipu_isa.registers import create_assembler_reg_enums


def _operand_slug(typ: str) -> str:
    return typ.lower().replace("_", "-")


def generate_operand_types_md(output_path: Path) -> None:
    """Write operand-types.md (linked from the instruction reference)."""
    lines: list[str] = [
        "# Operand types",
        "",
        "This page is **generated** by `ipu_gen.docs_pages` from `VALID_OPERAND_TYPES` in "
        "`instruction_spec.py` and the descriptions below. For encodings per instruction, see the "
        "[Instruction reference](instructions.md).",
        "",
    ]
    for typ in sorted(VALID_OPERAND_TYPES):
        slug = _operand_slug(typ)
        lines.append(f"## `{typ}` {{: #{slug} }}")
        lines.append("")
        lines.append(operand_type(typ).doc)
        lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"Generated operand types at {output_path}")


def generate_assembly_syntax_md(output_path: Path) -> None:
    """Write assembly-syntax.md (register table from ipu_isa register enums)."""
    enums = create_assembler_reg_enums()
    mult_vals = ", ".join(f"`{v}`" for v in enums.get("MultStageRegField", []))
    lr_vals = enums.get("LrRegField", [])
    cr_vals = enums.get("CrRegField", [])
    lr_span = f"`{lr_vals[0]}`–`{lr_vals[-1]}`" if lr_vals else ""
    cr_span = f"`{cr_vals[0]}`–`{cr_vals[-1]}`" if cr_vals else ""

    main = f"""# Assembly Language Syntax

The IPU assembler is a Python-based tool that turns assembly source into binary VLIW instructions for the emulator. Instruction names, operands, and encoding are driven by `instruction_spec.py` in `ipu_isa`; the [Instruction reference](instructions.md) and [Operand types](operand-types.md) are generated from that same toolchain.

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

The mult-stage and scalar register **tokens** below are derived from `REGISTER_DEFINITIONS` in `ipu_isa.registers` (same source as the assembler).

| Kind | Assembler tokens | Notes |
|------|------------------|--------|
| Mult stage | {mult_vals} | 128-byte vectors; 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | {lr_span} | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | {cr_span} | **Read-only** in assembly; initialized by the harness. **`cr0`** / **`cr1`** are often 0 and 1. **`cr15`** conventionally holds dstructure configuration (see [DstructureCrIdx](operand-types.md#dstructurecridx)) but is a valid `CrIdx`/`LcrIdx` operand like any other CR. See [CrIdx](operand-types.md#cridx). |

The **`mem_bypass`** vector may still exist in the **emulator regfile** for debugging, but it is **not** a valid mult-stage assembly operand.

**LR slot:** **`SET`** copies from a **`cr`** register (see [CrIdx](operand-types.md#cridx)); **`ADD`**/**`SUB`**'s **`src_b`** is register-only (see [LcrIdx](operand-types.md#lcridx)) — use **`INC`**/**`DEC`** for an immediate operand instead (see [LrIncDecImmediate](operand-types.md#lrincdecimmediate)); **`INCR_MOD_POW2`** uses a dedicated **k** immediate (see [LrModPow2KImmediate](operand-types.md#lrmodpow2kimmediate)).

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

### Local labels

A label made of decimal digits only, such as `1:`, is a **local label** (the GNU assembler's numeric labels). It may be defined any number of times. `1f` refers to the next `1:` after the current instruction word, and `1b` to the nearest `1:` at or before it, so a word may branch to its own label with `1b`. A bare `1` as a branch target is an error. Local labels let a macro or a loop body use one short label in every expansion:

```asm
{{% macro count_to(limit_cr) %}}
    SET LR0 CR0;;
1:
    INC LR0 1;;
    BNE LR0 {{{{ limit_cr }}}} 1b;;
{{% endmacro %}}
{{{{ count_to("CR2") }}}}
{{{{ count_to("CR3") }}}}
    BKPT;;
```

"""

    masking_section = """
## Masking

Multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE`) support **element masking**: after the multiply, elements in `MULT_RES` whose
corresponding mask bit is **1** are **active** and pass through to accumulation; elements whose bit is
**0** are **zeroed** (deactivated). A mask of all-ones leaves every element active (the reset default);
a mask of all-zeros deactivates every element.

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
registers directly. LR registers are 32-bit; negative values use 32-bit two's complement. Values
outside [−3, +3] **clamp** to ±3.

### Partition vectors

Each masking multiply instruction takes a mandatory CR-index operand (`cr_idx` on `MULT.RC.VV` /
`MULT.RC.VE` / `MULT.RC.VS`; `dstructure_cr_idx` on `MULT.VE` / `MULT.EE`, since those two already
use `cr_idx` for the scalar multiplier) naming the CR register that supplies the dstructure
configuration — there is no implicit default. Both vectors are derived from that named register's
`partition` field (see [DstructureCrIdx](operand-types.md#dstructurecridx)). The valid values of P
are **{0, 2, 4, 8, 16}**.
With `partition = 0` both vectors are all-ones (no boundaries). With `partition = P` the 128 elements
are split into `P` equal groups of `128 / P` elements:

| Vector | 0-bit positions | Used by |
|---|---|---|
| `partition_vector` | **first** element of each group | positive shifts (+1, +2, +3) |
| `inverse_partition_vector` | **last** element of each group | negative shifts (−1, −2, −3) |

For `partition = 2` (groups of 64):

- `partition_vector` = `0 1`⁶³` 0 1`⁶³ — 0 at element 0 and element 64
- `inverse_partition_vector` = `1`⁶³` 0 1`⁶³` 0` — 0 at element 63 and element 127

For `partition = 4` (groups of 32):

- `partition_vector` = `0 1`³¹` 0 1`³¹` 0 1`³¹` 0 1`³¹ — 0 at elements 0, 32, 64, 96
- `inverse_partition_vector` = `1`³¹` 0 1`³¹` 0 1`³¹` 0 1`³¹` 0` — 0 at elements 31, 63, 95, 127

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
`mask_shift = 0` every mask bit is 1, so every element is active. Each shift step clears one
boundary bit to **0**, deactivating one element at a time.

The tables show **active elements** — those whose derived mask bit is **1** and therefore contribute
to accumulation.

#### Example 1 — no partitioning (`partition = 0`)

With `partition = 0` the partition vector is all-ones, so shifts slide freely across all 128 elements.

| `mask_shift` | Active elements (mask bit = 1) |
|:---:|---|
| `−3` | 0–124 |
| `−2` | 0–125 |
| `−1` | 0–126 |
| `0` | 0–127 *(all active)* |
| `+1` | 1–127 |
| `+2` | 2–127 |
| `+3` | 3–127 |

Positive shifts deactivate elements from the low end (element 0 first); negative shifts deactivate elements
from the high end (element 127 first). Each step removes exactly one active element.

#### Example 2 — two partitions of 64 elements each (`partition = 2`)

With `partition = 2` the 128 elements are split into **group 0** (elements 0–63) and **group 1**
(elements 64–127). Each group evolves independently and symmetrically:

- `partition_vector` has **0** at element 0 and element 64 (used for positive shifts).
- `inverse_partition_vector` has **0** at element 63 and element 127 (used for negative shifts).

| `mask_shift` | Group 0 active (elements 0–63) | Group 1 active (elements 64–127) |
|:---:|---|---|
| `−3` | 0–60 | 64–124 |
| `−2` | 0–61 | 64–125 |
| `−1` | 0–62 | 64–126 |
| `0` | 0–63 *(all)* | 64–127 *(all)* |
| `+1` | 1–63 | 65–127 |
| `+2` | 2–63 | 66–127 |
| `+3` | 3–63 | 67–127 |

**Positive shifts** deactivate elements from the **start** of each group (element 0 and element 64).
**Negative shifts** deactivate elements from the **end** of each group (element 63 and element 127).
Each step removes exactly one element per group — a perfectly symmetric sliding window.

"""

    jinja_tail = """## Jinja2 preprocessing

The assembler runs Jinja2 on the source when `{{`, `{%`, and `{#` appear.

```jinja2
{% set off_reg = 6 %}
SET lr0 cr{{ off_reg }};;
LDR_MULT_REG r0 lr0 cr0;;
```

The harness must initialize **`cr6`** (here: the memory offset) before this program runs.

### ABI names

A kernel can name its registers by role instead of by number. The assembler takes a name → register mapping (from the kernel's `kernel.toml` `[abi]` table, or `ipu-as assemble --abi kernel.toml`) and exposes it as two read-only namespaces: `{{ cr.<name> }}` renders a `CR` register and `{{ lr.<name> }}` an `LR` register. An unknown name is a template error at its line, and a mapping whose value is not a register of its namespace is rejected before rendering. `cr` and `lr` cannot be used as parameter or define names.

```jinja2
SET {{ lr.data_ptr }} {{ cr.data_base }};;
LDR_MULT_REG R0 {{ lr.data_ptr }} CR0;;
```

With `[abi]` `data_base = "CR2"` and `data_ptr = "LR0"`, this renders `SET LR0 CR2;;`.

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


def _generate_formats_section() -> str:
    """Generate the Element formats table from ``ipu_isa.formats``."""
    lines: list[str] = [
        "## Element formats\n",
        "The element format is selected at run time. It sets the element width, "
        "and from it the XMEM row width, the register element widths and the "
        "accumulator type. An XMEM row, R0 and R1 hold 128 elements in every format. "
        "`EMULATOR` formats exist only in the emulator. A float element is "
        "1 sign bit, the exponent bits and the stored mantissa bits.\n",
        "| Format | Tag | Element bits | Exponent bits | Mantissa bits | Row bytes | Accumulator |",
        "|--------|-----|-------------:|--------------:|--------------:|----------:|-------------|",
    ]
    for fmt, info in FORMATS.items():
        lines.append(
            f"| `{fmt.name}` | `{info.tag.name}` | {info.element_bits} | {info.exp_bits} | "
            f"{info.man_bits} | {info.row_bytes} | `{info.accumulator.name}` |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_instruction_docs(output_path: Path) -> None:
    """Generate instruction reference documentation."""

    content = ["# IPU Assembly Instruction Reference\n"]
    content.append(
        "Per-opcode documentation is generated from `InstructionDoc` entries in "
        "`instruction_spec.py`. Operand **types** link to the shared "
        "[operand type reference](operand-types.md).\n"
    )

    content.append(_generate_slots_section())
    content.append(_generate_formats_section())

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
    content.append(struct_layout_svg())
    content.append("\n### Per-slot union layout\n")
    content.append(union_layout_svg())
    content.append("\n---\n\n")

    for inst_class in Inst.get_all_instruction_classes():
        content.append(inst_class.description())
        content.append("\n---\n")

    output_path.write_text("\n".join(content))
    print(f"Generated documentation at {output_path}")


def generate_instruction_format_md(output_path: Path) -> None:
    """Generate the instruction-format documentation page (with download links)."""
    width = word_bits()
    struct_svg = struct_layout_svg()
    union_svg = union_layout_svg()

    content = f"""# Instruction format

This page describes the **binary layout** of a single IPU **compound (VLIW) instruction**.
Field names, bit widths, opcode values, and union-field sharing are **generated** from
`instruction_spec.py` in `ipu_isa` (the same source the assembler and emulator use).

## Compound instruction layout ({width} bits)

### Whole-word bit layout

{struct_svg}

### Per-slot union layout

{union_svg}

## Generated artifacts

These files are produced at documentation build time (and via the `ipu-gen` CLI). They are
**not** checked into the repository.

| Artifact | Description | Download |
|----------|-------------|----------|
| SystemVerilog package | `ipu_instr_pkg` — enums, per-slot structs (opcode + operand union), per-instruction `union packed` views, nested compound type | [ipu_instr_pkg.sv](assets/ipu_instr_pkg.sv) |

### Regenerate locally

```bash
bazel build //docs:generate_instruction_format_artifacts
# outputs under bazel-bin/docs/

bazel run //src/tools/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
```

## Related documentation

- [Assembly syntax](assembly-syntax.md) — how to write compound instructions in assembly
- [Operand types](operand-types.md) — semantic types of instruction operands
- [Instruction reference](instructions.md) — per-opcode documentation
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"Generated instruction format page at {output_path}")


def generate_programmer_guide_md(output_path: Path) -> None:
    """Generate the programmer-facing pseudo-instruction/alias reference."""

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
        lines = Inst.render_opcode_doc(name, pseudo_def["doc"], pseudo_def["operands"])
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
    """Write the instruction, operand-type, assembly-syntax and programmer's guide pages."""
    generate_operand_types_md(operand_types_path)
    generate_assembly_syntax_md(assembly_syntax_path)
    generate_instruction_docs(instructions_path)
    generate_programmer_guide_md(programmer_guide_path)
