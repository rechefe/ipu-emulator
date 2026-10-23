"""Generate the MkDocs pages for IPU assembly (E08-14).

The instruction reference, operand types, assembly syntax, programmer's guide
and instruction-format pages, and the tables the element-formats page
includes. ``ipu-gen docs-pages OUT_DIR``, ``ipu-gen instruction-format-md OUT``
and ``ipu-gen element-format-tables OUT`` write them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from ipu_gen.layout_svg import struct_layout_svg, union_layout_svg
from ipu_isa.constants import (
    CR_ONE_REG_INDEX,
    CR_ONE_REG_VALUE,
    CR_ZERO_REG_INDEX,
    CR_ZERO_REG_VALUE,
    LR_CR_SCALAR_BITS,
)
from ipu_isa.cr_halves import (
    CONSTANT_LAYOUT,
    DEFAULT_DS,
    DS_LAYOUT,
    PARTITIONS,
    QF_LAYOUT,
    Field,
    MaskType,
)
from ipu_isa.faults import FaultId, instruction_faults
from ipu_isa.formats import FORMATS, ElementFormat, FormatInfo
from ipu_isa.geometry import LANES
from ipu_isa.instruction_spec import InstructionDoc, syntax
from ipu_isa.nx8 import SCALE_BIAS, Nx8Format, decode_element
from ipu_isa.registers import REGISTER_DEFINITIONS, create_assembler_reg_enums
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.timing import (
    KNOWN_DEVIATIONS,
    SLOT_EXECUTION_ORDER,
    RegisterAccess,
    View,
    contract_rule,
)
from ipu_isa.view import Instruction, Isa, Operand, isa


def _operand_slug(typ: str) -> str:
    return typ.lower().replace("_", "-")


def generate_operand_types_md(output_path: Path, view: Isa | None = None) -> None:
    """Write operand-types.md (linked from the instruction reference)."""
    view = isa() if view is None else view
    lines: list[str] = [
        "# Operand types",
        "",
        "This page is **generated** by `ipu_gen.docs_pages` from `VALID_OPERAND_TYPES` in "
        "`instruction_spec.py` and the descriptions below. For encodings per instruction, see the "
        "[Instruction reference](instructions.md).",
        "",
    ]
    for typ in sorted(view.operand_types):
        slug = _operand_slug(typ)
        lines.append(f"## `{typ}` {{: #{slug} }}")
        lines.append("")
        lines.append(view.operand_types[typ].doc)
        lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"Generated operand types at {output_path}")


def generate_assembly_syntax_md(output_path: Path) -> None:
    """Write assembly-syntax.md (register table from ipu_isa register enums)."""
    enums = create_assembler_reg_enums()
    view = isa()
    mult_vals = ", ".join(f"`{v.upper()}`" for v in enums.get("MultStageRegField", []))
    lr_vals = [v.upper() for v in enums.get("LrRegField", [])]
    cr_vals = [v.upper() for v in enums.get("CrRegField", [])]
    hardware = ", ".join(
        f"`{slot.value.upper()}`"
        + (f"×{view.slots[slot].count}" if view.slots[slot].count > 1 else "")
        for slot in view.layout.slots_with_tag(Tag.HARDWARE)
    )
    emulator = ", ".join(
        f"`{slot.value.upper()}`" for slot in view.layout.slots_with_tag(Tag.EMULATOR)
    )
    lr_span = f"`{lr_vals[0]}`–`{lr_vals[-1]}`" if lr_vals else ""
    cr_span = f"`{cr_vals[0]}`–`{cr_vals[-1]}`" if cr_vals else ""

    main = f"""# Assembly Language Syntax

The IPU assembler is a Python-based tool that turns assembly source into binary VLIW instructions for the emulator. Instruction names, operands, and encoding are driven by `instruction_spec.py` in `ipu_isa`; the [Instruction reference](instructions.md) and [Operand types](operand-types.md) are generated from that same toolchain.

## Features

- **Jinja2 preprocessing** — templates and macros for generated assembly
- **Label resolution** — forward and backward references
- **VLIW compounds** — multiple pipeline slots per cycle ({hardware}; emulator-only: {emulator})
- **Syntax validation** — Lark-based parser with detailed errors

## Basic Syntax

Assembly is line-oriented. One **compound instruction** may contain several **slot** instructions separated by `;`, terminated by `;;`.

```asm
# Comments start with # or //
label:                          # Labels end with a colon
    LD.MULT R0 LR0 CR0;           # LOAD: load into mult stage R0
    MULT.RC.VV LR1 R0 0 LR3 CR15.L CR14.L; # MULT: element-wise multiply (DS, QF halves)
    ACC.ADD;                      # ACC: accumulate
    INC LR0 1;                    # LR: bump address (increment via INC)
    BNE LR0 LR1 next;             # COND: branch
    ;;
next:
    HALT;;
```

By convention, **instruction mnemonics are written in upper case** in documentation and examples (e.g. `MULT.RC.VV`, `LD.MULT`). **Register names** are written in **upper case** (`LR0`, `R0`, `CR0`); operand placeholders such as `dest` are lower case. The assembler accepts **any case** for mnemonics and register tokens.

## Compound instructions

A full IPU instruction is one **compound** word: several slots execute in parallel in a single cycle.

**Pattern:**

<!-- asm-fragment: one placeholder per slot, not real instructions -->
```asm
load_inst; mult_inst; acc_inst; aaq_inst; aaq_exact_inst; lr_inst_a; lr_inst_b; lr_inst_c; cond_inst; break_inst;;
```

**Rules:**

- Each slot appears a fixed number of times (see `SLOT_COUNT` in `instruction_spec.py`); unused slots are filled with that slot’s **NOP** by the assembler.
- Slot order in the binary word (MSB → LSB) is defined by the toolchain: `cond`, three `lr` sub-slots, `load`, `mult`, `acc`, `aaq`, `aaq_exact`, `break` (see the layout diagram on the [Instruction reference](instructions.md#compound-instruction-layout)).
- The slots of one word execute in a fixed order, and each register read sees either the value at cycle start or the value forwarded from an earlier slot of the same word: see [Execution order and timing](#execution-order-and-timing).
- The emulator-only slots ({emulator}) are not implemented in real IPU hardware: see the [instruction format](instruction-format.md).

**Example (parallel slots):**

```asm
loop:
LD.MULT R0 LR0 CR0; MULT.RC.VV LR1 R0 0 LR3 CR15.L CR14.L; ACC.ADD; INC LR0 1; BNE LR0 LR1 loop;;
```

## Register names

The mult-stage and scalar register **tokens** below are derived from `REGISTER_DEFINITIONS` in `ipu_isa.registers` (same source as the assembler).

| Kind | Assembler tokens | Notes |
|------|------------------|--------|
| Mult stage | {mult_vals} | One XMEM row each (128 NX8 elements and the row's metadata word, or 128 FP32 elements in the FP32 mode); 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | {lr_span} | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | {cr_span} | **Read-only** in assembly; the host writes `CR2`–`CR15` before a kernel runs. **`CR0`** and **`CR1`** are the read-only constants 0 and 1 (see [Configuration registers](configuration-registers.md)). Each CR is also two 16-bit halves, `CR13.L` and `CR13.H`: the DS (data structure) and QF (quantization format) operands always name a half (see [HalfCrIdx](operand-types.md#halfcridx)), and an LR or COND source operand may read a full CR or a sign-extended half (see [LcrHalfIdx](operand-types.md#lcrhalfidx)). **`CR15.L`** conventionally holds the default DS, but any half is a valid operand. See [CrIdx](operand-types.md#cridx) and [Configuration registers](configuration-registers.md). |

The **`mem_bypass`** vector may still exist in the **emulator regfile** for debugging, but it is **not** a valid mult-stage assembly operand.

**LR slot:** **`SET`** copies from a **`CR`** register (see [CrIdx](operand-types.md#cridx)); **`ADD`**/**`SUB`**'s **`src_b`** is register-only (see [LcrIdx](operand-types.md#lcridx)) — use **`INC`**/**`DEC`** for an immediate operand instead (see [LrIncDecImmediate](operand-types.md#lrincdecimmediate)); **`INC.MODP2`** uses a dedicated **k** immediate (see [LrModPow2KImmediate](operand-types.md#lrmodpow2kimmediate)).

## Labels and branches

```asm
start:
    SET LR0 CR0;;
loop:
    INC LR0 1;;
    BNE LR0 LR1 loop;;
    HALT;;
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
    HALT;;
```

"""

    masking_section = """
## Masking

Multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE`) support **element masking**: after the multiply, elements in `MULT_RES` whose
corresponding mask bit is **1** are **active** and pass through to accumulation; elements whose bit is
**0** are deactivated and hold the pad value of the DS half's `mask_type` (`0` for `mask_type` 0,
+inf for 1, −inf for 2; 3, NaN, is an emulator error because NX8 has no NaN). A mask of all-ones
leaves every element active (the reset default); a mask of all-zeros deactivates every element.

### R_MASK register

`R_MASK` is a 128-byte register packed as **8 independent 128-bit slots** (slots 0–7). Load it
from XMEM with:

<!-- asm-fragment: operand placeholders -->
```asm
LD.MASK offset base;;
```

`offset` is an LR register holding the XMEM byte offset; `base` is a CR register holding the
base address. The full 1024-bit register is always overwritten in one cycle.
A multiply reads `R_MASK` at cycle start, so an `LD.MASK` in the same word as a masking multiply
is not seen by it: that multiply uses the previous `R_MASK`, and the next word sees the new one.

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

Each masking multiply instruction takes a mandatory DS operand, `ds`, naming the CR half that
holds the 16-bit DS (data structure) — there is no implicit default. Both vectors are derived from
that half's `num_partitions` field (see [Configuration registers](configuration-registers.md#ds-half)
and [HalfCrIdx](operand-types.md#halfcridx)). The valid values of P are **{1, 2, 4, 8}**.
With `num_partitions = 1` both vectors are all-ones (no boundaries). With `num_partitions = P` the
128 elements are split into `P` equal groups of `128 / P` elements:

| Vector | 0-bit positions | Used by |
|---|---|---|
| `partition_vector` | **first** element of each group | positive shifts (+1, +2, +3) |
| `inverse_partition_vector` | **last** element of each group | negative shifts (−1, −2, −3) |

For `num_partitions = 2` (groups of 64):

- `partition_vector` = `0 1`⁶³` 0 1`⁶³ — 0 at element 0 and element 64
- `inverse_partition_vector` = `1`⁶³` 0 1`⁶³` 0` — 0 at element 63 and element 127

For `num_partitions = 4` (groups of 32):

- `partition_vector` = `0 1`³¹` 0 1`³¹` 0 1`³¹` 0 1`³¹ — 0 at elements 0, 32, 64, 96
- `inverse_partition_vector` = `1`³¹` 0 1`³¹` 0 1`³¹` 0 1`³¹` 0` — 0 at elements 31, 63, 95, 127

The AND at each step prevents mask bits from crossing the group boundary in either shift direction.

`CR15.L` is the conventional DS half and is set by the host harness, but it must still be named
explicitly as the instruction's `ds` operand — there is no implicit fallback to `CR15.L`. The
second half operand, `qf`, names the QF (quantization format) half the NX8 elements are decoded
with; the FP32 mode ignores it.

### Example

```asm
# LR1 = XMEM byte offset of mask data; CR2 = XMEM base address
LD.MASK LR1 CR2;;

# Load a vector into R0, load cyclic data into R_CYCLIC
LD.MULT R0 LR0 CR0;;
LD.MULT.CYCLIC LR2 CR0 LR5;;

# Multiply R0 element-wise vs R_CYCLIC using slot 3, shift index in LR4, accumulate
MULT.RC.VV LR2 R0 3 LR4 CR15.L CR14.L; ACC.ADD;;
```

`3` is the `mask_offset` (slot 3 of `R_MASK`); `LR4` holds the `mask_shift` index; `CR15.L` is the
DS half supplying `num_partitions` and `CR14.L` the QF half (any half `CR0.L`–`CR15.H` may be named).

### Worked examples

Both examples load **all-ones** (`0xFF…FF`, 128 bits) into the selected `R_MASK` slot. At
`mask_shift = 0` every mask bit is 1, so every element is active. Each shift step clears one
boundary bit to **0**, deactivating one element at a time.

The tables show **active elements** — those whose derived mask bit is **1** and therefore contribute
to accumulation.

#### Example 1 — no partitioning (`num_partitions = 1`)

With `num_partitions = 1` the partition vector is all-ones, so shifts slide freely across all 128 elements.

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

#### Example 2 — two partitions of 64 elements each (`num_partitions = 2`)

With `num_partitions = 2` the 128 elements are split into **group 0** (elements 0–63) and **group 1**
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
SET LR0 CR{{ off_reg }};;
LD.MULT R0 LR0 CR0;;
```

The harness must initialize **`CR6`** (here: the memory offset) before this program runs.

### ABI names

A kernel can name its registers by role instead of by number. The assembler takes a name → register mapping (from the kernel's `kernel.toml` `[abi]` table, or `ipu-as assemble --abi kernel.toml`) and exposes it as two read-only namespaces: `{{ cr.<name> }}` renders a `CR` register and `{{ lr.<name> }}` an `LR` register. An unknown name is a template error at its line, and a mapping whose value is not a register of its namespace is rejected before rendering. `cr` and `lr` cannot be used as parameter or define names.

```jinja2
SET {{ lr.data_ptr }} {{ cr.data_base }};;
LD.MULT R0 {{ lr.data_ptr }} CR0;;
```

With `[abi]` `data_base = "CR2"` and `data_ptr = "LR0"`, this renders `SET LR0 CR2;;`.

## Running the assembler

**Command line (Bazel):**

```bash
bazel run //src/ipu-as:ipu-as -- assemble --input prog.asm --output prog.bin
```

**In Bazel build rules:** a kernel's `:assemble_<name>` target writes its program (see [Write a kernel](../../how-to/write-a-kernel.md)); `assemble_asm` in `src/ipu-as/defs.bzl` assembles any other source.

## Further reading

- [Operand types](operand-types.md) — field types used in `instruction_spec`
- [Instruction reference](instructions.md) — per-opcode documentation
- [Adding instructions](../../how-to/add-an-instruction.md)
- [Write a kernel](../../how-to/write-a-kernel.md)
"""
    output_path.write_text(main + timing_section() + masking_section + jinja_tail)
    print(f"Generated assembly syntax page at {output_path}")


def _register(name: str) -> str:
    return f"`{name.upper()}`"


def _access(access: RegisterAccess) -> str:
    """``LR`` via `dest_slot`, ``MULT_RES``."""
    text = _register(access.register)
    return f"{text} via `{access.via_operand}`" if access.via_operand else text


def _accesses(accesses: Iterable[RegisterAccess]) -> str:
    return ", ".join(_access(access) for access in accesses) or "—"


def timing_section(view: Isa | None = None) -> str:
    """The execution order, the timing contract, every instruction's register
    reads and writes, and the known deviations, from ``ipu_isa.timing`` (E03-14)."""
    order = " → ".join(f"**{slot.value.upper()}**" for slot in SLOT_EXECUTION_ORDER)
    lines = [
        "",
        "## Execution order and timing",
        "",
        "This section is **generated** from `ipu_isa.timing` and the register accesses each "
        "instruction declares in `instruction_spec.py`.",
        "",
        f"The slots of one instruction word execute in this order: {order}. "
        "**BREAK** is evaluated before any slot writes (it may halt the word); the **LR** "
        "sub-slots all read before any of them writes.",
        "",
        "A register read sees one of two values: the value at **cycle start** (when the word "
        "began), or the **forwarded** value (after the writes of earlier slots of the same "
        "word). The timing contract is four rules:",
        "",
        "1. The Control stage (**LR**, **COND**) reads LRs at cycle start; every other stage "
        "reads LRs after the LR writes of the same word.",
        "2. **MULT** reads `R0`, `R1`, `R_CYCLIC`, their row metadata and `R_MASK` at cycle start.",
        "3. `MULT_RES` is forwarded to **ACC** (including AGG and RESHAPE); `R_ACC` is forwarded "
        "to **AAQ** and **AAQ_EXACT**.",
        "4. LR sub-slots read cycle-start values.",
        "",
        "Any other read is forwarded when every slot that writes the register runs earlier in "
        "the word, and at cycle start otherwise. CRs are never written during a run, so CR "
        "reads are at cycle start.",
        "",
        f"The [timing model]({TIMING_PAGE}) explains the rules, with an example of each.",
        "",
        "### Register reads and writes",
        "",
        "`R` is the `R0`/`R1` pair. An operand of type `LcrIdx` reads an LR or a CR; one of type "
        "`LcrHalfIdx` an LR, a CR or a CR half; one of type `HalfCrIdx` a CR half.",
        "",
        "| Instruction | Slot | Reads at cycle start | Reads forwarded | Writes |",
        "|-------------|------|----------------------|-----------------|--------|",
    ]
    view = isa() if view is None else view
    for slot in SLOT_EXECUTION_ORDER:
        for instruction in view.instructions(slot):
            if instruction.mnemonic == "NOP":
                continue
            reads = instruction.reads()
            cycle_start = [a for a in reads if a.view is View.CYCLE_START]
            forwarded = [a for a in reads if a.view is View.FORWARDED]
            lines.append(
                f"| `{instruction.mnemonic}` | `{slot.value.upper()}` | {_accesses(cycle_start)} "
                f"| {_accesses(forwarded)} | {_accesses(instruction.writes())} |"
            )
    lines += ["", "### Known deviations", ""]
    if not KNOWN_DEVIATIONS:
        lines += ["None: every read in the emulator follows the contract.", ""]
        return "\n".join(lines)
    lines += [
        "Where the emulator differs from the contract today. Each row is removed by the "
        "issue named in the last column.",
        "",
        "| Instruction | Read | Today | Contract | Fixed by |",
        "|-------------|------|-------|----------|----------|",
    ]
    for deviation in KNOWN_DEVIATIONS:
        access = RegisterAccess(deviation.register, deviation.via_operand)
        lines.append(
            f"| `{deviation.mnemonic}` | {_access(access)} | {_view(deviation.today)} "
            f"| {_view(deviation.contract)} | {deviation.fixed_by} |"
        )
    lines.append("")
    return "\n".join(lines)


#: The explanation page of the timing contract, from a page under reference/isa/ (E09-18).
TIMING_PAGE = "../../explanation/timing-model.md"


def read_point(access: RegisterAccess) -> str:
    """The read point of one read, as the timing model page names it.

    ``cycle start``, ``after LR writes`` (a forwarded LR) or ``same-cycle
    forward`` (any other forwarded register).
    """
    if access.view is View.CYCLE_START:
        return "cycle start"
    return "after LR writes" if access.register == "lr" else "same-cycle forward"


def rule_anchor(slot: Slot, register: str) -> str:
    """The timing model page's heading for the rule that fixes a read's view:
    ``rule-1`` to ``rule-4``, or ``slot-order``."""
    rule = contract_rule(slot, register)
    return "slot-order" if rule is None else f"rule-{rule}"


def _rule_label(anchor: str) -> str:
    return "slot order" if anchor == "slot-order" else anchor.replace("-", " ").capitalize()


def timing_contract_table(view: Isa | None = None, page: str = "") -> str:
    """Every register read of the ISA, by slot, with its read point and the rule behind it.

    One row per slot, register and read point; the operands column names the
    operands that select the register (``implicit`` for a read no operand
    names). The rows come from the declared reads (``Instruction.reads()``),
    which ``load_isa`` checks against the contract. The timing model page
    includes this table (E09-18).

    Args:
        view: The ISA; default ``isa()``.
        page: The page the rule links point at ("" for the page that includes the table).

    Returns:
        The Markdown table.
    """
    view = isa() if view is None else view
    rows: dict[tuple[Slot, str, str], list[str]] = {}
    for slot in SLOT_EXECUTION_ORDER:
        for instruction in view.instructions(slot):
            for access in instruction.reads():
                names = rows.setdefault((slot, access.register, read_point(access)), [])
                name = f"`{access.via_operand}`" if access.via_operand else "implicit"
                if name not in names:
                    names.append(name)
    lines = [
        "| Slot | Register | Selected by | Read point | Rule |",
        "|------|----------|-------------|------------|------|",
    ]
    for (slot, register, point), names in rows.items():
        anchor = rule_anchor(slot, register)
        lines.append(
            f"| `{slot.value.upper()}` | {_register(register)} | {', '.join(names)} "
            f"| {point} | [{_rule_label(anchor)}]({page}#{anchor}) |"
        )
    return "\n".join(lines) + "\n"


def generate_timing_contract_table(output_path: Path, view: Isa | None = None) -> None:
    """Write the timing model page's table (:func:`timing_contract_table`)."""
    output_path.write_text(timing_contract_table(view))
    print(f"Generated the timing contract table at {output_path}")


def _read_target(access: RegisterAccess) -> str:
    """```offset` (`LR`)`` for an operand, ```R_CYCLIC``` for an implicit read."""
    register = _register(access.register)
    return f"`{access.via_operand}` ({register})" if access.via_operand else register


def register_reads_lines(instruction: Instruction, page: str = TIMING_PAGE) -> list[str]:
    """The instruction reference's "Register reads" line: each register the
    instruction reads, with its read point linked to the rule on the timing
    model page. Empty for an instruction that reads no register."""
    reads = [
        f"{_read_target(access)}: "
        f"[{read_point(access)}]({page}#{rule_anchor(instruction.slot, access.register)})"
        for access in instruction.reads()
    ]
    if not reads:
        return []
    return ["", "**Register reads:** " + "; ".join(reads) + "."]


def _view(view: View) -> str:
    return "cycle start" if view is View.CYCLE_START else "forwarded"


def _generate_slots_section(view: Isa) -> str:
    """Generate the Slots overview section from the slot registry."""
    lines: list[str] = [
        "## Slots\n",
        "A VLIW instruction word encodes one sub-instruction per slot. "
        "Slots are grouped into pipeline stages that execute sequentially within a cycle: "
        "**CTRL** (COND + LR run concurrently; LOAD address is resolved here and the data feeds MULT) "
        "→ **MULT** → **ACC** → **AAQ**, which activates, quantizes and stores its own row (ADR 0007). "
        f"{_emulator_only_sentence(view)} "
        "Any omitted slot is filled with `NOP` by the assembler automatically.\n",
        "| Slot | Count | Tag | Description |",
        "|------|------:|-----|-------------|",
    ]
    for slot in view.layout.layout_order:
        info = view.slots[slot]
        lines.append(
            f"| `{slot.value.upper()}` | {info.count} | `{info.tag.name}` | {info.description} |"
        )
    lines += [
        "",
        "### `NOP` — No Operation\n",
        "**Syntax:** `NOP`\n",
        "**Tag:** `HARDWARE` in the hardware slots, `EMULATOR` in the emulator-only slots.\n",
        f"{_faults_line(())}\n",
        "No operation. Every slot accepts `NOP`. "
        "The assembler fills any omitted slot with `NOP` automatically. "
        "When written explicitly in a compound instruction, `NOP` is assigned to "
        "the next unfilled slot in compound-layout order (COND → LR → LOAD → MULT → ACC → AAQ → AAQ_EXACT → BREAK).\n",
    ]
    return "\n".join(lines)


def _slot_names(view: Isa, tag: Tag) -> str:
    """``tag``'s slots, MSB first, with their copy counts, for example ``"`LR`×3, `LOAD`"``."""
    return ", ".join(
        f"`{slot.value.upper()}`"
        + (f"×{view.slots[slot].count}" if view.slots[slot].count > 1 else "")
        for slot in view.layout.slots_with_tag(tag)
    )


def _emulator_only_sentence(view: Isa) -> str:
    return f"{_slot_names(view, Tag.EMULATOR)} are emulator-only: not in the hardware word."


def _word_layout_sections(view: Isa, level: str) -> str:
    """The hardware word, then the emulator word, each with its diagrams.

    Args:
        view: The ISA.
        level: The heading level of the two sections (``"##"`` or ``"###"``).

    Returns:
        Markdown: the hardware word first, as the SV package describes it.
    """
    hardware = view.layout.slots_with_tag(Tag.HARDWARE)
    emulator = view.layout.slots_with_tag(Tag.EMULATOR)
    hw_bits = view.layout.word_bits(Tag.HARDWARE)
    all_bits = view.layout.word_bits()
    sub = level + "#"
    return f"""{level} Hardware word ({hw_bits} bits)

The part of the instruction word the IPU decodes: the hardware slots
({_slot_names(view, Tag.HARDWARE)}). The SystemVerilog package describes this
word (`IPU_COMPOUND_INST_WIDTH = {hw_bits}`).

{sub} Whole-word bit layout

{struct_layout_svg(view, hardware, title="Hardware word")}

{sub} Per-slot union layout

{union_layout_svg(view, hardware)}

{level} Emulator word ({all_bits} bits)

The word the emulator decodes and `.bin` files hold: the hardware word followed by
the emulator-only slots ({_slot_names(view, Tag.EMULATOR)}) below its bit 0. Its top
{hw_bits} bits are the hardware word.

{sub} Emulator-only slots: union layout

{union_layout_svg(view, emulator)}

{sub} Whole-word bit layout

{struct_layout_svg(view, title="Emulator word")}
"""


def _generate_formats_section() -> str:
    """The instruction reference's pointer to the element-format page (E09-19)."""
    return (
        "## Element formats\n\n"
        "The element format is selected at run time and sets the element width. "
        "[Element formats](../../explanation/element-formats.md) lists the formats and "
        "how each encodes values.\n"
    )


# ===========================================================================
# Element-format tables, included by explanation/element-formats.md (E09-19)
# ===========================================================================

#: The file the explanation page includes its tables from, and its sections.
ELEMENT_FORMAT_TABLES = "element-format-tables.md"
ELEMENT_FORMAT_SECTIONS = ("formats", "encodings", "cr-scalars")


def _value_text(value: float) -> str:
    """An exact, short text for a value of an 8-bit format."""
    if value == int(value) and abs(value) < 1 << 20:
        return str(int(value))
    mantissa, exponent = math.frexp(value)  # value = mantissa * 2**exponent, 0.5 <= mantissa < 1
    mantissa, exponent = 2 * mantissa, exponent - 1
    if -10 <= exponent < 20:
        return repr(value)
    power = f"2<sup>{exponent}</sup>"
    return power if mantissa == 1 else f"{mantissa!r} × {power}"


def _section(name: str, lines: list[str]) -> list[str]:
    return [f"<!-- --8<-- [start:{name}] -->", *lines, f"<!-- --8<-- [end:{name}] -->", ""]


def _formats_table(formats: Mapping[ElementFormat, FormatInfo]) -> list[str]:
    lines = [
        "| Format | Tag | Element bits | Metadata bits | Row bytes | Accumulator |",
        "|--------|-----|-------------:|--------------:|----------:|-------------|",
    ]
    for fmt, info in formats.items():
        lines.append(
            f"| `{fmt.name}` | `{info.tag.name}` | {info.element_bits} | {info.metadata_bits} | "
            f"{info.row_bytes} | `{info.accumulator.name}` |"
        )
    return lines


#: The NX8 element types the encodings table lists: signed, 8 bits, each ``fe``.
NX8_TABLE_TYPES = tuple(Nx8Format(sign=1, exp_format=fe, num_bits=8) for fe in range(1, 8))


def _encodings_table(formats: Mapping[ElementFormat, FormatInfo]) -> list[str]:
    """One row per signed 8-bit NX8 type (QF ``fe`` 1-7) at scale 127 (2^0).

    A row's metadata ``scale`` multiplies every value by 2^(scale - 127). The
    table is listed only when the formats include NX8.
    """
    lines = [
        "| Type | Exponent bits | Mantissa bits | Largest | Smallest normal | Smallest nonzero |",
        "|------|--------------:|--------------:|--------:|----------------:|-----------------:|",
    ]
    if ElementFormat.NX8 not in formats:
        return lines
    for nx8 in NX8_TABLE_TYPES:
        fe, fm = nx8.exp_bits(), nx8.man_bits()
        clamp = (1 << fe) - 1
        codes = range(1 << (fe + fm))  # the positive codes (sign 0), right-aligned
        values = [
            (decode_element(c << (8 - nx8.num_bits), nx8, SCALE_BIAS), c >> fm) for c in codes
        ]
        normal = [v for v, e in values if e < clamp]
        nonzero = [v for v, _ in values if v > 0]
        lines.append(
            f"| `E{fe}` | {fe} | {fm} | {_value_text(max(nonzero))} | "
            f"{_value_text(min(normal)) if normal else 'none'} | {_value_text(min(nonzero))} |"
        )
    return lines


#: How a MULT constant operand reads its CR in each mode (ADR 0007).
_CR_SCALAR_DECODE = {
    ElementFormat.NX8: (
        "`[23:0]` (bits 31:24 ignored)",
        "one NX8 element with its own scale and type (the constant layout); "
        "`CR0` is 0.0 and `CR1` 1.0",
    ),
    ElementFormat.FP32: ("`[31:0]`", "an IEEE binary32 value; `CR0` is 0.0 and `CR1` 1.0"),
}


def _cr_scalar_table(formats: Mapping[ElementFormat, FormatInfo]) -> list[str]:
    """Which CR bits a MULT constant operand reads, and how they are decoded, per mode."""
    lines = ["| Format | CR bits | Decoded as |", "|--------|---------|------------|"]
    for fmt in formats:
        bits, decoded = _CR_SCALAR_DECODE[fmt]
        lines.append(f"| `{fmt.name}` | {bits} | {decoded} |")
    return lines


def render_element_format_tables(formats: Mapping[ElementFormat, FormatInfo] = FORMATS) -> str:
    """The tables ``explanation/element-formats.md`` includes, one snippet section each.

    Args:
        formats: The formats to list; default every format of ``ipu_isa.formats``.

    Returns:
        The file's text: sections ``ELEMENT_FORMAT_SECTIONS``, marked for
        ``pymdownx.snippets``.
    """
    lines = [
        "<!-- Generated by ipu_gen.docs_pages.render_element_format_tables (E09-19). -->",
        "",
        *_section("formats", _formats_table(formats)),
        *_section("encodings", _encodings_table(formats)),
        *_section("cr-scalars", _cr_scalar_table(formats)),
    ]
    return "\n".join(lines)


def generate_element_format_tables_md(output_path: Path) -> None:
    """Write ``render_element_format_tables()`` to ``output_path``."""
    output_path.write_text(render_element_format_tables())
    print(f"Generated element-format tables at {output_path}")


# ===========================================================================
# Instruction reference (Markdown), for the generated docs
# ===========================================================================


def _operand_type_md_link(typ: str) -> str:
    slug = typ.lower().replace("_", "-")
    return f"[`{typ}`](operand-types.md#{slug})"


def _md_table_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


#: The fault list, from a page under reference/isa/.
FAULTS_PAGE = "../../requirements/faults.md#fault-list"


def _faults_line(faults: Iterable[FaultId], via: str | None = None) -> str:
    """The faults an instruction can raise. Word-level faults (F-2 by sequential
    flow, F-8, F-9) belong to no instruction and are not listed."""
    labels = ", ".join(fault.label for fault in faults) or "none"
    through = f", through `{via}`" if via else ""
    return f"**Faults:** {labels}{through} ([fault list]({FAULTS_PAGE}))."


def render_slot_docs(
    slot: str,
    view: Isa | None = None,
    extra: Callable[[Instruction], Sequence[str]] | None = None,
) -> str:
    """The reference section of one slot, in Markdown.

    The heading and intro come from the slot registry (``SlotInfo.doc_heading``,
    ``doc_intro``), then one entry per documented instruction.

    Args:
        slot: The slot name, e.g. ``"lr"``.
        view: The ISA; default ``isa()``.
        extra: Lines appended to each instruction's entry (the register read
            points, E09-18); none by default.

    Returns:
        The section text.

    Raises:
        ValueError: ``slot`` is no slot.
    """
    view = isa() if view is None else view
    info = view.slots[Slot(slot)]
    heading, intro = info.doc_heading, info.doc_intro
    lines = [f"## {heading}", ""]
    if info.tag is Tag.EMULATOR:
        lines.append(
            "> **Emulator-only slot** — not implemented in real IPU hardware: not in the "
            "hardware word or the SV package."
        )
        lines.append("")
    if intro.strip():
        lines.append(intro.strip())
        lines.append("")

    for instruction in view.instructions(info.slot):
        if instruction.mnemonic == "NOP":
            continue  # documented once globally in the Slots section
        if instruction.doc is None:
            continue
        lines.extend(
            render_opcode_doc(
                instruction.mnemonic,
                instruction.doc,
                instruction.operands,
                instruction.tag,
                extra_lines=[_faults_line(instruction_faults(view, instruction))],
            )
        )
        if extra is not None:
            lines.extend(extra(instruction))
        lines.append("")

    return "\n".join(lines).rstrip()


#: How the instruction reference describes each tag (E03-23).
TAG_TEXT = {
    Tag.HARDWARE: "implemented by the IPU",
    Tag.EMULATOR: "emulator only; not in the hardware word or the SV package",
    Tag.ASSEMBLER: "expanded by the assembler; no opcode",
}


def render_opcode_doc(
    opcode: str,
    doc: InstructionDoc,
    spec_operands: Sequence[Operand],
    tag: Tag | None = None,
    *,
    extra_lines: Sequence[str] = (),
) -> list[str]:
    """Markdown documentation of one opcode or pseudo-instruction.

    Args:
        opcode: The mnemonic, as written in assembly.
        doc: Its ``InstructionDoc``.
        spec_operands: Its operands, in assembly order.
        tag: Its tag, shown under the syntax; omitted when ``None``.
        extra_lines: Lines shown after the tag: its register reads and its faults.

    Returns:
        The Markdown lines: heading, syntax, tag, operands, and the doc's prose.
    """
    lines = [f"### `{opcode}` — {doc.title}", ""]
    names = [operand.name for operand in spec_operands]
    lines.append(f"**Syntax:** `{syntax(opcode, names)}`")
    lines.append("")
    if tag is not None:
        lines.append(f"**Tag:** `{tag.name}` ({TAG_TEXT[tag]})")
        lines.append("")
    for line in extra_lines:
        lines.append(line)
        lines.append("")

    if spec_operands:
        lines.append(
            "**Operands:** *(the **Type** column links to the "
            "[operand type reference](operand-types.md))*"
        )
        lines.append("")
        lines.append("| Name | Type | Details |")
        lines.append("|------|------|---------|")
        for i, sop in enumerate(spec_operands):
            name = sop.name
            typ = sop.type.name
            link = _operand_type_md_link(typ)
            detail = doc.operands[i] if i < len(doc.operands) else "—"
            lines.append(f"| `{name}` | {link} | {_md_table_cell(detail)} |")
        lines.append("")
    elif doc.operands:
        lines.append("**Operands:**")
        lines.extend([f"- {operand}" for operand in doc.operands])
        lines.append("")

    lines.append("**General description:**")
    lines.append(doc.summary)
    lines.append("")

    if doc.notes:
        lines.append(f"> {doc.notes}")
        lines.append("")

    if doc.operation:
        lines.append("**Pseudo code:**")
        lines.append(f"`{doc.operation}`")
        lines.append("")

    if doc.example:
        lines.append("**Example of usage:**")
        lines.append("```asm")
        lines.append(doc.example)
        lines.append("```")

    return lines


def generate_instruction_docs(output_path: Path, view: Isa | None = None) -> None:
    """Generate instruction reference documentation from ``view`` (default: ``isa()``)."""

    content = ["# IPU Assembly Instruction Reference\n"]
    content.append(
        "Per-opcode documentation is generated from `InstructionDoc` entries in "
        "`instruction_spec.py`. Operand **types** link to the shared "
        "[operand type reference](operand-types.md).\n"
    )

    view = isa() if view is None else view
    content.append(_generate_slots_section(view))
    content.append(_generate_formats_section())

    content.append("## Compound Instruction Layout\n")
    content.append(
        "The compound (VLIW) instruction word is shown two ways below, first as the "
        "hardware word, then as the emulator word.\n\n"
        "* **Whole-word bit layout** — every operand token in its actual bit "
        "position (MSB → LSB), coloured by the owning slot. This is what an "
        "encoded VLIW word looks like in memory.\n"
        "* **Per-slot union layout** — for each slot, the union fields packed "
        "by the layout solver, with a per-opcode grid showing which operand "
        "each opcode places in each field. Useful for seeing where cross-"
        "opcode field sharing comes from.\n"
    )
    content.append(_word_layout_sections(view, "###"))
    content.append("\n---\n\n")

    for slot in view.layout.layout_order:
        content.append(render_slot_docs(slot.value, view, register_reads_lines))
        content.append("\n---\n")

    output_path.write_text("\n".join(content))
    print(f"Generated documentation at {output_path}")


def generate_instruction_format_md(output_path: Path, view: Isa | None = None) -> None:
    """Generate the instruction-format documentation page (with download links).

    The hardware word comes first, as the SV package describes it; the emulator
    word follows as the hardware word plus the emulator-only slots (E03-23).
    """
    view = isa() if view is None else view
    content = f"""# Instruction format

This page describes the **binary layout** of a single IPU **compound (VLIW) instruction**.
Field names, bit widths, opcode values, and union-field sharing are **generated** from
`instruction_spec.py` in `ipu_isa` (the same source the assembler and emulator use).

{_word_layout_sections(view, "##")}
## Generated artifacts

These files are produced at documentation build time (and via the `ipu-gen` CLI). They are
**not** checked into the repository.

| Artifact | Description | Download |
|----------|-------------|----------|
| SystemVerilog package | `ipu_instr_pkg` — enums, per-slot structs (opcode + operand union), per-instruction `union packed` views, the hardware word as the compound type | [ipu_instr_pkg.sv](../../assets/ipu_instr_pkg.sv) |

### Regenerate locally

```bash
bazel build //docs:generate_instruction_format_artifacts
# outputs under bazel-bin/docs/

bazel run //src/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
```

## Related documentation

- [Assembly syntax](assembly-syntax.md) — how to write compound instructions in assembly
- [Operand types](operand-types.md) — semantic types of instruction operands
- [Instruction reference](instructions.md) — per-opcode documentation
"""
    output_path.write_text(content, encoding="utf-8")
    print(f"Generated instruction format page at {output_path}")


def generate_programmer_guide_md(output_path: Path, view: Isa | None = None) -> None:
    """Generate the programmer-facing pseudo-instruction/alias reference."""

    content = ["# Programmer's Guide: Pseudo-Instructions\n"]
    content.append(
        "Pseudo-instructions are assembly mnemonics that the assembler "
        "expands into a real instruction at compile time. They never get "
        "an opcode and never appear in the binary, so they cost nothing at "
        "runtime — see [Adding a pseudo-instruction](../../how-to/add-an-instruction.md#pseudo-instructions) "
        "for how they're declared. This page is **generated** from "
        "`PSEUDO_INSTRUCTION_SPEC` in `instruction_spec.py`.\n"
    )

    view = isa() if view is None else view
    for pseudo in view.pseudos():
        target = pseudo.expands_to
        real = view.instruction(target.slot, target.mnemonic)
        lines = render_opcode_doc(
            pseudo.mnemonic,
            pseudo.doc,
            pseudo.operands,
            pseudo.tag,
            extra_lines=[_faults_line(instruction_faults(view, real), via=target.mnemonic)],
        )
        lines.append("")
        lines.append(
            f"**Expands to:** `{target.mnemonic} {' '.join(pseudo.args)}` "
            f"({target.slot.value} slot)"
        )
        content.append("\n".join(lines))
        content.append("\n---\n")

    output_path.write_text("\n".join(content))
    print(f"Generated programmer's guide at {output_path}")


#: The reference pages ``generate_all_docs`` writes, by file name.
REFERENCE_PAGES = (
    "instructions.md",
    "operand-types.md",
    "assembly-syntax.md",
    "programmer-guide.md",
    "registers.md",
    "configuration-registers.md",
)


def generate_registers_md(output_path: Path) -> None:
    """Write the register reference from ``REGISTER_DEFINITIONS`` (E09-14)."""
    lines = [
        "# Registers",
        "",
        "This page is **generated** from `REGISTER_DEFINITIONS` in `ipu_isa.registers`. "
        "A register with assembly names is an instruction operand; any other register is "
        "named only in the instruction reference, or is emulator storage for debugging.",
        "",
        "| Register | Kind | Count | Bytes each | Vector | Assembly names |",
        "|----------|------|------:|-----------:|--------|----------------|",
    ]
    for name, meta in REGISTER_DEFINITIONS.items():
        values = meta.get("assembler_values", [])
        names = (
            f"`{values[0].upper()}`–`{values[-1].upper()}`"
            if len(values) > 2
            else ", ".join(f"`{v.upper()}`" for v in values)
        )
        lines.append(
            f"| `{name.upper()}` | {meta['kind'].name} | {meta.get('count', 1)} "
            f"| {meta['size_bytes']} | {'yes' if meta['vector'] else 'no'} | {names or '—'} |"
        )
    lines += [
        "",
        f"`LR` and `CR` hold {LR_CR_SCALAR_BITS}-bit scalars. `CR0` and `CR1` are read-only "
        "constants; see [Configuration registers](configuration-registers.md).",
        "",
    ]
    output_path.write_text("\n".join(lines))
    print(f"Generated register reference at {output_path}")


#: What each DS ``mask_type`` writes into a masked-out MULT_RES element.
_MASK_FILL = {
    MaskType.ZERO: "`0`",
    MaskType.POS_INF: "`+inf`",
    MaskType.NEG_INF: "`-inf`",
    MaskType.NAN: "refused (`CrContentError`)",
}


def _bits(shift: int, width: int) -> str:
    return f"`[{shift + width - 1}:{shift}]`"


def _field_rows(layout: Mapping[str, Field], notes: Mapping[str, str]) -> str:
    return "\n".join(
        f"| `{name}` | {_bits(field.lsb, field.width)} | {notes[name]} |"
        for name, field in layout.items()
    )


def _half_users(view: Isa, operand: str) -> str:
    users = sorted(
        {
            instruction.mnemonic
            for slot in view.layout.layout_order
            for instruction in view.instructions(slot)
            if any(
                op.name == operand and op.type.name == "HalfCrIdx" for op in instruction.operands
            )
        }
    )
    return ", ".join(f"`{m}`" for m in users)


_DS_NOTES = {
    "valid_elements": f"active elements per partition (`0`–`{LANES}`)",
    "num_partitions": ", ".join(f"`{i}`: {p}" for i, p in enumerate(PARTITIONS))
    + " partitions (groups of elements)",
    "mask_type": "the fill of a masked-out element (below)",
    "valid_partitions": "how many partitions are active (`1`–`num_partitions`)",
}
_QF_NOTES = {
    "num_bits": "element width (`1`–`8`)",
    "exp_format": "exponent bits `fe` (`1`–`7`)",
    "sign": "`1`: signed",
    "dynamic": "`1`: dynamic exponent (signed 8-bit only)",
}
_CONSTANT_NOTES = {
    "value": "the NX8 element code",
    "scale": "its scale (2<sup>scale − 127</sup>)",
    "exp_format": "its exponent bits `fe`",
    "sign": "`1`: signed",
}


def generate_configuration_registers_md(output_path: Path, view: Isa | None = None) -> None:
    """Write the configuration-register reference from ``ipu_isa.cr_halves`` (ADR 0007)."""
    view = isa() if view is None else view
    mask_rows = "\n".join(
        f"| `{mode.name}` | `{mode.value}` | {_MASK_FILL[mode]} |" for mode in MaskType
    )
    content = f"""# Configuration registers

This page is **generated** from `ipu_isa.constants` and `ipu_isa.cr_halves`. The host
writes the CRs before a kernel runs; a kernel only reads them.

## The CRs

| Register | Value | Notes |
|----------|-------|-------|
| `CR{CR_ZERO_REG_INDEX}` | `{CR_ZERO_REG_VALUE}` | Read-only constant. As a MULT constant it reads 0.0. |
| `CR{CR_ONE_REG_INDEX}` | `{CR_ONE_REG_VALUE}` | Read-only constant. As a MULT constant it reads 1.0 in both modes (see [CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `CR2`–`CR15` | set by the host | Base addresses, strides, loop bounds, MULT constants, DS and QF halves. By convention `CR15.L` holds the default DS, but any CR half is a valid operand. |

Each LR and CR holds a {LR_CR_SCALAR_BITS}-bit scalar. A negative constant is stored as its
{LR_CR_SCALAR_BITS}-bit two's complement (`value & LR_CR_SCALAR_VALUE_MASK`). A CR also holds two
16-bit halves, `CRn.L` (bits 15:0) and `CRn.H` (bits 31:16); an LR or COND operand that names a
half reads it sign-extended. The run mode (NX8 or FP32) is not a CR: see
[Element formats](../../explanation/element-formats.md).

## DS half

A DS half describes the active elements of a 128-element vector. The instructions with a `ds`
operand read it: {_half_users(view, "ds")}.

| Field | Bits | Values |
|-------|------|--------|
{_field_rows(DS_LAYOUT, _DS_NOTES)}

The emulator's initial `CR15.L` is `{DEFAULT_DS:#06x}`: every element active, one partition.
Any other CR starts at 0, which is no valid DS (`valid_partitions` 0): an instruction that
reads it is refused.

### `mask_type`

| Type | Code | Fill value |
|------|------|------------|
{mask_rows}

## QF half

A QF half names the NX8 type a row is read or quantized in. The instructions with a `qf`
operand read it in NX8 mode: {_half_users(view, "qf")}.

| Field | Bits | Values |
|-------|------|--------|
{_field_rows(QF_LAYOUT, _QF_NOTES)}

## MULT constant

A MULT constant operand reads a whole CR as one NX8 element with its own scale and type
(FP32 mode: its IEEE binary32 bits).

| Field | Bits | Values |
|-------|------|--------|
{_field_rows(CONSTANT_LAYOUT, _CONSTANT_NOTES)}

```asm
AGG.SUM LR0 CR15.L;;
ACTIVATE.QUANTIZE relu CR15.L CR14.L CR3 LR0;;
MULT.RC.VV LR2 R0 0 LR4 CR15.L CR14.L;;
```

## From the emulator

```python
from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK
from ipu_isa.cr_halves import ds, qf
from ipu_isa.nx8 import Nx8Format

state.regfile.set_cr(2, 0x40000)                           # a base address
state.regfile.set_cr(9, (-128) & LR_CR_SCALAR_VALUE_MASK)  # a negative constant
state.set_cr_half(3, False, ds(valid_elements=64))         # CR3.L: a DS
state.set_cr_half(14, False, qf(Nx8Format(sign=1, exp_format=4, num_bits=8)))  # CR14.L: E4
```
"""
    output_path.write_text(content)
    print(f"Generated configuration-register reference at {output_path}")


def generate_all_docs(out_dir: Path, view: Isa | None = None) -> None:
    """Write every page of ``REFERENCE_PAGES`` into ``out_dir``."""
    generate_operand_types_md(out_dir / "operand-types.md", view)
    generate_assembly_syntax_md(out_dir / "assembly-syntax.md")
    generate_instruction_docs(out_dir / "instructions.md", view)
    generate_programmer_guide_md(out_dir / "programmer-guide.md", view)
    generate_registers_md(out_dir / "registers.md")
    generate_configuration_registers_md(out_dir / "configuration-registers.md", view)
