# Assembly Language Syntax

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
| Mult stage | `r0`, `r1` | 128-byte vectors; 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | `lr0`–`lr15` | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | `cr0`–`cr15` | **Read-only** in assembly; initialized by the harness. **`cr0`** / **`cr1`** are often 0 and 1. **`cr15`** conventionally holds dstructure configuration (see [DstructureCrIdx](operand-types.md#dstructurecridx)) but is a valid `CrIdx`/`LcrIdx` operand like any other CR. See [CrIdx](operand-types.md#cridx). |

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
