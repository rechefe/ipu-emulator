# Assembly Language Syntax

The IPU assembler is a Python-based tool that turns assembly source into binary VLIW instructions for the emulator. Instruction names, operands, and encoding are driven by `instruction_spec.py` in `ipu_isa`; the [Instruction reference](instructions.md) and [Operand types](operand-types.md) are generated from that same toolchain.

## Features

- **Jinja2 preprocessing** — templates and macros for generated assembly
- **Label resolution** — forward and backward references
- **VLIW compounds** — multiple pipeline slots per cycle (`COND`, `LR`×3, `LOAD`, `MULT`, `ACC`, `AAQ`; emulator-only: `AAQ_EXACT`, `BREAK`)
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
- The emulator-only slots (`AAQ_EXACT`, `BREAK`) are not implemented in real IPU hardware: see the [instruction format](instruction-format.md).

**Example (parallel slots):**

```asm
loop:
LD.MULT R0 LR0 CR0; MULT.RC.VV LR1 R0 0 LR3 CR15.L CR14.L; ACC.ADD; INC LR0 1; BNE LR0 LR1 loop;;
```

## Register names

The mult-stage and scalar register **tokens** below are derived from `REGISTER_DEFINITIONS` in `ipu_isa.registers` (same source as the assembler).

| Kind | Assembler tokens | Notes |
|------|------------------|--------|
| Mult stage | `R0`, `R1` | One XMEM row each (128 NX8 elements and the row's metadata word, or 128 FP32 elements in the FP32 mode); 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | `LR0`–`LR15` | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | `CR0`–`CR15` | **Read-only** in assembly; the host writes `CR2`–`CR15` before a kernel runs. **`CR0`** and **`CR1`** are the read-only constants 0 and 1 (see [Configuration registers](configuration-registers.md)). Each CR is also two 16-bit halves, `CR13.L` and `CR13.H`: the DS (data structure) and QF (quantization format) operands always name a half (see [HalfCrIdx](operand-types.md#halfcridx)), and an LR or COND source operand may read a full CR or a sign-extended half (see [LcrHalfIdx](operand-types.md#lcrhalfidx)). **`CR15.L`** conventionally holds the default DS, but any half is a valid operand. See [CrIdx](operand-types.md#cridx) and [Configuration registers](configuration-registers.md). |

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
{% macro count_to(limit_cr) %}
    SET LR0 CR0;;
1:
    INC LR0 1;;
    BNE LR0 {{ limit_cr }} 1b;;
{% endmacro %}
{{ count_to("CR2") }}
{{ count_to("CR3") }}
    HALT;;
```


## Execution order and timing

This section is **generated** from `ipu_isa.timing` and the register accesses each instruction declares in `instruction_spec.py`.

The slots of one instruction word execute in this order: **BREAK** → **LR** → **LOAD** → **MULT** → **ACC** → **AAQ** → **AAQ_EXACT** → **COND**. **BREAK** is evaluated before any slot writes (it may halt the word); the **LR** sub-slots all read before any of them writes.

A register read sees one of two values: the value at **cycle start** (when the word began), or the **forwarded** value (after the writes of earlier slots of the same word). The timing contract is four rules:

1. The Control stage (**LR**, **COND**) reads LRs at cycle start; every other stage reads LRs after the LR writes of the same word.
2. **MULT** reads `R0`, `R1`, `R_CYCLIC`, their row metadata and `R_MASK` at cycle start.
3. `MULT_RES` is forwarded to **ACC** (including AGG and RESHAPE); `R_ACC` is forwarded to **AAQ** and **AAQ_EXACT**.
4. LR sub-slots read cycle-start values.

Any other read is forwarded when every slot that writes the register runs earlier in the word, and at cycle start otherwise. CRs are never written during a run, so CR reads are at cycle start.

The [timing model](../../explanation/timing-model.md) explains the rules, with an example of each.

### Register reads and writes

`R` is the `R0`/`R1` pair. An operand of type `LcrIdx` reads an LR or a CR; one of type `LcrHalfIdx` an LR, a CR or a CR half; one of type `HalfCrIdx` a CR half.

| Instruction | Slot | Reads at cycle start | Reads forwarded | Writes |
|-------------|------|----------------------|-----------------|--------|
| `BREAK` | `BREAK` | — | — | — |
| `BREAK.IFEQ` | `BREAK` | `LR` via `reg` | — | — |
| `SET` | `LR` | `LR` via `src`, `CR` via `src` | — | `LR` via `reg` |
| `ADD` | `LR` | `LR` via `src_a`, `LR` via `src_b`, `CR` via `src_b` | — | `LR` via `dest` |
| `SUB` | `LR` | `LR` via `src_a`, `LR` via `src_b`, `CR` via `src_b` | — | `LR` via `dest` |
| `INC.MODP2` | `LR` | `LR` via `step`, `CR` via `step`, `LR` via `dest` | — | `LR` via `dest` |
| `INC` | `LR` | `LR` via `dest` | — | `LR` via `dest` |
| `DEC` | `LR` | `LR` via `dest` | — | `LR` via `dest` |
| `ADDB` | `LR` | `LR` via `src_b`, `CR` via `src_b`, `LR` via `dest` | — | `LR` via `dest` |
| `ADDBI` | `LR` | `LR` via `dest` | — | `LR` via `dest` |
| `LD.MULT` | `LOAD` | `CR` via `base` | `LR` via `offset` | `R` via `dest`, `R_META` via `dest` |
| `LD.MULT.CYCLIC` | `LOAD` | `CR` via `base` | `LR` via `offset`, `LR` via `index` | `R_CYCLIC`, `R_CYCLIC_META` |
| `LD.MASK` | `LOAD` | `CR` via `base` | `LR` via `offset` | `R_MASK` |
| `MULT.RC.VV` | `MULT` | `R` via `ra`, `R_CYCLIC`, `R_CYCLIC_META`, `R_META`, `R_MASK`, `CR` via `ds`, `CR` via `qf` | `LR` via `rc_idx`, `LR` via `mask_shift` | `MULT_RES` |
| `MULT.RC.VE` | `MULT` | `CR` via `src`, `R`, `R_META`, `R_CYCLIC`, `R_CYCLIC_META`, `R_MASK`, `CR` via `ds`, `CR` via `qf` | `LR` via `rc_idx`, `LR` via `mask_shift`, `LR` via `src` | `MULT_RES` |
| `MULT.RC.VS` | `MULT` | `R_CYCLIC`, `R_CYCLIC_META`, `R_MASK`, `CR` via `ds`, `CR` via `qf` | `LR` via `rc_idx`, `LR` via `mask_shift` | `MULT_RES` |
| `MULT.VE` | `MULT` | `CR` via `cr_idx`, `R`, `R_META`, `R_MASK`, `CR` via `ds`, `CR` via `qf` | `LR` via `ra_idx`, `LR` via `mask_shift` | `MULT_RES` |
| `MULT.EE` | `MULT` | `CR` via `cr_idx`, `R`, `R_META`, `R_MASK`, `CR` via `ds`, `CR` via `qf` | `LR` via `ra_idx`, `LR` via `mask_shift` | `MULT_RES` |
| `ACC.ADD` | `ACC` | `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.ADD.FIRST` | `ACC` | — | `MULT_RES` | `R_ACC` |
| `ACC.MAX` | `ACC` | `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.MAX.FIRST` | `ACC` | — | `MULT_RES` | `R_ACC` |
| `ACC.SUB` | `ACC` | `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.SUB.FIRST` | `ACC` | — | `MULT_RES` | `R_ACC` |
| `ACC.STRIDE` | `ACC` | — | `LR` via `offset`, `MULT_RES` | `R_ACC` |
| `AGG.SUM.FIRST` | `ACC` | `CR` via `ds` | `LR` via `dest_slot`, `MULT_RES` | `R_ACC` |
| `AGG.SUM` | `ACC` | `CR` via `ds`, `R_ACC` | `LR` via `dest_slot`, `MULT_RES` | `R_ACC` |
| `AGG.MAX.FIRST` | `ACC` | `CR` via `ds` | `LR` via `dest_slot`, `MULT_RES` | `R_ACC` |
| `AGG.MAX` | `ACC` | `CR` via `ds`, `R_ACC` | `LR` via `dest_slot`, `MULT_RES` | `R_ACC` |
| `ACC.RESHAPE` | `ACC` | — | `LR` via `source`, `LR` via `dest`, `LR` via `reshape_mask`, `MULT_RES` | `R_ACC` |
| `LD.LUT` | `AAQ` | `CR` via `table_id` | `LR` via `table_offset` | `AAQ_LUT` |
| `ACTIVATE.QUANTIZE` | `AAQ` | `CR` via `table_id`, `CR` via `ds`, `CR` via `qf`, `AAQ_LUT` | `LR` via `table_offset`, `R_ACC` | — |
| `ACTIVATE.EXACT` | `AAQ_EXACT` | `CR` via `table_id`, `CR` via `ds` | `LR` via `table_offset`, `R_ACC` | — |
| `BEQ` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BNE` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BLT.S` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BGE.S` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BR` | `COND` | `LR` via `reg`, `CR` via `reg` | — | — |
| `HALT` | `COND` | — | — | — |

### Known deviations

None: every read in the emulator follows the contract.

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

## Jinja2 preprocessing

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
