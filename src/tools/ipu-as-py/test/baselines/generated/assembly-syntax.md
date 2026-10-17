# Assembly Language Syntax

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
    LD.MULT r0 lr0 cr0;           # LOAD: load into mult stage r0
    MULT.RC.VV lr1 r0 0 lr3 cr15; # MULT: element-wise multiply
    ACC.ADD;                      # ACC: accumulate
    INC lr0 1;                    # LR: bump address (increment via INC)
    BNE lr0 lr1 next;             # COND: branch
    ;;
```

By convention, **instruction mnemonics are written in upper case** in documentation and examples (e.g. `MULT.RC.VV`, `LD.MULT`). **Operand tokens** use **lower case** (`lr0`, `r0`, `cr0`). The assembler accepts **any case** for mnemonics and register tokens.

## Compound instructions

A full IPU instruction is one **compound** word: several slots execute in parallel in a single cycle.

**Pattern:**

```asm
load_inst; mult_inst; acc_inst; aaq_inst; store_inst; acc_store_inst; lr_inst_a; lr_inst_b; lr_inst_c; cond_inst; break_inst;;
```

**Rules:**

- Each slot appears a fixed number of times (see `SLOT_COUNT` in `instruction_spec.py`); unused slots are filled with that slot’s **NOP** by the assembler.
- Slot order in the binary word (MSB → LSB) is defined by the toolchain: `cond`, three `lr` sub-slots, `load`, `mult`, `acc`, `aaq`, `store`, `acc_store`, `break` (see the layout diagram on the [Instruction reference](instructions.md#compound-instruction-layout)).
- The slots of one word execute in a fixed order, and each register read sees either the value at cycle start or the value forwarded from an earlier slot of the same word: see [Execution order and timing](#execution-order-and-timing).
- The **acc_store** slot (`ST.ACC`) is **simulation-only** — not implemented in real IPU hardware.

**Example (parallel slots):**

```asm
LD.MULT r0 lr0 cr0; MULT.RC.VV lr1 r0 0 lr3 cr15; ACC.ADD; INC lr0 1; BNE lr0 lr1 loop;;
```

## Register names

The mult-stage and scalar register **tokens** below are derived from `REGISTER_DEFINITIONS` in `ipu_isa.registers` (same source as the assembler).

| Kind | Assembler tokens | Notes |
|------|------------------|--------|
| Mult stage | `r0`, `r1` | 128-byte vectors; 2-bit mult-stage field (`2` reserved). See [MultStageReg](operand-types.md#multstagereg). |
| Cyclic / mask | *(architectural)* | Cyclic (**RC**) and mask (**RM**) register files are documented per instruction in the reference; operands pass **byte offsets** via `LR` values. |
| Loop / scalar | `lr0`–`lr15` | General-purpose; **read/write**. See [LrIdx](operand-types.md#lridx). |
| Constant | `cr0`–`cr15` | **Read-only** in assembly; initialized by the harness. **`cr0`** / **`cr1`** are often 0 and 1. **`cr15`** conventionally holds dstructure configuration (see [DstructureCrIdx](operand-types.md#dstructurecridx)) but is a valid `CrIdx`/`LcrIdx` operand like any other CR. See [CrIdx](operand-types.md#cridx). |

The **`mem_bypass`** vector may still exist in the **emulator regfile** for debugging, but it is **not** a valid mult-stage assembly operand.

**LR slot:** **`SET`** copies from a **`cr`** register (see [CrIdx](operand-types.md#cridx)); **`ADD`**/**`SUB`**'s **`src_b`** is register-only (see [LcrIdx](operand-types.md#lcridx)) — use **`INC`**/**`DEC`** for an immediate operand instead (see [LrIncDecImmediate](operand-types.md#lrincdecimmediate)); **`INC.MODP2`** uses a dedicated **k** immediate (see [LrModPow2KImmediate](operand-types.md#lrmodpow2kimmediate)).

## Labels and branches

```asm
start:
    SET lr0 cr0;;
loop:
    INC lr0 1;;
    BNE lr0 lr1 loop;;
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

The slots of one instruction word execute in this order: **BREAK** → **LR** → **LOAD** → **MULT** → **ACC** → **AAQ** → **STORE** → **ACC_STORE** → **COND**. **BREAK** is evaluated before any slot writes (it may halt the word); the **LR** sub-slots all read before any of them writes.

A register read sees one of two values: the value at **cycle start** (when the word began), or the **forwarded** value (after the writes of earlier slots of the same word). The timing contract is four rules:

1. The Control stage (**LR**, **COND**) reads LRs at cycle start; every other stage reads LRs after the LR writes of the same word.
2. **MULT** reads `R0`, `R1`, `R_CYCLIC` and `R_MASK` at cycle start.
3. `MULT_RES` is forwarded to **ACC** (including AGG and RESHAPE); `R_ACC` is forwarded to **AAQ**.
4. LR sub-slots read cycle-start values.

Any other read is forwarded when every slot that writes the register runs earlier in the word, and at cycle start otherwise. CRs are never written during a run, so CR reads are at cycle start.

### Register reads and writes

`R` is the `R0`/`R1` pair. An operand of type `LcrIdx` reads an LR or a CR.

| Instruction | Slot | Reads at cycle start | Reads forwarded | Writes |
|-------------|------|----------------------|-----------------|--------|
| `BREAK` | `BREAK` | — | — | — |
| `BREAK.IFEQ` | `BREAK` | `LR` via `reg` | — | — |
| `SET` | `LR` | `CR` via `src` | — | `LR` via `reg` |
| `ADD` | `LR` | `LR` via `src_a`, `LR` via `src_b`, `CR` via `src_b` | — | `LR` via `dest` |
| `SUB` | `LR` | `LR` via `src_a`, `LR` via `src_b`, `CR` via `src_b` | — | `LR` via `dest` |
| `INC.MODP2` | `LR` | `LR` via `step`, `CR` via `step`, `LR` via `dest` | — | `LR` via `dest` |
| `INC` | `LR` | `LR` via `dest` | — | `LR` via `dest` |
| `DEC` | `LR` | `LR` via `dest` | — | `LR` via `dest` |
| `ADDB` | `LR` | `LR` via `src_b`, `CR` via `src_b`, `LR` via `dest` | — | `LR` via `dest` |
| `ADDBI` | `LR` | `LR` via `dest` | — | `LR` via `dest` |
| `LD.MULT` | `LOAD` | `CR` via `base` | `LR` via `offset` | `R` via `dest` |
| `LD.MULT.CYCLIC` | `LOAD` | `CR` via `base` | `LR` via `offset`, `LR` via `index` | `R_CYCLIC` |
| `LD.MASK` | `LOAD` | `CR` via `base` | `LR` via `offset` | `R_MASK` |
| `MULT.RC.VV` | `MULT` | `R` via `ra`, `R_CYCLIC`, `CR` via `cr_idx` | `LR` via `rc_idx`, `LR` via `mask_shift`, `R_MASK` | `MULT_RES` |
| `MULT.RC.VE` | `MULT` | `CR` via `src`, `R`, `R_CYCLIC`, `CR` via `cr_idx` | `LR` via `rc_idx`, `LR` via `mask_shift`, `LR` via `src`, `R_MASK` | `MULT_RES` |
| `MULT.RC.VS` | `MULT` | `R_CYCLIC`, `CR` via `cr_idx` | `LR` via `rc_idx`, `LR` via `mask_shift`, `R_MASK` | `MULT_RES` |
| `MULT.VE` | `MULT` | `CR` via `cr_idx`, `R`, `CR` via `dstructure_cr_idx` | `LR` via `ra_idx`, `LR` via `mask_shift`, `R_MASK` | `MULT_RES` |
| `MULT.EE` | `MULT` | `CR` via `cr_idx`, `R`, `CR` via `dstructure_cr_idx` | `LR` via `ra_idx`, `LR` via `mask_shift`, `R_MASK` | `MULT_RES` |
| `ACC.ADD` | `ACC` | `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.ADD.FIRST` | `ACC` | — | `MULT_RES` | `R_ACC` |
| `ACC.MAX` | `ACC` | `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.MAX.FIRST` | `ACC` | — | `MULT_RES` | `R_ACC` |
| `ACC.SUB` | `ACC` | `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.SUB.FIRST` | `ACC` | — | `MULT_RES` | `R_ACC` |
| `ACC.STRIDE` | `ACC` | — | `LR` via `offset`, `MULT_RES` | `R_ACC` |
| `AGG.SUM.FIRST` | `ACC` | `LR` via `dest_slot`, `CR` via `cr_idx` | `MULT_RES` | `R_ACC` |
| `AGG.SUM` | `ACC` | `LR` via `dest_slot`, `CR` via `cr_idx`, `R_ACC` | `MULT_RES` | `R_ACC` |
| `AGG.MAX.FIRST` | `ACC` | `LR` via `dest_slot`, `CR` via `cr_idx` | `MULT_RES` | `R_ACC` |
| `AGG.MAX` | `ACC` | `LR` via `dest_slot`, `CR` via `cr_idx`, `R_ACC` | `MULT_RES` | `R_ACC` |
| `ACC.RESHAPE` | `ACC` | `LR` via `source`, `LR` via `dest`, `MULT_RES` | `LR` via `reshape_mask` | `R_ACC` |
| `ACTIVATE.QUANTIZE` | `AAQ` | `CR` via `cr_idx` | `R_ACC` | `POST_AAQ_REG` |
| `ST.AAQ` | `STORE` | `CR` via `base` | `LR` via `offset`, `POST_AAQ_REG` | — |
| `ST.ACC` | `ACC_STORE` | `CR` via `base` | `LR` via `offset`, `R_ACC` | — |
| `BEQ` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BNE` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BLT.S` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BGE.S` | `COND` | `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2` | — | — |
| `BR` | `COND` | `LR` via `reg`, `CR` via `reg` | — | — |
| `HALT` | `COND` | — | — | — |

### Known deviations

Where the emulator differs from the contract today. Each row is removed by the issue named in the last column.

| Instruction | Read | Today | Contract | Fixed by |
|-------------|------|-------|----------|----------|
| `AGG.SUM.FIRST` | `LR` via `dest_slot` | cycle start | forwarded | E04-25 |
| `AGG.SUM` | `LR` via `dest_slot` | cycle start | forwarded | E04-25 |
| `AGG.MAX.FIRST` | `LR` via `dest_slot` | cycle start | forwarded | E04-25 |
| `AGG.MAX` | `LR` via `dest_slot` | cycle start | forwarded | E04-25 |
| `ACC.RESHAPE` | `LR` via `source` | cycle start | forwarded | E04-25 |
| `ACC.RESHAPE` | `LR` via `dest` | cycle start | forwarded | E04-25 |
| `ACC.RESHAPE` | `MULT_RES` | cycle start | forwarded | E04-26 |
| `MULT.RC.VV` | `R_MASK` | forwarded | cycle start | E04-27 |
| `MULT.RC.VE` | `R_MASK` | forwarded | cycle start | E04-27 |
| `MULT.RC.VS` | `R_MASK` | forwarded | cycle start | E04-27 |
| `MULT.VE` | `R_MASK` | forwarded | cycle start | E04-27 |
| `MULT.EE` | `R_MASK` | forwarded | cycle start | E04-27 |

## Masking

Multiply instructions (`MULT.RC.VV`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.VE`, `MULT.EE`) support **element masking**: after the multiply, elements in `MULT_RES` whose
corresponding mask bit is **1** are **active** and pass through to accumulation; elements whose bit is
**0** are **zeroed** (deactivated). A mask of all-ones leaves every element active (the reset default);
a mask of all-zeros deactivates every element.

### R_MASK register

`R_MASK` is a 128-byte register packed as **8 independent 128-bit slots** (slots 0–7). Load it
from XMEM with:

```asm
LD.MASK offset, base;;
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
LD.MASK LR1, CR2;;

# Load a vector into R0, load cyclic data into R_CYCLIC
LD.MULT R0, LR0, CR0;;
LD.MULT.CYCLIC LR2, CR0, LR5;;

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
LD.MULT r0 lr0 cr0;;
```

The harness must initialize **`cr6`** (here: the memory offset) before this program runs.

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
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input prog.asm --output prog.bin
```

**In Bazel build rules:** use the project’s `ipu_asm` rule (see [Building Applications](building-applications.md)).

## Further reading

- [Operand types](operand-types.md) — field types used in `instruction_spec`
- [Instruction reference](instructions.md) — per-opcode documentation
- [Adding instructions](how-to/add-an-instruction.md)
- [Building applications](building-applications.md)
