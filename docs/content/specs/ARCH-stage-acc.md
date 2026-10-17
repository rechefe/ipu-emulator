# Accumulator Stage

## 1. Overview

The ACC (Accumulator) stage owns `R_ACC`, a 128-element register of 32-bit
elements, and combines each cycle's multiply result into it. It performs the
following tasks:

- **Accumulate:** Adds, subtracts or takes the maximum of the multiply result
  and `R_ACC`, element by element.
- **Stride:** Reshapes the multiply result with a predetermined pattern (every
  second column and/or row) into a range of `R_ACC`.
- **Aggregate:** Reduces the valid elements of the multiply result to one
  value (sum or maximum) and writes it into one `R_ACC` element.
- **Reshape:** Copies up to 8 elements of the multiply result to `R_ACC`. The
  source and destination of each element are arbitrary.

> **Note:** `R_ACC` is the stage's output to the AaQ stage every cycle.

## Table of Contents

1. [Overview](#1-overview)
2. [Block Diagram](#2-block-diagram)
3. [Interfaces](#3-interfaces)
    - 3.1 [`dstructure` Layout](#31-dstructure-layout)
    - 3.2 [Element Format](#32-element-format)
    - 3.3 [Running and `.FIRST` Variants](#33-running-and-first-variants)
4. [Accumulator ISA](#4-accumulator-isa)
    - 4.1 [`NOP`: No Operation](#41-nop-no-operation)
    - 4.2 [`ACC.ADD`, `ACC.SUB`, `ACC.MAX`: Element-wise Accumulate](#42-accadd-accsub-accmax-element-wise-accumulate)
    - 4.3 [`ACC.STRIDE`: Accumulator Stride](#43-accstride-accumulator-stride)
    - 4.4 [`AGG.SUM`, `AGG.MAX`: Aggregate](#44-aggsum-aggmax-aggregate)
    - 4.5 [`ACC.RESHAPE`: Reshape](#45-accreshape-reshape)

## 2. Block Diagram

```mermaid
block-beta
    columns 7
    block:INS:1
        columns 1
        IN_DATA["input data<br>128 x 32-bit"]
        INSTR["Instruction"]
    end
    space
    block:STAGE:3
        columns 3
        TITLE["ACC Stage"] space:2
        ALU["Accumulate /<br>Aggregate"] space RACC["R_ACC<br>128 x 32-bit"]
        space:3
        PERM["Stride /<br>Reshape"] space:2
    end
    space
    block:OUTS:1
        columns 1
        OUT_DATA["output data<br>128 x 32-bit"]
    end

    IN_DATA --> ALU
    IN_DATA --> PERM
    INSTR --> ALU
    ALU <--> RACC
    PERM --> RACC
    RACC --> OUT_DATA

    classDef io fill:none,stroke:none
    classDef blk fill:#2e9e8c,stroke:#1a7060,color:#fff
    classDef title fill:none,stroke:none,font-weight:bold
    class IN_DATA,INSTR,OUT_DATA,INS,OUTS io
    class ALU,RACC,PERM blk
    class TITLE title
    style STAGE fill:none,stroke:#888,stroke-width:2px
```

## 3. Interfaces

| Name | Direction | Description |
|------|-----------|-------------|
| input data | Input | This cycle's multiply result from the MULT stage: 128 elements of 32 bits. |
| Instruction | Input | The operation that the stage does on the input data. |
| output data | Output | `R_ACC`: 128 elements of 32 bits, to the AaQ stage. |

### 3.1 `dstructure` Layout

The `cr_idx` operand of `AGG.SUM` and `AGG.MAX` names a 32-bit `CR` that holds
the data structure format of the 128-element vector. It has four 8-bit fields:

| Bits | Field | Description |
|------|-------|-------------|
| `[31:24]` | `N/A` | Not used. |
| `[23:16]` | `valid_partition` | Number of valid groups. The rightmost `num_partition - valid_partition` groups are masked. |
| `[15:8]` | `num_partition` | Number of groups: `1`, `2`, `4` or `8`. `1` means one line of 128 elements, `2` means two lines of 64 elements. |
| `[7:0]` | `valid_elements` | Number of valid elements in a group. |

### 3.2 Element Format

Each `R_ACC` element is an FP32 value (IEEE-754 single precision), the same as
each element of `input data`. It is never `inf` or `-inf`.

### 3.3 Running and `.FIRST` Variants

`ACC.ADD`, `ACC.SUB`, `ACC.MAX`, `AGG.SUM` and `AGG.MAX` each have a `.FIRST`
variant:

| Variant | Uses the current `R_ACC` value | Purpose |
|---------|--------------------------------|---------|
| Running (`ACC.ADD`) | Yes | Accumulate across cycles. |
| `.FIRST` (`ACC.ADD.FIRST`) | No: the result depends on `input data` only | Start a new operation. |

Running variants read `R_ACC` as it was at the start of the cycle.

## 4. Accumulator ISA

The ACC slot is resolved by CTRL and forwarded down the pipeline. The stage
does not read the CR/LR register files itself (see the Control Stage spec,
section 5).

The opcode enum (`acc_inst_opcode_t`, package `ipu_instr_pkg`) is generated
from [`instruction_spec.py`](../../../src/tools/ipu-common/src/ipu_common/instruction_spec.py)
by [`gen_codegen.py`](../../../src/tools/ipu-as-py/src/ipu_as/gen_codegen.py).

> **Note:** Each `CR0`-`CR15` and `LR0`-`LR15` register named in the
> instruction operands is **32 bits** wide. An `LRD` operand (a pair of LRs)
> comes with its value, as a `CR` operand does.

**Instructions in this section:**

| Instruction | Purpose |
|-------------|---------|
| [`NOP`](#41-nop-no-operation) | Do nothing. |
| [`ACC.ADD`, `ACC.SUB`, `ACC.MAX`](#42-accadd-accsub-accmax-element-wise-accumulate) | Combine `input data` with `R_ACC`, element by element. |
| [`ACC.STRIDE`](#43-accstride-accumulator-stride) | Reshape `input data` into a range of `R_ACC` with a predetermined pattern. |
| [`AGG.SUM`, `AGG.MAX`](#44-aggsum-aggmax-aggregate) | Reduce `input data` to one `R_ACC` element. |
| [`ACC.RESHAPE`](#45-accreshape-reshape) | Copy chosen `input data` elements to chosen `R_ACC` elements. |

---

### 4.1 `NOP`: No Operation

No operation for the ACC slot. `R_ACC` keeps its value.

#### Syntax

```text
NOP
```

#### Operands

None.

#### Example

```text
NOP;;
```

---

### 4.2 `ACC.ADD`, `ACC.SUB`, `ACC.MAX`: Element-wise Accumulate

Combines each element of `input data` with the same element of `R_ACC`. All
128 elements are written.

#### Syntax

```text
ACC.ADD
ACC.ADD.FIRST
ACC.SUB
ACC.SUB.FIRST
ACC.MAX
ACC.MAX.FIRST
```

#### Operands

None.

#### Operation

For each element `i` in `0..127`:

| Mnemonic | `R_ACC[i]` becomes |
|----------|--------------------|
| `ACC.ADD` | `R_ACC[i] + input_data[i]` |
| `ACC.ADD.FIRST` | `input_data[i]` |
| `ACC.SUB` | `R_ACC[i] - input_data[i]` |
| `ACC.SUB.FIRST` | `0 - input_data[i]` |
| `ACC.MAX` | `input_data[i]` if `input_data[i] > R_ACC[i]`, else `R_ACC[i]` |
| `ACC.MAX.FIRST` | `input_data[i]` |

`ACC.SUB.FIRST` computes `0 - x`, so FP32 `+0.0` gives `+0.0`. In `ACC.MAX`, a
NaN in either operand leaves `R_ACC[i]` unchanged.

#### Example

```text
ACC.ADD.FIRST;;      // first partial product of a dot product
ACC.ADD;;            // add the next ones
```

---

### 4.3 `ACC.STRIDE`: Accumulator Stride

A reshape with a predetermined pattern. It views `input data` as rows, keeps
every second column and/or every second row, and writes the kept elements, row
by row, into a range of `R_ACC`. The other `R_ACC` elements keep their values.

#### Syntax

```text
ACC.STRIDE elements_in_row, horizontal_stride, vertical_stride, offset
```

#### Operands

The Type column links to the [operand type reference](../operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `elements_in_row` | [`Imm`](../operand-types.md#elementsinrow) | `16`, `32` or `64`. Elements per row of `input data`. |
| `horizontal_stride` | [`Imm`](../operand-types.md#horizontalstride) | `off`, `on` or `on_inv`. Column pattern. |
| `vertical_stride` | [`Imm`](../operand-types.md#verticalstride) | `off`, `on` or `on_inv`. Row pattern. |
| `offset` | [`LrIdx`](../operand-types.md#lridx) | `LR0`-`LR15`. Its value, after this cycle's LR updates, selects the first written `R_ACC` element: `start = (offset % 4) * 32`. |

**Pattern modes:**

| Mode | Columns kept (`horizontal_stride`) | Rows kept (`vertical_stride`) |
|------|------------------------------------|-------------------------------|
| `off` | all | all |
| `on` | even: `0, 2, 4, ...` | even: `0, 2, 4, ...` |
| `on_inv` | odd: `1, 3, 5, ...` | odd: `1, 3, 5, ...` |

#### Operation

```text
rows = 128 / elements_in_row
view input_data as [rows][elements_in_row]          // row r = input_data[r*elements_in_row ...]
keep the columns selected by horizontal_stride
keep the rows selected by vertical_stride
kept = the kept elements, row by row                  // N elements

start = (offset % 4) * 32
R_ACC[start + k] = kept[k]   for k in 0..N-1
```

The number of written elements is:

| `horizontal_stride` | `vertical_stride` | `N` |
|---------------------|-------------------|-----|
| `off` | `off` | 128 |
| `on` / `on_inv` | `off` | 64 |
| `off` | `on` / `on_inv` | 64 |
| `on` / `on_inv` | `on` / `on_inv` | 32 |

`start + N` is at most 128.

**Example:** `elements_in_row = 16`, `horizontal_stride = on`,
`vertical_stride = on`, `offset % 4 = 0`. `input data` is 8 rows of 16. The
even columns of rows 0, 2, 4 and 6 are kept: 32 elements, written to
`R_ACC[0..31]`.

![ACC.STRIDE example: 8 rows of 16, even columns of even rows kept](assets/acc-stride.svg)

> **Note:** When a data structure has fewer than 8 elements, the hardware pads
> it to 16.

#### Example

```text
ACC.STRIDE 16, on, on, LR0;;
```

---

### 4.4 `AGG.SUM`, `AGG.MAX`: Aggregate

Reduces the first `valid_elements` elements of `input data` to one value and
writes it into one `R_ACC` element. The other `R_ACC` elements keep their
values.

#### Syntax

```text
AGG.SUM dest_slot, cr_idx
AGG.SUM.FIRST dest_slot, cr_idx
AGG.MAX dest_slot, cr_idx
AGG.MAX.FIRST dest_slot, cr_idx
```

#### Operands

The Type column links to the [operand type reference](../operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](../operand-types.md#lridx) | `LR0`-`LR15`. Its value, before this cycle's LR updates, selects the written element: `dest = dest_slot % 128`. |
| `cr_idx` | [`CrIdx`](../operand-types.md#cridx) | `CR0`-`CR15`. Supplies `valid_elements` (section 3.1). |

#### Operation

```text
n    = min(valid_elements, 128)
dest = dest_slot % 128
```

| Mnemonic | `R_ACC[dest]` becomes |
|----------|-----------------------|
| `AGG.SUM` | `R_ACC[dest] + sum(input_data[0..n-1])` |
| `AGG.SUM.FIRST` | `sum(input_data[0..n-1])` |
| `AGG.MAX` | `max(R_ACC[dest], input_data[0..n-1])` |
| `AGG.MAX.FIRST` | `max(seed, input_data[0..n-1])` |

| Rule | Value |
|------|-------|
| Sum order | Element `0` first, then `1`, ..., `n-1`, one at a time. |
| Max | A candidate replaces the current maximum only when it is greater. |

> **TBD - ask Jacob:** the value of `seed`. Can `seed` be `-inf`? FP32 elements
> are never `inf` or `-inf`.

> **TBD - ask Aviv:** whether the aggregate uses the full active-elements rule
> (`valid_partition`, `num_partition`, `valid_elements`) or only the first
> `valid_elements` elements of the vector. The full rule might be overkill.

#### Example

```text
AGG.SUM.FIRST LR0, CR3;;      // R_ACC[LR0 % 128] = sum of the first valid_elements elements
```

---

### 4.5 `ACC.RESHAPE`: Reshape

A reshape with an arbitrary pattern: each element goes from any source index
to any destination index. It copies up to 8 elements of `input data` to
`R_ACC`. Two `LRD` register pairs hold the element indices: one list of 8
source indices and one list of 8 destination indices. The other `R_ACC`
elements keep their values.

#### Syntax

```text
ACC.RESHAPE source, dest, reshape_mask
```

#### Operands

The Type column links to the [operand type reference](../operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `source` | `LrdIdx` | `LRD0`, `LRD2`, ..., `LRD14`. 8 byte indices `source[0..7]` into `input data`. |
| `dest` | `LrdIdx` | `LRD0`, `LRD2`, ..., `LRD14`. 8 byte indices `dest[0..7]` into `R_ACC`. |
| `reshape_mask` | `LrIdx` / `Imm` | An immediate `0`-`7`, or `LR0`-`LR15` whose value is `0`-`8`. Elements `reshape_mask..7` take part. |

**`reshape_mask` values.** `reshape_mask` skips the first `reshape_mask`
elements of the index lists. Elements `k = reshape_mask..7` are copied:

| `reshape_mask` | Skipped `k` | Copied `k` | Elements copied |
|----------------|-------------|------------|-----------------|
| `0` | none | `0`-`7` | 8 |
| `1` | `0` | `1`-`7` | 7 |
| `2` | `0`-`1` | `2`-`7` | 6 |
| `3` | `0`-`2` | `3`-`7` | 5 |
| `4` | `0`-`3` | `4`-`7` | 4 |
| `5` | `0`-`4` | `5`-`7` | 3 |
| `6` | `0`-`5` | `6`-`7` | 2 |
| `7` | `0`-`6` | `7` | 1 |
| `8` (`LR` only) | `0`-`7` | none | 0 |

> **TBD - ask Aviv:** whether `reshape_mask` skips elements as in the table
> (elements `0`..`reshape_mask - 1` are skipped, the rest are copied), or
> counts from left to right (the first `reshape_mask` elements are copied).

**Index lists.** `LRDn` is the pair `{LR(n+1), LR(n)}`. Byte `k` of its value is
index `k`:

| Index | Source |
|-------|--------|
| `0`-`3` | Bytes 0-3 of `LR(n)` (byte 0 = bits `[7:0]`) |
| `4`-`7` | Bytes 0-3 of `LR(n+1)` |

Every participating `source[k]` and `dest[k]` is `0`-`127`.

#### Operation

```text
for k in reshape_mask..7:                     // ascending k
    R_ACC[dest[k]] = input_data[source[k]]
```

When two `dest[k]` are equal, the higher `k` wins.

**Example:** `reshape_mask = 2`, `source = 5, 9, 127, 100, 64, 42, 33, 20`,
`dest = 0, 1, 18, 40, 64, 90, 101, 126`. Elements `k = 2..7` are copied, for
example `R_ACC[18] = input_data[127]` and `R_ACC[40] = input_data[100]`.
Elements `k = 0, 1` are skipped. The figure has one column per `k`.

![ACC.RESHAPE example: reshape_mask = 2, k = 2..7 copied, k = 0, 1 skipped](assets/acc-reshape.svg)

#### Example

```text
ACC.RESHAPE LRD0, LRD2, 2;;      // copy elements 2-7 of the index lists
```
