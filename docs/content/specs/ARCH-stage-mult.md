# Multiply Stage

## 1. Overview

The MULT (Multiply) stage performs the following tasks:

- **Dequantize:** Converts each NX8 input element to FP32, with the format
  from the quantization format (`qf`) and the vector scale.
- **Multiply:** Multiplies 128 pairs of FP32 elements. The operands come from
  `R0`/`R1`, `R_CYCLIC` or a scalar. Each product is an FP32 value.
- **Mask:** Replaces the products of masked elements with a pad value. The mask
  comes from `R_MASK`.
- **Load:** Owns the load instructions (`LDR_MULT_REG`, `LDR_CYCLIC_MULT_REG`,
  `LDR_MULT_MASK_REG`). Writes the load data from XMEM into one of the stage
  registers: `R0`, `R1`, one slot of `R_CYCLIC` or `R_MASK`.

## Table of Contents

1. [Overview](#1-overview)
2. [Block Diagram](#2-block-diagram)
3. [Interfaces](#3-interfaces)
    - 3.1 [Element Format](#31-element-format)
    - 3.2 [Operand Addressing](#32-operand-addressing)
    - 3.3 [Element Masking](#33-element-masking)
4. [Multiply ISA](#4-multiply-isa)
    - 4.1 [`NOP`: No Operation](#41-nop-no-operation)
    - 4.2 [`LDR_MULT_REG`: Load R0 or R1](#42-ldr_mult_reg-load-r0-or-r1)
    - 4.3 [`LDR_CYCLIC_MULT_REG`: Load an R_CYCLIC Slot](#43-ldr_cyclic_mult_reg-load-an-r_cyclic-slot)
    - 4.4 [`LDR_MULT_MASK_REG`: Load R_MASK](#44-ldr_mult_mask_reg-load-r_mask)
    - 4.5 [`MULT.RC.VV`: R_CYCLIC Vector x Ra Vector](#45-multrcvv-r_cyclic-vector-x-ra-vector)
    - 4.6 [`MULT.RC.VE`: R_CYCLIC Vector x Scalar](#46-multrcve-r_cyclic-vector-x-scalar)
    - 4.7 [`MULT.RC.VS`: R_CYCLIC Vector Square](#47-multrcvs-r_cyclic-vector-square)
    - 4.8 [`MULT.VE`: Ra Vector x CR Scalar](#48-multve-ra-vector-x-cr-scalar)
    - 4.9 [`MULT.EE`: Ra Element x CR Scalar, Broadcast](#49-multee-ra-element-x-cr-scalar-broadcast)
5. [Disagreements with the Emulator](#5-disagreements-with-the-emulator)

## 2. Block Diagram

```mermaid
flowchart LR
    LD_DATA["load data<br>128 x 8-bit<br>+ metadata"]:::io
    INSTR["Instruction"]:::io
    QF["quantization format<br>(qf)"]:::io
    DS["data structure<br>(ds)"]:::io
    OUT_DATA["output data<br>MULT_RES<br>128 x 32-bit"]:::io

    subgraph STAGE["MULT Stage"]
        LDW["Load<br>Registers"]:::blk
        RA["R0 / R1<br>2 x 128 x 8-bit"]:::reg
        RC["R_CYCLIC<br>512 x 8-bit"]:::reg
        RMASK["R_MASK<br>8 x 128-bit"]:::reg
        SEL["Operand<br>Select"]:::blk
        MUL["Multiply"]:::blk
        MASK["Mask"]:::blk
    end

    LD_DATA --> LDW
    LDW --> RA
    LDW --> RC
    LDW --> RMASK
    RA --> SEL
    RC --> SEL
    INSTR ----> SEL
    SEL --> MUL
    QF -----> MUL
    MUL --> MASK
    RMASK --> MASK
    DS ------> MASK
    MASK --> OUT_DATA

    classDef io fill:none,stroke:none
    classDef blk fill:#2e9e8c,stroke:#1a7060,color:#fff
    classDef reg fill:#e6b800,stroke:#b38a00,color:#000
    style STAGE fill:none,stroke:#888,stroke-width:2px
```

## 3. Interfaces

| Name | Direction | Description |
|------|-----------|-------------|
| load data | Input | One XMEM row: 128 8-bit elements (128 B) and the vector metadata ([IPU ARCH 7.2](ARCH-ipu-top.md#72-row-layout)). |
| Instruction | Input | The operation that the stage does on its registers. |
| quantization format (`qf`) | Input | The NX8 format of the input elements ([IPU ARCH 8.1](ARCH-ipu-top.md#81-quantization-format-qf)). |
| data structure (`ds`) | Input | The groups of the 128-element vector and the pad value of masked elements ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). |
| output data | Output | `MULT_RES`: 128 FP32 elements, to the ACC stage. |

> **Note:** The MULT stage owns its load instructions, but the CTRL stage
> prefetches their XMEM read address (`offset + base`). The MULT stage gets
> only the load data.

> **Note:** [IPU ARCH 6.3](ARCH-ipu-top.md#63-register-read-timing) gives the value the stage reads from each register.

### 3.1 Element Format

Input elements are in the NX8 format ([IPU ARCH 7.1](ARCH-ipu-top.md#71-nx8-element-format)). One format applies to all
elements. Each product (`MULT_RES` element) is FP32.

Every MULT instruction converts each NX8 input element to FP32 before it
multiplies.

`R0`, `R1` and each 128-element slot of `R_CYCLIC` keep the metadata of the
vector loaded into them. Each element uses the metadata of the register or
slot that holds it.

A `CR` that supplies a scalar (`src` in `MULT.RC.VE`, `cr_idx` in `MULT.VE` and
`MULT.EE`) holds an NX Constant ([IPU ARCH 8.4](ARCH-ipu-top.md#84-nx-constant)).

### 3.2 Operand Addressing

| Operand source | Element `j` | Range of `j` |
|----------------|-------------|--------------|
| `Ra` | `R0[j]` for `j < 128`, `R1[j - 128]` for `j >= 128` | `0`-`255` |
| `R_CYCLIC` | `R_CYCLIC[j]` | `0`-`511` |

An `Ra` vector operand is `R0` or `R1`.

An `R_CYCLIC` vector operand is 128 consecutive elements from a start index:
element `i` of the operand is `R_CYCLIC[start + i]`.

The start index of an `R_CYCLIC` operand is a multiple of 16, or a value at
most 3 from a multiple of 16: `0`-`3`, `13`-`19`, `29`-`35`, ..., `493`-`499`,
`509`-`511`.

### 3.3 Element Masking

Every multiply instruction masks its result. Bit `i` of a mask belongs to
element `i`. Elements whose mask bit is `1` keep their product; elements whose
bit is `0` get the pad value (`mask_type`, [IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)).

The mask is built in two steps:

1. `mask_offset` selects one of the 8 slots of `R_MASK`.
2. `mask_shift` shifts that slot mask by up to 3 elements. After each
   one-element shift, the mask is ANDed with a partition vector, so mask bits
   never cross a group boundary.

![Element masking: mask_offset selects an R_MASK slot, two shift chains make every shift, mask_shift selects one](assets/mult-masking.svg)

```text
shift = clamp(mask_shift, -3, 3)                       // mask_shift is two's complement
mask  = R_MASK.slot[mask_offset]                        // 128 bits
repeat |shift| times:
    if shift > 0: mask = (mask << 1) & partition_vector          // towards higher elements
    if shift < 0: mask = (mask >> 1) & inverse_partition_vector  // towards lower elements

for i in 0..127:
    if mask[i] == 0: MULT_RES[i] = pad_value
```

**Partitions.** The 128 elements form `num_partitions` partitions (groups) of
`128 / num_partitions` elements ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). Partition `p` holds elements
`p * (128 / num_partitions)` to `(p + 1) * (128 / num_partitions) - 1`.

**Partition vectors.** Two 128-bit vectors mark the partition boundaries:

| Vector | Bits that are `0` | Used by |
|--------|-------------------|---------|
| `partition_vector` | The first element of each partition | `mask_shift > 0` |
| `inverse_partition_vector` | The last element of each partition | `mask_shift < 0` |

All other bits are `1`. The vectors for each `num_partitions`, element `0` first
(`1^n` is `n` ones):

| `num_partitions` | Elements per partition | `partition_vector` | `inverse_partition_vector` |
|-----------------|------------------------|--------------------|----------------------------|
| `1` | `128` | `1^128` | `1^128` |
| `2` | `64` | `0 1^63 0 1^63` | `1^63 0 1^63 0` |
| `4` | `32` | `(0 1^31)` x 4 | `(1^31 0)` x 4 |
| `8` | `16` | `(0 1^15)` x 8 | `(1^15 0)` x 8 |

> **Note:** After reset, every bit of `R_MASK` is `1`: every element keeps its
> product.

> **TBD - ask Aviv:** the default value of `R_MASK`.

> **TBD - ask Aviv:** do we add masking examples?

## 4. Multiply ISA

The MULT and LOAD slots are resolved by CTRL and forwarded down the pipeline. The stage
does not read the CR/LR register files itself (see the [Control Stage spec](ARCH-stage-ctrl.md),
section 3.2).

The opcode enum (`mult_inst_opcode_t`, package `ipu_instr_pkg`) is generated
from [`instruction_spec.py`](../../../src/ipu-isa/src/ipu_isa/instruction_spec.py)
by [`sv_package.py`](../../../src/ipu-gen/src/ipu_gen/sv_package.py).

**Instructions in this section:**

| Instruction | Purpose |
|-------------|---------|
| [`NOP`](#41-nop-no-operation) | Do nothing. |
| [`LDR_MULT_REG`](#42-ldr_mult_reg-load-r0-or-r1) | Load one XMEM row into `R0` or `R1`. |
| [`LDR_CYCLIC_MULT_REG`](#43-ldr_cyclic_mult_reg-load-an-r_cyclic-slot) | Load one XMEM row into one slot of `R_CYCLIC`. |
| [`LDR_MULT_MASK_REG`](#44-ldr_mult_mask_reg-load-r_mask) | Load one XMEM row into `R_MASK`. |
| [`MULT.RC.VV`](#45-multrcvv-r_cyclic-vector-x-ra-vector) | `R_CYCLIC` vector x `R0` or `R1`. |
| [`MULT.RC.VE`](#46-multrcve-r_cyclic-vector-x-scalar) | `R_CYCLIC` vector x a scalar from `Ra` or a `CR`. |
| [`MULT.RC.VS`](#47-multrcvs-r_cyclic-vector-square) | `R_CYCLIC` vector x itself. |
| [`MULT.VE`](#48-multve-ra-vector-x-cr-scalar) | `Ra` vector x a `CR` scalar. |
| [`MULT.EE`](#49-multee-ra-element-x-cr-scalar-broadcast) | One `Ra` element x a `CR` scalar, broadcast to all 128 elements. |

In every `MULT` Operation below, `x` converts both NX8 operands to FP32
([IPU ARCH 7.1](ARCH-ipu-top.md#71-nx8-element-format)) and multiplies them. `apply_mask` is section 3.3.

---

### 4.1 `NOP`: No Operation

No operation for the MULT slot.

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

### 4.2 `LDR_MULT_REG`: Load R0 or R1

Loads one XMEM row into `R0` or `R1`, with its vector metadata.

#### Syntax

```text
LDR_MULT_REG dest, offset, base
```

#### Operands

| Name | Type | Details |
|------|------|---------|
| `dest` | [`Imm`](../reference/isa/operand-types.md#multstagereg) | `R0` or `R1`. The register that is written. |
| `offset` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Row offset in XMEM. |
| `base` | [`CrIdx`](../reference/isa/operand-types.md#cridx) | `CR0`-`CR15`. Base row in XMEM. |

#### Operation

```text
dest = XMEM[offset + base]      // 128 elements + metadata (IPU ARCH 7.2)
```

#### Example

```text
LDR_MULT_REG R0, LR0, CR0;;
```

---

### 4.3 `LDR_CYCLIC_MULT_REG`: Load an R_CYCLIC Slot

Loads one XMEM row into one 128-element slot of `R_CYCLIC`, with its vector
metadata. The other slots keep their values.

#### Syntax

```text
LDR_CYCLIC_MULT_REG offset, base, index
```

#### Operands

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Row offset in XMEM. |
| `base` | [`CrIdx`](../reference/isa/operand-types.md#cridx) | `CR0`-`CR15`. Base row in XMEM. |
| `index` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15` whose value is `0`, `128`, `256` or `384`. The first element of the written slot. |

#### Operation

```text
R_CYCLIC[index .. index + 127] = XMEM[offset + base]   // 128 elements + metadata (IPU ARCH 7.2)
```

#### Example

```text
LDR_CYCLIC_MULT_REG LR0, CR0, LR1;;
```

---

### 4.4 `LDR_MULT_MASK_REG`: Load R_MASK

Loads one XMEM row into `R_MASK` (8 slots of 128 bits, section 3.3).

#### Syntax

```text
LDR_MULT_MASK_REG offset, base
```

#### Operands

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Row offset in XMEM. |
| `base` | [`CrIdx`](../reference/isa/operand-types.md#cridx) | `CR0`-`CR15`. Base row in XMEM. |

#### Operation

```text
R_MASK = XMEM[offset + base]    // 128 B = 8 x 128 bits
```

#### Example

```text
LDR_MULT_MASK_REG LR0, CR0;;
```

---

### 4.5 `MULT.RC.VV`: R_CYCLIC Vector x Ra Vector

Multiplies a 128-element window of `R_CYCLIC` by `R0` or `R1`, element by
element.

#### Syntax

```text
MULT.RC.VV rc_idx, ra, mask_offset, mask_shift, ds, qf
```

#### Operands

The Type column links to the [operand type reference](../reference/isa/operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Start element of the `R_CYCLIC` window (section 3.2). |
| `ra` | [`Imm`](../reference/isa/operand-types.md#multstagereg) | `R0` or `R1`. |
| `mask_offset` | [`Imm`](../reference/isa/operand-types.md#multmaskoffsetimmediate) | `0`-`7`. `R_MASK` slot (section 3.3). |
| `mask_shift` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Mask shift, `-3`-`+3` (section 3.3). |
| `ds` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies `num_partitions` and `mask_type` ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). |
| `qf` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies the NX8 format of the input elements ([IPU ARCH 8.1](ARCH-ipu-top.md#81-quantization-format-qf)). |

#### Operation

```text
for i in 0..127:
    MULT_RES[i] = R_CYCLIC[rc_idx + i] x ra[i]
apply_mask(mask_offset, mask_shift, ds)
```

#### Example

```text
MULT.RC.VV LR0, R0, 0, LR2, CR15.L, CR14.L;;
```

---

### 4.6 `MULT.RC.VE`: R_CYCLIC Vector x Scalar

Multiplies a 128-element window of `R_CYCLIC` by one scalar. The scalar is an
element of `Ra` or a `CR` constant.

#### Syntax

```text
MULT.RC.VE rc_idx, src, mask_offset, mask_shift, ds, qf
```

#### Operands

The Type column links to the [operand type reference](../reference/isa/operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Start element of the `R_CYCLIC` window (section 3.2). |
| `src` | [`LcrIdx`](../reference/isa/operand-types.md#lcridx) | `LR0`-`LR15` or `CR0`-`CR15`. An `LR`'s value selects the scalar `Ra[src]`. A `CR` supplies the scalar, a constant ([IPU ARCH 8.4](ARCH-ipu-top.md#84-nx-constant)). |
| `mask_offset` | [`Imm`](../reference/isa/operand-types.md#multmaskoffsetimmediate) | `0`-`7`. `R_MASK` slot (section 3.3). |
| `mask_shift` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Mask shift, `-3`-`+3` (section 3.3). |
| `ds` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies `num_partitions` and `mask_type` ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). |
| `qf` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies the NX8 format of the input elements ([IPU ARCH 8.1](ARCH-ipu-top.md#81-quantization-format-qf)). |

#### Operation

```text
scalar = (src is an LR) ? Ra[src] : constant(src)          // IPU ARCH 8.4
for i in 0..127:
    MULT_RES[i] = R_CYCLIC[rc_idx + i] x scalar
apply_mask(mask_offset, mask_shift, ds)
```

#### Example

```text
MULT.RC.VE LR0, LR3, 0, LR2, CR15.L, CR14.L;;      // scalar = Ra[LR3]
```

---

### 4.7 `MULT.RC.VS`: R_CYCLIC Vector Square

Multiplies a 128-element window of `R_CYCLIC` by itself, element by element.

#### Syntax

```text
MULT.RC.VS rc_idx, mask_offset, mask_shift, ds, qf
```

#### Operands

The Type column links to the [operand type reference](../reference/isa/operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Start element of the `R_CYCLIC` window (section 3.2). |
| `mask_offset` | [`Imm`](../reference/isa/operand-types.md#multmaskoffsetimmediate) | `0`-`7`. `R_MASK` slot (section 3.3). |
| `mask_shift` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Mask shift, `-3`-`+3` (section 3.3). |
| `ds` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies `num_partitions` and `mask_type` ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). |
| `qf` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies the NX8 format of the input elements ([IPU ARCH 8.1](ARCH-ipu-top.md#81-quantization-format-qf)). |

#### Operation

```text
for i in 0..127:
    element     = R_CYCLIC[rc_idx + i]
    MULT_RES[i] = element x element
apply_mask(mask_offset, mask_shift, ds)
```

#### Example

```text
MULT.RC.VS LR0, 0, LR2, CR15.L, CR14.L;;
```

---

### 4.8 `MULT.VE`: Ra Vector x CR Scalar

Multiplies `R0` or `R1` by a `CR` constant.

#### Syntax

```text
MULT.VE ra_idx, cr_idx, mask_offset, mask_shift, ds, qf
```

#### Operands

The Type column links to the [operand type reference](../reference/isa/operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `ra_idx` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15` whose value is `0` (`R0`) or `128` (`R1`) (section 3.2). |
| `cr_idx` | [`CrIdx`](../reference/isa/operand-types.md#cridx) | `CR0`-`CR15`. Supplies the scalar, a constant ([IPU ARCH 8.4](ARCH-ipu-top.md#84-nx-constant)). |
| `mask_offset` | [`Imm`](../reference/isa/operand-types.md#multmaskoffsetimmediate) | `0`-`7`. `R_MASK` slot (section 3.3). |
| `mask_shift` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Mask shift, `-3`-`+3` (section 3.3). |
| `ds` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies `num_partitions` and `mask_type` ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). |
| `qf` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies the NX8 format of the input elements ([IPU ARCH 8.1](ARCH-ipu-top.md#81-quantization-format-qf)). |

#### Operation

```text
for i in 0..127:
    MULT_RES[i] = Ra[ra_idx + i] x constant(cr_idx)          // IPU ARCH 8.4
apply_mask(mask_offset, mask_shift, ds)
```

#### Example

```text
MULT.VE LR0, CR3, 0, LR2, CR15.L, CR14.L;;
```

---

### 4.9 `MULT.EE`: Ra Element x CR Scalar, Broadcast

Multiplies one element of `Ra` by a `CR` constant and writes the
product to all 128 elements.

#### Syntax

```text
MULT.EE ra_idx, cr_idx, mask_offset, mask_shift, ds, qf
```

#### Operands

The Type column links to the [operand type reference](../reference/isa/operand-types.md).

| Name | Type | Details |
|------|------|---------|
| `ra_idx` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. The `Ra` element, `Ra[ra_idx]` (section 3.2). |
| `cr_idx` | [`CrIdx`](../reference/isa/operand-types.md#cridx) | `CR0`-`CR15`. Supplies the scalar, a constant ([IPU ARCH 8.4](ARCH-ipu-top.md#84-nx-constant)). |
| `mask_offset` | [`Imm`](../reference/isa/operand-types.md#multmaskoffsetimmediate) | `0`-`7`. `R_MASK` slot (section 3.3). |
| `mask_shift` | [`LrIdx`](../reference/isa/operand-types.md#lridx) | `LR0`-`LR15`. Mask shift, `-3`-`+3` (section 3.3). |
| `ds` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies `num_partitions` and `mask_type` ([IPU ARCH 8.2](ARCH-ipu-top.md#82-data-structure-ds)). |
| `qf` | [`HalfCrIdx`](../reference/isa/operand-types.md#halfcridx) | `CR0.L`-`CR15.H`. Supplies the NX8 format of the input elements ([IPU ARCH 8.1](ARCH-ipu-top.md#81-quantization-format-qf)). |

#### Operation

```text
product = Ra[ra_idx] x constant(cr_idx)                    // IPU ARCH 8.4
for i in 0..127:
    MULT_RES[i] = product
apply_mask(mask_offset, mask_shift, ds)
```

#### Example

```text
MULT.EE LR0, CR3, 0, LR2, CR15.L, CR14.L;;
```

## 5. Disagreements with the Emulator

| Topic | Spec | Emulator |
|-------|------|----------|
| Load mnemonics | `LDR_MULT_REG`, `LDR_CYCLIC_MULT_REG` and `LDR_MULT_MASK_REG`. | `LD.MULT`, `LD.MULT.CYCLIC` and `LD.MASK`. |
| `R_CYCLIC` start index | Section 3.2. | Any value, modulo 512. |
| `MULT.VE` `ra_idx` | `0` (`R0`) or `128` (`R1`). | Any value, modulo 256. The vector is 128 consecutive elements of `Ra` and can cross from `R0` to `R1`. |
| `mask_type = 3` | The pad value is `NaN`. | Configuration error. |
