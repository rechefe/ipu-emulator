# AaQ Stage

## 1. Overview

The AaQ (Activation and Quantization) stage perfome the following tasks:

- **Activation:** The stage applies activation functions and special functions
  to each element of the fp32 vector. ReLU and GeLU are examples of the
  activation functions.
- **Quantization:** The stage changes each fp32 element into the NX8 format.
  The quantization method is fixed or dynamic.
- **Store:** After the quantization, the stage writes the result to XMEM.

> **Note:** A configurable lookup table (LUT) implements some of the activation
> functions and special functions.

## 2. Block Diagram

```mermaid
block-beta
    columns 7
    block:INS:1
        columns 1
        ACC_OUT["input data<br>128 x fp32"]
        INSTR["Instruction"]
    end
    space
    block:STAGE:3
        columns 3
        TITLE["AaQ Stage"] space:2
        ACT["Activation"] space QUANT["Quantization"]
        space:3
        LUT["LUT"] space:2
    end
    space
    block:OUTS:1
        columns 1
        NX_OUT["output data<br>128 x NX"]
        XMEM_ADDR["XMEM write<br>address"]
    end

    ACC_OUT --> ACT
    INSTR --> ACT
    ACT --> QUANT
    ACT <--> LUT
    QUANT --> NX_OUT
    QUANT --> XMEM_ADDR

    classDef io fill:none,stroke:none
    classDef blk fill:#2e9e8c,stroke:#1a7060,color:#fff
    classDef title fill:none,stroke:none,font-weight:bold
    class ACC_OUT,INSTR,NX_OUT,XMEM_ADDR,INS,OUTS io
    class ACT,QUANT,LUT blk
    class TITLE title
    style STAGE fill:none,stroke:#888,stroke-width:2px
```

## 3. Interfaces

| Name | Direction | Description |
|------|-----------|-------------|
| input data | Input | The 128-element fp32 vector from the ACC stage. |
| Instruction | Input | The operation that the stage does on the input data. |
| output data | Output | The 128-element NX vector. |
| XMEM write address | Output | The XMEM address to which the stage writes the output data. |


## 5. AaQ Operations

### 5.0 Activate and Quantize (`ACTIVATE.QUANTIZE`)

Activation, quantization and store happen in a single instruction; there is no
separate activate-only or quantize-only instruction. It applies an
element-wise activation function to every element of `input data`.

**Syntax:** `ACTIVATE.QUANTIZE activation_fn, dstructure,quantizion_format,table_id,table_offset`

**Operands:** *(the Type column links to the [operand type reference](../operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `activation_fn` | [`Imm`](../operand-types.md#activationfn) | enum that names the activation.  |
| `dstructure` | [`CrIdx`](../operand-types.md#dstructurecridx) | `CR0`–`CR15`. This register supplies a cr that discribe the stractrue of the 128 element vector. |
| `quantizion_format` | [`CrIdx`](../operand-types.md#cridx) | `CR0`–`CR15`. This register supplies the quantization format. |
| `table_id` | [`CrIdx`](../operand-types.md#cridx) | `CR0`–`CR15`. This register supplies the ID of the destination table. |
| `table_offset` | [`LrIdx`](../operand-types.md#lridx) | `LR0`–`LR15`. This register supplies the offset in the destination table. |

**`activation_fn` enum:**

| Encoding | Name | Formula | Notes |
|----------|------|---------|-------|
| 0 | `identity` | `f(x) = x` | Pass-through; no change. |
| 1 | `relu` | `f(x) = max(0, x)` |  |
| 2 | `relu6` | `f(x) = min(max(0, x), 6)` | Clipped ReLU. |
| 3 | `reciprocal` | `f(x) = 1/x` | implumented using LUT + dedicated logic to calculates the function . If x is in the non negative subnormal range of fp32 add epslion = 2^-127 if in the negative add -2^-127 , if x is `inf` f(x) = 0 . |
| 4 | `rsqrt` | `f(x) = 1/√x` | implumented using LUT + dedicated logic to calculates the function . If x is in the non negtive subnormal range of fp32 add epslion = 2^-127 under the sqrt (i.e 1/sqrt(x+epsilon)), if x is `inf` f(x) = 0 if x is a `Negative number` f(x) = `NaN`. |
| 5 | `exp2` | `f(x) = 2^x` |implumented using LUT + dedicated logic to calculates the function. used only for `non positive numbers` if x = -inf f(x) = 0 |
| 6 | `generic` | `f(x) = Generic(x)` | The LUT and dedicated logic calculate the function. If `x` is not larger than `2^max_exp`, the LUT gives the result. If `x` is smaller than `2^max_exp`, `range_mode` selects the result: `f(x) = x` (identity) or `f(x) = pos_c` (a constant). `max_exp` (`max_exp_num`), `pos_c` and `range_mode` are in the LUT metadata (section 5.2.2). |

The LUT for `generic` has three versions. The drawings that follow are a
reference for each version.

> **Note:** `pos_c` is the constant for the positive side of the x axis.
> `neg_c` is the constant for the negative side of the x axis. The names give
> the side of `x`, not the sign of the value. The value of each constant can
> be positive or negative.

**Version 1:** All 256 LUT entries are for positive `x`. Out of the LUT range,
`range_mode` selects the result: `x` (identity) or a constant.

$$
f(x) =
\begin{cases}
\mathrm{LUT}(x) & 0 \le x < 2^{\mathrm{max\_exp}} \\
x \ \text{or} \ \mathrm{pos\_c} & x \ge 2^{\mathrm{max\_exp}} \\
x \ \text{or} \ \mathrm{neg\_c} & x < 0
\end{cases}
$$

![Reference drawing of the generic activation function, version 1](assets/aaq-generic-activation.svg)

**Version 2:** All 256 LUT entries are for negative `x`. Out of the LUT range,
`range_mode` selects the result: `x` (identity) or a constant.

$$
f(x) =
\begin{cases}
\mathrm{LUT}(x) & -2^{\mathrm{max\_exp}} < x \le 0 \\
x \ \text{or} \ \mathrm{neg\_c} & x \le -2^{\mathrm{max\_exp}} \\
x \ \text{or} \ \mathrm{pos\_c} & x > 0
\end{cases}
$$

![Reference drawing of the generic activation function, version 2](assets/aaq-generic-activation-v2.svg)

**Version 3:** The LUT entries are divided into 128 entries for negative `x`
and 128 entries for positive `x`. Out of the LUT range, `range_mode` selects
the result: `x` (identity) or a constant.

$$
f(x) =
\begin{cases}
\mathrm{LUT}(x) & -2^{\mathrm{max\_exp}} < x < 2^{\mathrm{max\_exp}} \\
x \ \text{or} \ \mathrm{pos\_c} & x \ge 2^{\mathrm{max\_exp}} \\
x \ \text{or} \ \mathrm{neg\_c} & x \le -2^{\mathrm{max\_exp}}
\end{cases}
$$

![Reference drawing of the generic activation function, version 3](assets/aaq-generic-activation-v3.svg)


The function is selected via `function_type` and applied to each FP32
element. `identity`, `relu`, and `relu6` are computed directly; `exp2`,
`reciprocal`, `rsqrt`, and `generic` are evaluated through the LUT, so the
selected function must already be loaded there (section 5.2).



`FP32_MAX` is the largest finite FP32 value, `0x7F7FFFFF`
(`(2 - 2^-23) · 2^127 ≈ 3.4028235e38`). The guards never produce `inf`.

There is a single LUT holding one function at a time: whichever was last
loaded by `LOAD` (section 6.3). For the LUT-based types (`exp2`, `generic`,
`reciprocal`, `rsqrt`), `function_type` selects only the guards in the
pseudocode above; software must load the matching function before issuing
`ACTIVATE.QUANTIZE`.

Supported function types: activation and special functions grouped onto a
single field:




Examples of functions run through `generic` (encoding 6), each loaded into and called through the same LUT:


### 5.1 Quantization Algorithm

After activation (section 5.0), each activated FP32 element `a` (IEEE-754 single
precision: 1-bit sign `S`, 8-bit exponent `e`, 23-bit mantissa) is quantized
to the format selected by `format` (section 3.3): a sign bit present only if
`format[7] = 1` (signed; omitted when unsigned), `fe` exponent bits
(`format[6:4]`), and `fm` mantissa bits derived as the leftover
`fm = W - fe - sign_bit`, where `W = format[3:0]` (stored directly, no
offset) is the total element width. Since the mantissa is the remainder of the width, `sign + fe + fm`
always totals exactly `W`; when `W < 8` the leftover low-order bits of the
8-bit quantized element are zero-padded. `fe` is at minimum 1 bit; `fm` may be
0 bits. FP32 inputs are always treated as normalized (implicit leading 1);
subnormal inputs are not specially handled.

The scale factor `s` (8 bits, `e8m0`: exponent only, no mantissa) is the
batch's shared scale, computed as the maximum raw exponent across the
active elements of the batch (`active[i]`, section 6), before quantization.
Invalid and masked elements do not contribute to `s`:

```text
s = max(e[i] for i in 0..127 if active[i])   // active[i]: section 6
```

For each element, define the exponent distance from the batch scale:

```text
E = -(e - s)   // = s - e
```

Since `s` is the batch maximum, `E >= 0` for every element, with `E = 0` at
the batch-max element(s) and `E` growing as an element's magnitude shrinks
relative to the batch max. The output exponent and mantissa fields are then:

```text
if 0 <= E < 2^fe - 1:                // representable directly in fe bits
    Exp = E
    M   = RTN(1.M)                   // round the 23-bit mantissa (implicit leading 1) to fm bits

    if M == 2^fm:                    // edge 1: 1.11...1 rounded up to 10.00...0
        if E > 0:
            Exp = E - 1              // the carry moved the value one binade up
            M   = 0
        else:                        // E = 0: no binade above the batch scale
            Exp = 0
            M   = 2^fm - 1           // saturate at the largest representable value

else:                                // E exceeds what fe bits can represent
    Exp = 2^fe - 1                   // exponent field saturates at its max value
    M   = RTN(1.M >> [E - (2^fe - 1) + 1])  // extra right-shift preserves magnitude instead of flushing to zero

    if M == 2^fm:                    // edge 2: 0.11...1 rounded up to 1.00...0
        Exp = 2^fe - 2               // the smallest directly-encoded exponent
        M   = 0
```

Both edges are the same event — `RTN` carrying out of the `fm` mantissa bits —
resolved differently per branch. In the direct branch the carry means the value
moved one binade up, so the exponent decrements and the mantissa clears; at
`E = 0` there is no binade above the batch scale, so the element saturates at
the largest representable value instead of wrapping. In the clamp branch the
carry turns the shifted fraction into `1.00...0`, which is no longer a
saturated value but the smallest directly-encoded one, so it is re-encoded as
`Exp = 2^fe - 2`, `M = 0`.

`RTN` = round to nearest; an exact half rounds **up** (away from zero in
magnitude), the same rule as section 5.1.1. Sign `S` (when present, `format[7] = 1`) is passed
through unchanged. The final 8-bit quantized element is
`{S?, Exp, M, 0-pad}`: `S`, `Exp` (`fe` bits), and `M` (`fm` bits) packed at
the high end, zero-padded at the low end to fill 8 bits. The write payload
also carries the batch `Scale` (`s`) and the `dynamic_exponent` bit, as
described in section 3.2. `format` is not written; it is table-level.

> **Note:** the `S`/`Exp`/`M` encoding above is the same in both modes,
> and `s` is computed the same way (batch max, as shown above) regardless
> of `format[8]`. `format[8] = 1` (dynamic) restricts
> `format` to exactly two supported formats, both signed and both 8 bits
> wide (`format[3:0] = 8`): `e2m5` and `e1m6` — that pair is exactly
> `fe = 2` and `fe = 1` at `W = 8`, signed, since `fm` is the leftover
> (section 3.3). Only `fe` is chosen; the rule in section 5.1.1 makes
> that choice per block.

A **block** is the full 128-element batch of one `ACTIVATE.QUANTIZE`: one
scale `s` and one `dynamic_exponent` decision per instruction, computed over
the block's active elements (section 6). `partition` does not split it into smaller
blocks.

#### 5.1.1 Dynamic Exponent-Field Selection (`fe = 1` vs `fe = 2`)

Applies only when `format[8] = 1`. The dynamic decision selects **the exponent
field width alone**: `fe = 1` or `fe = 2`. The input `format[6:4]` is ignored,
and the choice is written to the row's `dynamic_exponent` bit (section 3.2). The mantissa is not
chosen independently — it is the leftover `fm = W - fe - sign_bit` (section
3.3), so moving from `fe = 1` to `fe = 2` always buys one more exponent bit at
the cost of exactly one mantissa bit, whatever `W` and `sign_bit` are. Each
block shares one scale `s` and the quantizer picks one of the two splits per
block:

| Split | Exponent bits | Mantissa bits | Encodable `E` | Step at `E` |
|---|---|---|---|---|
| `fe1` | 1 | `fm1 = W - 1 - sign_bit` | 0, 1 | `2^(s - E - fm1)`, clamped at `E >= 1` |
| `fe2` | 2 | `fm1 - 1` | 0, 1, 2, 3 | `2^(s - E - fm1 + 1)`, clamped at `E >= 3` |

`fe1` is one bit finer near the top of the block. `fe2` reaches two binades
further down before it clamps. The rule below decides which trade is better for
the block actually being quantized. It depends on `fm1` only through the two
multipliers, so it holds for any `W`/`sign_bit` pairing, not just the signed
8-bit one.

##### Definitions

- `s`: shared block scale exponent, selected by the same block scale procedure
  used for static quantization (section 5.1).
- `e`: exponent of the individual element.
- `E = s - e`. Always non-negative (section 5.1). `E = 0` means the scale
  represents the element exactly.
- `fm1 = W - 1 - sign_bit`: the last mantissa bits under `fe1`. `fe2` always has exactly
  one fewer. For the signed, 8-bit case `fm1 = 6`.
- `err1`: truncation error of the element under `fe1`.
- `err2`: truncation error of the element under `fe2`.

##### Element classification

Each element contributes to at most one accumulator:

| Condition | Accumulator | Meaning |
|---|---|---|
| `E = 0` | A | `fe1` is finer here. A measures the cost of choosing `fe2`. |
| `E = 1` | none | Don't care. |
| `1 < E <= fm1 + 2` | B | `fe2` is finer here. B measures the cost of choosing `fe1`. |
| `E > fm1 + 2` | none | Out of range for both splits. |

Elements at `E = 1` are don't cares. They enter neither sum and have no effect
on the decision.

##### Per-element terms

```text
A = RTN( |err2^2 - err1^2| * 2^(2*fm1) )
B = RTN( |err2^2 - err1^2| * 2^(2*(fm1 + 1)) )
```

`RTN` is round to nearest integer, and an exact half rounds **up**: `0.5 -> 1`,
`1.5 -> 2`, `2.5 -> 3`. 

##### Decision

```text
if sum(B) - 4 * sum(A) < 0:  choose fe1        // dynamic_exponent = 0
else:                        choose fe2        // dynamic_exponent = 1
```

The constant 4 is a unit reconciliation, not a tuning parameter. The B
multiplier `2^(2*(fm1 + 1))` is exactly four times the A multiplier
`2^(2*fm1)`, so scaling A by 4 returns both accumulators to a common error.

##### Both estimators are unbiased

Assume mantissa bits are uniformly distributed, so `r` — the finer split's
error as a fraction of its own step — is uniform on `[0, 1)` and the dropped
bit is 1 with probability one half.

| Term | Pre-RTN value | Distribution | Outcomes | E[RTN] | E[pre-RTN] |
|---|---|---|---|---|---|
| A | `1 + 2r` | uniform on `[1, 3)` | 1, 2, 3 at 0.25, 0.5, 0.25 | 2 | 2 |
| B | `(1 + 2r) / 4` | uniform on `[0.25, 0.75)` | 0, 1 at 0.5, 0.5 | 0.5 | 0.5 |

In both rows the rounded mean equals the unrounded mean exactly. Including the
zero case, `E[A] = 1` and `E[B] = 0.25`. Neither sum drifts as the block length
grows, so rounding contributes noise but never a systematic tilt toward one
format.

##### Worked example

Block of ten values, `s = 0`, `fm1 = 4` (signed, `W = 6`). The A multiplier is
`2^8 = 256` and the B multiplier is `2^10 = 1024`. `fm1 = 4` keeps the numbers
short; for the signed 8-bit case `fm1 = 6`, the multipliers are `2^12` and
`2^14`, and the B region is `1 < E <= 8`.

| # | v | e | E | Acc | err1 | err2 | \|err2^2 - err1^2\| | Pre-RTN | A | B |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.83 | 0 | 0 | A | 0.0175 | 0.0800 | 6.09375e-3 | 1.560 | 2 | |
| 2 | 1.21 | 0 | 0 | A | 0.0225 | 0.0850 | 6.71875e-3 | 1.720 | 2 | |
| 3 | 1.57 | 0 | 0 | A | 0.0075 | 0.0700 | 4.84375e-3 | 1.240 | 1 | |
| 4 | 0.94 | -1 | 1 | none | don't care | don't care | | | | |
| 5 | 0.67 | -1 | 1 | none | don't care | don't care | | | | |
| 6 | 0.41 | -2 | 2 | B | 0.00375 | 0.00375 | 0 | 0 | | 0 |
| 7 | 0.2345 | -3 | 3 | B | 0.01575 | 0.000125 | 2.48047e-4 | 0.254 | | 0 |
| 8 | 0.1780 | -3 | 3 | B | 0.02175 | 0.006125 | 4.35547e-4 | 0.446 | | 0 |
| 9 | 0.0920 | -4 | 4 | B | 0.02950 | 0.013875 | 6.77734e-4 | 0.694 | | 1 |
| 10 | 0.0555 | -5 | 5 | B | 0.02425 | 0.008625 | 5.13672e-4 | 0.526 | | 1 |

`sum(A) = 5`, `sum(B) = 2`.

`2 - 4 * 5 = -18`, which is negative, so the block selects **`fe1`**.



### 5.2 Function Estimation LUT

The stage holds a function estimation LUT of **256 entries × 17 bits**,
plus a **75-bit metadata block** that configures range handling. Both are
filled by `LOAD` (section 6.3), which writes one **segment** per instruction;
the 3-bit `lut_addr` selects the segment.

#### 5.2.1 Segment Transfer

A segment is **1024 bits**, sourced from the **lower 32 words** of the
accumulator, `r_acc[31:0]` (32 words × 32 bits = 1024 bits). The upper words
`r_acc[127:32]` are ignored by `LOAD`. The 1024-bit payload is the little-endian
concatenation of those words:

```text
payload[1023:0] = {r_acc[31], r_acc[30], ..., r_acc[1], r_acc[0]}
                                          // payload[32*j +: 32] == r_acc[j]
```

The table itself is a **4,352-bit flat bitstream** — not split per word, and
not reset per segment: `LOAD` transfers one 1024-bit slice of that stream
per instruction, and a table entry may straddle the boundary between two
segments (the same way the metadata field straddles word boundaries,
section 5.2.2). `lut_addr` selects which slice:

| `lut_addr` | Contents | Interpretation |
|------------|----------|----------------|
| `0` | Table bits `[1023:0]` | 1st 1024-bit slice of the flat table bitstream. |
| `1` | Table bits `[2047:1024]` | 2nd slice. |
| `2` | Table bits `[3071:2048]` | 3rd slice. |
| `3` | Table bits `[4095:3072]` | 4th slice — the table's first 4,096 bits are now fully loaded. |
| `4` | Table bits `[4351:4096]` (the last 256 bits) + metadata | **Mixed.** `payload[255:0]` = the table's last 256 bits; `payload[330:256]` = the 75-bit metadata block (section 5.2.2); `payload[1023:331]` is unused/reserved. |
| `5`–`7` | Invalid | Not a valid `LOAD` encoding. |

Five `LOAD`s (`lut_addr` `0`–`4`) are required to fully program the LUT and
its metadata. Only the last of them (`lut_addr = 4`) is partially reserved —
the first four are fully packed with table bits (256 entries × 17 bits =
4,352 bits total).

#### 5.2.2 Metadata Block

The metadata occupies bits `[330:256]` of the `lut_addr = 4` segment's
1024-bit payload (i.e. local bits `[74:0]` of the metadata field itself,
offset by the 256 bits of leftover table data ahead of it); the remaining
bits `[1023:331]` are unused. Bits-to-word mapping follows
`payload[32*j +: 32] == r_acc[j]` (section 5.2.1), so absolute bit `256 + m`
(for local metadata bit `m`) lands in `r_acc[(256+m) / 32][(256+m) % 32]`.
Because the field is flat, it straddles word boundaries — for example
`max_exp_num` is `r_acc[8][7:0]` and `neg_c` spans `r_acc[8][31:8]` together
with `r_acc[9][7:0]`.

| Bits (local, within metadata) | Name | Consumed by | Meaning |
|------|------|-------------|---------|
| `[7:0]` | `max_exp_num` | `Quantactivation` | Largest **biased** FP32 exponent still inside the LUT range. `exp > max_exp_num` → lane flagged out-of-range. |
| `[39:8]` | `neg_c` | `pack_lutout_activ` | FP32 constant substituted for negative out-of-range lanes. |
| `[71:40]` | `pos_c` | `pack_lutout_activ` | FP32 constant substituted for positive out-of-range lanes. |
| `[72]` | `asymetric` | `Quantactivation` | Table domain + address form. |
| `[74:73]` | `range_mode` | `pack_lutout_activ` | Out-of-range policy. |

> **TBD:** the encodings of `asymetric` (which table domain and address form
> each value selects) and of `range_mode` (which out-of-range policy each of
> the 4 values selects) are not yet specified.

## 6. ISA-AaQ: Instruction Reference

The opcode enum
(`aaq_inst_opcode_t`, package `ipu_instr_pkg`) is generated from
[`instruction_spec.py`](../../../src/tools/ipu-common/src/ipu_common/instruction_spec.py)
by [`gen_codegen.py`](../../../src/tools/ipu-as-py/src/ipu_as/gen_codegen.py).

The AaQ slot is resolved by CTRL and forwarded down the pipeline;
the stage does not read the CR/LR register files itself (see the
Control Stage spec, section 5). The active elements are determined by
`ACTIVATE.QUANTIZE`'s mandatory `cr_idx` operand (supplying `valid_elements`) together
with `partition` and `partition_mask` (section 3.1), at cycle start. The 128
elements are split into `partition` groups of `G = 128/partition` elements;
the rightmost `partition_mask` groups are masked entirely, and in every
remaining group the first `valid_elements` elements (from the left of the
group) are active:

```text
G = 128 / partition                                   // group size
for i in 0..127:
    g = i / G                                         // group index
    j = i % G                                         // position within group
    active[i] = (g < partition - partition_mask) and (j < valid_elements)
```

Example: `partition = 4` (groups of 32), `partition_mask = 1`,
`valid_elements = 10` → active elements `0–9`, `32–41`, `64–73`; group 3
(`96–127`) is masked. 30 elements are active. There is no implicit default register; `cr_idx` must always
be named explicitly (any `CR0`-`CR15`; `CR15` remains the conventional choice
but is never assumed).

### 6.1 `NOP`: No Operation

- **Summary:** No operation for the AaQ slot; performs no state changes and no memory write.
- **Syntax:** `NOP`
- **Operands:** none.

### 6.2 `ACTIVATE.QUANTIZE`: Activate and Quantize

- **Summary:** Apply an element-wise activation function to the active elements of `r_acc`, quantize the result, and write the resulting 8-bit values, scale factor, and `dynamic_exponent` bit to `POST_AAQ_REG`. `STR_POST_AAQ_REG` performs the XMEM write in the STORE slot. `identity`, `relu`, and `relu6` are computed directly; `exp2`, `generic`, `reciprocal`, and `rsqrt` are evaluated through the function pre-loaded into the LUT by `LOAD` (section 6.3), as described in section 5.0. `r_acc` is not modified.
- **Syntax:** `ACTIVATE.QUANTIZE function_type, cr_idx`
- **Operands:**
  - `function_type`: activation/special-function keyword (see section 5.0): `identity`, `relu`, `relu6`, `generic`, `reciprocal`, `rsqrt`, `exp2`.
  - `cr_idx`: `CR0`…`CR15`, dstructure register supplying `valid_elements` (must be given explicitly; no implicit default).
- **Operation:**
  ```text
  for i in 0..127:                                     // active[i]: section 6
      if active[i]:
          activated[i] = activate(function_type, r_acc[i]) // section 5.0: direct or via LUT
          aaq_out.elements[i] = quantize(activated[i])     // section 5.1
      else:
          aaq_out.elements[i] = 0
  aaq_out.scale = s                                    // section 5.1
  aaq_out.dynamic_exponent = format[8] ? (fe == 2) : 0 // section 5.1.1
  POST_AAQ_REG = aaq_out                              // 1033 bits, section 3.2
  ```
- **Example:** `ACTIVATE.QUANTIZE relu, CR15;;`

### 6.3 `LOAD`: Load Function Estimation LUT Segment

- **Summary:** Fill one 1024-bit segment of the function estimation LUT (section 5.2) from the lower 32 words of `r_acc`. The 3-bit `lut_addr` selects the target segment: `0`–`3` each load a 1024-bit slice of the flat 4,352-bit table bitstream, `4` loads the table's last 256 bits plus the metadata block (section 5.2.1). Performs no XMEM write and does not modify `r_acc`.
- **Syntax:** `LOAD lut_addr`
- **Operands:**
  - `lut_addr`: LUT segment index, `0`–`4` (`0`–`3` = table slices, `4` = last table bits + metadata). `5`–`7` are invalid.
- **Operation:**
  ```text
  payload[1023:0] = {r_acc[31], ..., r_acc[0]}        // 32 words x 32 bits

  if lut_addr <= 3:                                  // full table slice
      TABLE_BITS[1024*lut_addr +: 1024] = payload     // flat bitstream, section 5.2.1
  else if lut_addr == 4:                             // last table bits + metadata
      TABLE_BITS[4351:4096] = payload[255:0]
      {range_mode, asymetric, pos_c, neg_c, max_exp_num} = payload[330:256]
      // payload[1023:331] unused; field layout in section 5.2.2
  ```
- **Example:** `LOAD 4;;` (load the metadata block)

### 6.4 Summary Table

| Slot | Mnemonic | Operands | One-line Effect |
|------|----------|----------|-----------------|
| AaQ | `NOP`               | -                       | no state change |
| AaQ | `ACTIVATE.QUANTIZE` | `function_type, cr_idx` | `aaq_out.elements[i] = active[i] ? quantize(activate(function_type, r_acc[i])) : 0`, `aaq_out.scale/dynamic_exponent` set, `POST_AAQ_REG = aaq_out`; `active[i]` from `partition`/`partition_mask`/`valid_elements` (section 6) |
| AaQ | `LOAD`              | `lut_addr`              | `TABLE_BITS[1024*lut_addr +: 1024] = r_acc[31:0]` for `lut_addr` `0`-`3`; `4` = table bits `[4351:4096]` + metadata block (section 5.2.1) |


### 3.3 `format` Field Layout

`format` is 9 bits. It is a property of the destination table (shared by
every row of that table, across all of its XMEM banks) and is not written to
XMEM:

| Bits | Name | Description |
|------|------|-------------|
| `[8]` | `dynamic` | Quantization mode: `0` = static, `1` = dynamic (section 5.1.1). |
| `[7]` | `sign` | `0` = unsigned, `1` = signed. |
| `[6:4]` | `exp_bits` (`fe`) | Number of exponent bits, `1`–`7`. `0` is reserved/invalid. |
| `[3:0]` | `width` | Total element width `W`, stored directly (no offset): value `1`–`8`. `0` is reserved/invalid. |

Bits `[7:0]` have the same meaning in both modes. In dynamic mode
(`format[8] = 1`) the exponent width is chosen per row (section 5.1.1) and
recorded in that row's `dynamic_exponent` bit, so `format[6:4]` is ignored.

The mantissa is **the remainder** — what is left of the element width once the
sign and exponent bits are taken out:

```text
sign_bit = format[7]                 // 0 or 1
fe       = format[6:4]               // exponent bits
W        = format[3:0]               // total element width, 1..8 bits (stored directly)
fm       = W - fe - sign_bit         // mantissa bits (the leftover)
```

`fm` must be `>= 0`, i.e. `W >= fe + sign_bit`; encodings that violate this are
invalid. `fm = 0` is legal (no mantissa bits). `W <= 8` always, so a quantized
element always fits in its 8-bit output slot; when `W < 8` the unused
low-order bits are zero-padded.

Example: signed, 2 exponent bits, 8-bit width (`e2m5`) is
`format = {1, 3'd2, 4'd8}` → `fm = 8 - 2 - 1 = 5`.