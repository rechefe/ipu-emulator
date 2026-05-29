# AAQ Stage

## 1. Purpose

The AAQ (Activation Aggregation and Quantization) stage collapses the 128-lane accumulator
into scalar activation values and/or quantizes the accumulator into an FP8
vector for output. It produces:

- Scalar results in `aaq0`–`aaq3` via `agg` and `agg.first`.
- A 128-byte `aaq_result` vector via `aaq` (any 8-bit format: INT8 or FP8 e(x)m(7-x)).

## 2. Block Diagram

```mermaid
flowchart LR
    mult_stage:::blue
    acc_stage:::blue
    ACC(["r_acc 128x32bit"]):::yellow
    OUT(["aaq_result 128x8bit || 8bit scale | 4bit repr_type"]):::red
    ACT["Activation"]:::teal
    QUANT["Quantization"]:::teal
    RED["Adder tree / Max tree"]:::teal
    PFN["Post_func"]:::teal
    RF["RF regs 4x32bit aaq0-aaq3 with shadow regs 17 bits"]:::purple

    ACC -->|128x32| ACT
    ACC -->|128x32| RED
    ACT -->|128x32| QUANT
    ACT --> |128x32| RED
    QUANT -->|128x8| OUT
    mult_stage --> |128x32| acc_stage
    acc_stage --> |128x32| ACC
    ACC --> RF
    RED --> PFN
    PFN -->|FP32| RF
    RF -.->|feedback 32bit| RED
    RF -.->|feedback 17 bit| mult_stage
    RF -.->|feedback 32bit| acc_stage

    subgraph LEGEND["Legend"]
        L_blue["Stages"]:::blue
        L_teal["AAQ Main Blocks"]:::teal
        L_purple["Register File"]:::purple
        L_yellow(["Input Data"]):::yellow
        L_red(["Output Data"]):::red
    end

    classDef blue fill:#4a80c4,stroke:#2a5090,color:#fff
    classDef teal fill:#2e9e8c,stroke:#1a7060,color:#fff
    classDef purple fill:#7b5ea7,stroke:#5a3d8a,color:#fff
    classDef yellow fill:#e6b800,stroke:#b38a00,color:#000
    classDef red fill:#c0392b,stroke:#922b21,color:#fff
```

## 3. Interfaces

### 3.0 Black Box Diagram

```
                         ┌──────────────────────────────────────┐
              clk  ─────>│                                      │
              rst  ─────>│                                      ├────> aaq_result        [1035:0]
            valid  ─────>│                                      │
               op  ─────>│                                      ├────> aaq_rf_to_acc[0..3] [31:0]
            r_acc  ─────>│             AAQ Stage                │
         agg_mode  ─────>│                                      ├────> aaq_rf_to_mult[0..3][16:0]
          post_fn  ─────>│                                      │
           cr_idx  ─────>│                                      │
       act_cr_idx  ─────>│                                      │
       aaq_rf_idx  ─────>│                                      │
          dtype  ─────>│                                      │
   valid_elements  ─────>│                                      │
    full_xmem_row  ─────>│                                      │
                         └──────────────────────────────────────┘
```

### 3.1 Inputs

| Name | Type and Direction | Description |
|------|--------------------|-------------|
| `clk` | `input logic` | Clock signal. |
| `rst` | `input logic` | Synchronous reset. |
| `valid` | `input logic` | Stage enable. When deasserted(valid = 0), the stage executes `aaq_nop` regardless of `op`. |
| `op` | `input logic [2:0]` | Selects the AAQ operation: `agg`=1, `agg.first`=2, `aaq`=3. Sampled only when `valid`=1. |
| `r_acc` | `input logic [127:0][31:0]` | 128-lane accumulator (128 × 32-bit). |
| `agg_mode` | `input logic [0:0]` | Aggregation select: 0 = sum, 1 = max. |
| `post_fn` | `input logic [1:0]` | Post-function select (see §8). |
| `cr_idx` | `input logic [3:0]` | CR register index used by `value_cr` post function. |
| `act_cr_idx` | `input logic [3:0]` | CR register index whose value selects the activation function (see §7.0). |
| `aaq_rf_idx` | `input logic [1:0]` | AAQ register index (0–3). |
| `dtype` | `input logic [2:0]` | Global data type forwarded from outside configuration; governs lane interpretation for all AAQ operations. Must be `DType.INT8` (0) for `aaq`. |
| `valid_elements` | `input logic [7:0]` | Active lane count. Sourced from `CR15.valid_elements` (bits [7:0] of the CR15 dstructure register). Ignored when `full_xmem_row` = 1. |
| `full_xmem_row` | `input logic [0:0]` | Lane-count override. 1 = always use all 128 lanes (ignore `valid_elements`); 0 = use `valid_elements`. Applies to `AGG`, `AGG.FIRST`, `AAQ`, and `ACTIVATE`. |

### 3.2 Outputs

| Name | Type and Direction | Description |
|------|--------------------|-------------|
| `aaq_result` | `output logic [1035:0]` | Quantized output: 128 × 8-bit lanes (1024 bits) plus 12 bits of metadata: bits [1035:1028] = 8-bit scale factor, bits [1027:1024] = 4-bit representation type (e.g. INT8, e6m1). |
| `aaq_rf_to_acc[0..3]` | `output logic [31:0]` | Full 32-bit view of the AAQ RF registers (`aaq0`–`aaq3`) fed back to the ACC stage. |
| `aaq_rf_to_mult[0..3]` | `output logic [16:0]` | Same AAQ RF registers as `aaq_rf_to_acc`, exposed to the MULT stage with only the lower 17 bits; the MULT stage does not require the full 32-bit precision. |

## 4. Parameters

| Name | Default | Description |
|------|---------|-------------|
| `LANES` | `128` | Number of accumulator lanes. |
| `ACC_LANE_WIDTH` | `32` | Bits per accumulator lane. |
| `AAQ_REG_COUNT` | `4` | Number of AAQ scalar registers (`aaq0`–`aaq3`). |
| `AAQ_RESULT_BYTES` | `128` | Byte width of `aaq_result` output vector. |
| `AGG_MODE_COUNT` | `2` | Aggregation modes: 0 = sum, 1 = max. |
| `POST_FN_COUNT` | `5` | Post functions: 0 = identity, 1 = value_cr, 2 = inv, 3 = inv_sqrt, 4 = sqrt. |

## 5. Data and Register Model

- `r_acc` is 512 bytes (128 × 32-bit lanes). Lanes are always FP32.
- `aaq0`–`aaq3` are 32-bit registers. They store float32 bit patterns.
- `aaq_result` is produced only by `aaq`. It contains 128 × 8 bit quantized
  values (1024 bits) plus 12 bits of metadata appended by the Quantization
  block: bits [11:4] = 8-bit scale factor, bits [3:0] = 4-bit representation
  type (INT8, e6m1, e5m2, …). It is written to XMEM via `xmem.store_aaq_result`.

## 6. Disclaimers

- The AAQ slot executes once per VLIW cycle.
- Slot execution order within a VLIW word: CTRL → MULT → ACC → **AAQ** → STR.
- `aaq_nop` performs no state changes.

## 7. AAQ Operations

### 7.0 Activation

The Activation block applies an element-wise function to every valid lane of `r_acc` before the result is passed downstream — both to the Adder/Max tree (`agg`, `agg.first`) and to the Quantization block (`aaq`).

The function is selected at runtime by reading the CR register indexed by `act_cr_idx`: `activation_fn = cr[act_cr_idx]`.

```text
// Applied to all valid lanes before aggregation or quantization
activation_fn = cr[act_cr_idx]
n             = 128 if full_xmem_row else min(valid_elements, 128)
activated[i]  = activation_fn(r_acc[i])   for i in 0..n-1
```

Supported activation functions:

| Encoding | Name | Formula | Notes |
|----------|------|---------|-------|
| 0 | `identity` | `f(x) = x` | Pass-through; no transform. |
| 1 | `relu` | `f(x) = max(0, x)` | Most common non-linearity. |
| 2 | `relu6` | `f(x) = min(max(0, x), 6)` | Clipped ReLU; used in MobileNet. |
| 3 | `sigmoid` | `f(x) = 1 / (1 + e^−x)` | Squashes to (0, 1). |
| 4 | `tanh` | `f(x) = (e^x − e^−x) / (e^x + e^−x)` | Squashes to (−1, 1). |
| 5 | `gelu` | `f(x) = x · Φ(x)` | Φ = standard normal CDF; used in BERT/GPT. |
| 6 | `softplus` | `f(x) = ln(1 + e^x)` | Smooth approximation of ReLU. |
| 7 | `elu` | `f(x) = x if x ≥ 0 else α·(e^x − 1)` | Smooth negative region; reduces vanishing gradient. |
| 8 | `exp2` | `f(x) = 2^x` | Used for dequantization, softmax and attention scaling. |
| 9 | `reciprocal` | `f(x) = 1/x` (0 if x = 0) | Multiplicative inverse; useful for normalization. |
| 10 | `rsqrt` | `f(x) = 1/√x` (0 if x ≤ 0) | Reciprocal square root; used in layer normalization. |

### 7.1 Aggregate (`agg`)

**Assembly syntax:** `AGG agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row`

The active lane count is controlled by `full_xmem_row`: when `1`, all 128 lanes are used; when `0`, the count is read implicitly from `CR15.valid_elements` (bits [7:0] of the CR15 dstructure register).

```text
// r_acc lanes are always FP32
n      = 128 if full_xmem_row else min(valid_elements, 128)
values = [activation(r_acc[i]) for i in 0..n-1]

// Adder tree / max tree (feedback from RF included in max mode)
if agg_mode == sum:
    raw = sum(values)                          // FP32 scalar; input to post_fn
else:  // max
    raw = max(values + [aaq[aaq_rf_idx]])      // FP32 scalar; input to post_fn; includes RF feedback

out = post_fn(raw, cr[cr_idx])
aaq[aaq_rf_idx] = pack32(out)                  // stored as float32
```

Notes:
- Only lanes `0..n-1` are fed into the adder/max tree (where `n = 128` if `full_xmem_row=1`, else `min(CR15.valid_elements, 128)`); lanes beyond that index are ignored.
- The activation function applied is the one selected for the current operation (see §7.0).
- In `max` mode the current value of the target AAQ register is fed back from RF into
  the adder/max tree. Use `agg.first` to avoid contamination from an uninitialised register.

#### Adder / Max Tree — Conceptual Structure

```mermaid
flowchart TD
    V0["v[0]"] & V1["v[1]"] --> PA["＋ / max"]
    V2["v[2]"] & V3["v[3]"] --> PB["＋ / max"]
    VMORE["v[4] … v[127]"] --> PC["＋ / max  ×62"]

    PA & PB & PC --> MID["⋮  5 more levels  ⋮"]
    MID --> LEAF["＋ / max"]

    RF["aaq[aaq_rf_idx]  —  RF feedback "] --> RFMAX["＋ / max"]
    LEAF --> RFMAX

    RFMAX --> RAW(["post_fn output -> aaq[aaq_rf_idx]  "])
    RAW --> RF

    classDef tree fill:#2e9e8c,stroke:#1a7060,color:#fff
    classDef rf   fill:#7b5ea7,stroke:#5a3d8a,color:#fff
    classDef out  fill:#e6b800,stroke:#b38a00,color:#000

    class PA,PB,PC,MID,LEAF,RFMAX tree
    class RF rf
    class RAW out
```

#### Post-Function Details

`post_fn` is applied to the aggregated FP32 scalar `raw` before writing to the AAQ register. All post functions take a single FP32 input and produce a float32 result.

| Encoding | Name | Formula | Corner case |
|----------|------|---------|-----------|
| 0 | `identity` | `x` | — |
| 1 | `value_cr` | `x * cr[cr_idx]` | — |
| 2 | `inv` | `1 / x` | 0 if x == 0 |
| 3 | `inv_sqrt` | `1 / sqrt(x)` | 0 if x ≤ 0 |
| 4 | `sqrt` | `sqrt(x)` | 0 if x < 0 |

#### 7.1.1 Aggregate First (`agg.first`)

**Assembly syntax:** `AGG.FIRST agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row`

Same lane-count rules as `AGG` — `full_xmem_row=1` uses all 128 lanes; `full_xmem_row=0` reads `valid_elements` from `CR15.valid_elements`.

Identical to `agg` except that `max` mode ignores the RF feedback value:

```text
// r_acc lanes are always FP32
n      = 128 if full_xmem_row else min(valid_elements, 128)
values = [activation(r_acc[i]) for i in 0..n-1]

if agg_mode == sum:
    raw = sum(values)       // FP32 scalar; input to post_fn
else:  // max
    raw = max(values)       // FP32 scalar; input to post_fn; RF feedback NOT included

out = post_fn(raw, cr[cr_idx])
aaq[aaq_rf_idx] = pack32(out)                  // stored as float32
```

Use `agg.first` at the start of a new accumulation sequence to avoid
reading stale data from the target register.


### 7.2 Quantize (`TBD`) Work in progress

Requires INT8 mode (`dtype == DType.INT8`). Takes no operands. Reads `r_acc`
lanes as FP32, quantizes to INT8, clamps, and writes the 128-byte result to
`aaq_result`.

```text
// dtype must be DType.INT8; r_acc lanes are FP32
n = 128 if full_xmem_row else min(CR15.valid_elements, 128)
for i in 0..n-1:
    val            = r_acc[i]        // FP32
    aaq_result[i]  = clamp(round(val), -128, 127)
aaq_result[n..127] = 0
```

Notes:
- The Quantization block appends 12 bits of metadata to the 1024-bit data
  payload to form the full 1036-bit `aaq_result`: bits [11:4] = 8-bit scale
  factor, bits [3:0] = 4-bit representation type (INT8, e6m1, e5m2, …).
- `aaq_result` must be flushed to XMEM with `XMEM.STORE_AAQ_RESULT offset base`
  before the next `AAQ` overwrites it.
- In wide-vector debug mode `AAQ` is a no-op unless
  `wide_vector_quantize_output` is explicitly set (debug feature only).

**ISA Interactions** — instructions in other slots that consume `aaq_result` written by `AAQ`:

| Instruction | Slot | Operation |
|-------------|------|-----------|
| `XMEM.STORE_AAQ_RESULT offset base` | XMEM | Writes the 128-byte `aaq_result` register to XMEM at address `offset + base`. Must be issued before the next `AAQ` to avoid overwrite. |

## 8. ISA — Instruction Reference

The AAQ stage executes **four mnemonics** in its single AAQ slot (one
per VLIW word): `AAQ_NOP`, `AGG`, `AGG.FIRST`, and `AAQ`. Detailed
binary encoding is maintained in `SLOT_BINARY_LAYOUT` in
`src/tools/ipu-common/src/ipu_common/instruction_spec.py` and is not
duplicated here.

The AAQ slot is resolved by CTRL and forwarded down the dispatch chain;
the stage does not read the CR/LR register files itself (see the
Control Stage spec, §5). The active lane count is determined by `full_xmem_row`:
`1` = always 128, `0` = `CR15.valid_elements` at cycle start — see §7.1 for the masking rule.

### 8.1 `AAQ_NOP` — No Operation

- **Summary:** No operation for the AAQ slot; performs no state changes.
- **Syntax:** `AAQ_NOP`
- **Operands:** none.
- **Operation:** none — `r_acc`, the AAQ RF (`AAQ0`–`AAQ3`), and `aaq_result` are unchanged.
- **Notes:** Inserted automatically when the AAQ slot is omitted from a VLIW word, or whenever `valid = 0` (see §3.1).

### 8.2 `AGG` — Accumulator Aggregate

- **Summary:** Collapse `r_acc` lanes into one FP32 scalar (`sum` or `max`), apply the post function, and store the result to the selected AAQ register. `full_xmem_row` controls whether all 128 lanes or only `CR15.valid_elements` lanes participate.
- **Syntax:** `AGG agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row`
- **Operands:**
  - `agg_mode` — aggregation mode: `sum` (0) or `max` (1).
  - `post_fn` — post function applied to the aggregated scalar: `value` (0), `value_cr` (1), `inv` (2), `inv_sqrt` (3). See §7.1 *Post-Function Details*.
  - `cr_idx` — `CR0`–`CR14`; the CR value used by the `value_cr` post function.
  - `aaq_rf_idx` — destination AAQ register, `AAQ0`–`AAQ3`.
  - `full_xmem_row` — `1` = always use 128 lanes; `0` = use `CR15.valid_elements` (default `0`).
- **Operation:**
  ```text
  n = 128 if full_xmem_row else min(CR15.valid_elements, 128)
  values = [activation(r_acc[i]) for i in 0..n-1]      // activation per §7.0
  if agg_mode == sum:
      raw = sum(values)
  else:  // max — includes RF feedback of the target register
      raw = max(values + [AAQ[aaq_rf_idx]])
  AAQ[aaq_rf_idx] = pack32(post_fn(raw, CR[cr_idx]))    // stored as float32
  ```
- **Examples:**
  - `AGG sum, value, CR0, AAQ0, 0;;` — sum the first `CR15.valid_elements` lanes into `AAQ0`.
  - `AGG sum, value, CR0, AAQ0, 1;;` — sum all 128 lanes into `AAQ0` regardless of `CR15`.
  - `AGG max, value_cr, CR5, AAQ1, 0;;` — max of `CR15.valid_elements` lanes (with `AAQ1` fed back), scaled by `CR5`.
- **Notes:** In `max` mode the current value of `AAQ[aaq_rf_idx]` is fed back into the max tree; use `AGG.FIRST` to avoid contamination from an uninitialised register.

### 8.3 `AGG.FIRST` — Accumulator Aggregate First

- **Summary:** Identical to `AGG` except that `max` mode does **not** fold in the previous AAQ register value (no RF feedback). `full_xmem_row` controls the active lane count as in `AGG`.
- **Syntax:** `AGG.FIRST agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row`
- **Operands:** identical to `AGG` (including `full_xmem_row`).
- **Operation:**
  ```text
  n = 128 if full_xmem_row else min(CR15.valid_elements, 128)
  values = [activation(r_acc[i]) for i in 0..n-1]
  if agg_mode == sum:
      raw = sum(values)
  else:  // max — RF feedback NOT included
      raw = max(values)
  AAQ[aaq_rf_idx] = pack32(post_fn(raw, CR[cr_idx]))
  ```
- **Example:** `AGG.FIRST max, value, CR0, AAQ0, 0;;`.
- **Notes:** Use at the start of a new accumulation sequence so a stale/uninitialised `AAQ[aaq_rf_idx]` cannot contaminate the `max`.

### 8.4 `AAQ` — Quantize Accumulator

- **Summary:** Quantize wide lanes in `POST_AAQ_REG` to 8-bit and write the 128-byte `aaq_result` register. Requires INT8 mode. The `full_xmem_row` flag controls how many lanes are active.
- **Syntax:** `AAQ full_xmem_row`
- **Operands:**
  - `full_xmem_row`: `1` = always process all 128 lanes (full XMEM row, ignores `CR15.valid_elements`); `0` = process only the first `CR15.valid_elements` lanes (clamped to 128) and zero the rest. Defaults to `0`.
- **Operation:**
  ```text
  // requires dtype == DType.INT8
  n = 128 if full_xmem_row else min(CR15.valid_elements, 128)
  for i in 0..n-1:
      aaq_result[i] = clamp(trunc(POST_AAQ_REG[i]), -128, 127)
  aaq_result[n..127] = 0
  ```
  The Quantization block appends 12 bits of metadata to the 1024-bit payload (see §5 / §7.2).
- **Example:** `AAQ 1;;` (full row), `AAQ 0;;` (use `CR15.valid_elements`).
- **Notes:** `aaq_result` must be flushed with `XMEM.STORE_AAQ_RESULT offset base` before the next `AAQ` overwrites it (see §7.2).

### 8.5 Summary Table

| Slot | Mnemonic | Operands | One-line Effect |
|------|----------|----------|-----------------|
| AAQ | `AAQ_NOP`   | —                                                      | no state change |
| AAQ | `AGG`       | `agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row` | `AAQ[idx] = post_fn(agg(r_acc[0..n-1], AAQ[idx]))`, n = 128 if full_xmem_row else CR15.valid_elements |
| AAQ | `AGG.FIRST` | `agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row` | `AAQ[idx] = post_fn(agg(r_acc[0..n-1]))` (no RF feedback), n = 128 if full_xmem_row else CR15.valid_elements |
| AAQ | `AAQ`       | `full_xmem_row`                                        | `aaq_result[0..n-1] = clamp(trunc(POST_AAQ_REG[i]), -128, 127)`, n = 128 if full_xmem_row else CR15.valid_elements |
| AAQ | `ACTIVATE`  | `activation_fn, full_xmem_row`                         | `POST_AAQ_REG[0..n-1] = activation_fn(r_acc[i])`, n = 128 if full_xmem_row else CR15.valid_elements |
