# Universal Convolution Apps — Guide

This directory contains convolution programs for the IPU emulator organized
into two tiers:

**Universal apps** — runtime-parameterized binaries that handle any valid
configuration via CR registers (no recompilation needed):

| Sub-package | Convolution type | Kernel size |
|---|---|---|
| `conv_universal/` | Standard (cross-channel) | 3 × 3 |
| `depthwise_conv_universal/` | Depthwise (per-channel) | 3 × 3 |
| `pointwise_conv_universal/` | Pointwise (1 × 1) | 1 × 1 |

**Fixed-size apps** — optimized for specific spatial/channel configurations,
useful when the universal app's general overhead matters:

| Sub-package | Convolution type | Spatial | Notes |
|---|---|---|---|
| `conv_8x8/` | Standard 3 × 3 | 8 × 8 | Flexible channels |
| `conv_first_layer/` | Standard 3 × 3 | 256 × 256 → 128 × 128 | Hardcoded first layer (3 IC, 16 OC, stride 2) |
| `depthwise_8x8/` | Depthwise 3 × 3 | 8 × 8 | Flexible channels |
| `depthwise_conv_stride2/` | Depthwise 3 × 3, stride 2 | 128 × 128 → 64 × 64 | For large spatial |
| `depthwise_conv_stride2_small/` | Depthwise 3 × 3, stride 2 | 32–64 × 32–64 | For small spatial |
| `pointwise_8x8/` | Pointwise 1 × 1 | 8 × 8 | Flexible channels |
| `residual_add/` | Element-wise add | Any | Residual connection |

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Supported Configurations](#2-supported-configurations)
3. [File Structure](#3-file-structure)
4. [Memory Layout](#4-memory-layout)
5. [Standard Convolution — Internals](#5-standard-convolution--internals)
6. [Depthwise Convolution — Internals](#6-depthwise-convolution--internals)
7. [Pointwise Convolution — Internals](#7-pointwise-convolution--internals)
8. [Fixed-Size Apps](#8-fixed-size-apps)
9. [Common Techniques](#9-common-techniques)
10. [Known Pitfalls](#10-known-pitfalls)
11. [Legacy Specialized Apps](#11-legacy-specialized-apps)



---

## 1. Quick Start

All commands assume you are in the repository root.

```bash
# Set up PYTHONPATH (no Bazel needed)
PYPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"
```

### Assemble

```bash
PYTHONPATH="$PYPATH" python -m ipu_as.cli assemble \
    --input src/tools/ipu-apps/src/ipu_apps/convolutions_universal/conv_universal/conv_universal.asm \
    --output /tmp/conv_universal.bin --format bin
```

Same pattern for `depthwise_conv_universal.asm` and `pointwise_conv_universal.asm`.

### Generate test data

```bash
# Standard conv: 32x32 spatial, 16 in-channels, 16 out-channels
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.conv_universal.gen_test_data \
    --rows 32 --cols 32 --in-channels 16 --out-channels 16 \
    --output-dir /tmp/test_data/conv_32x32_16to16

# Depthwise conv: 64x64 spatial, 256 channels
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.depthwise_conv_universal.gen_test_data \
    --rows 64 --cols 64 --channels 256 \
    --output-dir /tmp/test_data/dw_64x64x256

# Pointwise conv: 32x32 spatial, 16 in-channels, 32 out-channels
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.pointwise_conv_universal.gen_test_data \
    --rows 32 --cols 32 --in-channels 16 --out-channels 32 \
    --output-dir /tmp/test_data/pw_32x32_16to32
```

### Run

```bash
# Standard convolution
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.conv_universal \
    --inst /tmp/conv_universal.bin \
    --input /tmp/test_data/conv_32x32_16to16/int8/input_int8.bin \
    --kernel /tmp/test_data/conv_32x32_16to16/int8/kernel_int8.bin \
    --output /tmp/output.bin \
    --rows 32 --cols 32 --in-channels 16 --out-channels 16

# Depthwise and pointwise follow the same pattern (see --help for each).
```

### Python API

```python
from ipu_apps.convolutions_universal.conv_universal import ConvUniversalApp

app = ConvUniversalApp(
    inst_path="conv_universal.bin",
    input_path="input.bin",
    kernel_path="kernel.bin",
    output_path="output.bin",
    dtype="INT8",
    rows=32, cols=32, in_channels=16, out_channels=16,
)
state, cycles = app.run()
```

```python
from ipu_apps.convolutions_universal.depthwise_conv_universal import DepthwiseConvUniversalApp

app = DepthwiseConvUniversalApp(
    inst_path="depthwise_conv_universal.bin",
    input_path="input.bin",
    kernel_path="kernel.bin",
    output_path="output.bin",
    dtype="INT8",
    rows=64, cols=64, channels=256,
)
state, cycles = app.run()
```

```python
from ipu_apps.convolutions_universal.pointwise_conv_universal import PointwiseConvUniversalApp

app = PointwiseConvUniversalApp(
    inst_path="pointwise_conv_universal.bin",
    input_path="input.bin",
    kernel_path="kernel.bin",
    output_path="output.bin",
    dtype="INT8",
    rows=32, cols=32, in_channels=16, out_channels=32,
)
state, cycles = app.run()
```

---

## 2. Supported Configurations

### Standard convolution (`conv_universal`)

| Parameter | Constraint |
|---|---|
| `cols` | Power of 2: 16, 32, 64, or 128 |
| `rows` | Any, such that `rows * cols >= 256` (at least 2 chunks) |
| `in_channels` | Multiple of 8, >= 8 |
| `out_channels` | >= 1 |
| `dtype` | INT8, FP8_E4M3, FP8_E5M2 |

### Depthwise convolution (`depthwise_conv_universal`)

| Parameter | Constraint |
|---|---|
| `cols` | Power of 2: 16, 32, 64, or 128 |
| `rows` | Any, such that `rows * cols >= 256` (at least 2 chunks) |
| `channels` | Multiple of 8, >= 8 |
| `dtype` | INT8, FP8_E4M3, FP8_E5M2 |

### Pointwise convolution (`pointwise_conv_universal`)

| Parameter | Constraint |
|---|---|
| `rows` | Power of 2: 16, 32, 64, or 128 |
| `cols` | Power of 2: 16, 32, 64, or 128 |
| `in_channels` | Must divide 128, >= 4 (i.e. 4, 8, 16, 32, 64, 128) |
| `out_channels` | Must be divisible by `2 * (128 / in_channels)` |
| `dtype` | INT8, FP8_E4M3, FP8_E5M2 |

The `out_channels` constraint for pointwise arises from the pipeline strategy:
each kernel-group load fills two 128-byte mult-stage registers (r0 and r1),
and each register holds `128 / in_channels` output channels.

---

## 3. File Structure

Each sub-package follows the standard `ipu_apps` pattern:

```
convolutions_universal/
├── __init__.py                       # Package marker + shared helpers
├── GUIDE.md                          # This file
│
│   # Universal (runtime-parameterized) apps:
├── conv_universal/
│   ├── __init__.py                   # ConvUniversalApp harness
│   ├── __main__.py                   # CLI entry point
│   ├── conv_universal.asm            # Assembly source
│   ├── gen_test_data.py              # Golden reference + data generator
│   └── benchmark/benchmark.py        # Cycle-count benchmarks
├── depthwise_conv_universal/
│   ├── __init__.py                   # DepthwiseConvUniversalApp harness
│   ├── __main__.py
│   ├── depthwise_conv_universal.asm
│   ├── gen_test_data.py
│   └── benchmark/benchmark.py
├── pointwise_conv_universal/
│   ├── __init__.py                   # PointwiseConvUniversalApp harness
│   ├── __main__.py
│   ├── pointwise_conv_universal.asm
│   ├── gen_test_data.py
│   └── benchmark/benchmark.py
│
│   # Fixed-size apps:
├── conv_8x8/
│   ├── __init__.py                   # Conv8x8App harness (8×8, flexible channels)
│   ├── conv_8x8.asm
│   └── benchmark/benchmark.py
├── conv_first_layer/
│   ├── __init__.py                   # ConvFirstLayerApp (256×256×3 → 128×128×16)
│   └── conv_first_layer.asm
├── depthwise_8x8/
│   ├── __init__.py                   # Depthwise8x8App harness (8×8, flexible channels)
│   ├── depthwise_8x8.asm
│   └── benchmark/benchmark.py
├── depthwise_conv_stride2/
│   ├── __init__.py                   # DepthwiseConvStride2App (128×128 → 64×64)
│   ├── depthwise_conv_stride2.asm
│   └── benchmark/benchmark.py
├── depthwise_conv_stride2_small/
│   ├── __init__.py                   # DepthwiseConvStride2SmallApp (32–64 spatial)
│   ├── depthwise_conv_stride2_small.asm  # Jinja template, rendered at runtime
│   └── benchmark/benchmark.py
├── pointwise_8x8/
│   ├── __init__.py                   # Pointwise8x8App harness (8×8, flexible channels)
│   ├── pointwise_8x8.asm
│   └── benchmark/benchmark.py
└── residual_add/
    ├── __init__.py                   # ResidualAddApp
    └── residual_add.asm
```

Each harness class inherits from `ipu_apps.base.IpuApp` and implements:
- `__init__()` — validates parameters, computes derived constants
- `setup(state)` — loads data into XMEM, sets CR registers
- `teardown(state)` — dumps output from XMEM to disk

---

## 4. Memory Layout

### 4.1 Chunk-based input layout (standard & depthwise)

The IPU processes data in 128-byte chunks.  Multiple spatial rows of the
**same channel** are packed into one chunk when `cols < 128`:

```
rows_per_chunk = 128 / cols
```

For a 32x32 image: `rows_per_chunk = 4`, so 4 spatial rows fit in one 128B chunk.
For a 128x128 image: `rows_per_chunk = 1`, one row per chunk.

Chunks are interleaved across channels within each row group:

```
[chunk0_ch0][chunk0_ch1]...[chunk0_chN] [chunk1_ch0][chunk1_ch1]...
```

Address formula for pixel at `(row, col, channel)`:

```
chunk_index = row / rows_per_chunk
local_row   = row % rows_per_chunk
offset      = (chunk_index * num_channels + channel) * 128 + local_row * cols + col
```

### 4.2 Output layout

Same interleaving as input.  Each output record is **128 bytes of int8**
(the AAQ unit quantizes the 128-element int32 accumulator to int8 via
clamping to [−128, 127] before storing):

```
output_offset = (chunk_index * num_out_channels + filter) * 128 + element
```

The `aaq` instruction fires in the same VLIW cycle as the final `acc` for
each output record, so quantization adds zero extra cycles.

### 4.3 Kernel layout

**Standard conv:** `kernel[filter * in_channels * 9 + ic * 9 + dr * 3 + dc]`
The harness packs this into 128-byte blocks (8 input channels × 9 taps = 72 bytes + 56 padding per block).

**Depthwise conv:** `kernel[channel * 9 + dr * 3 + dc]`
Packed into groups of 128 bytes (8 channels × 9 taps = 72 bytes + 56 padding).

**Pointwise conv:** `kernel[oc * in_channels + ic]`
Padded to a multiple of 128 bytes. The assembly reads `in_channels` bytes per
output channel directly from the mult-stage register.

### 4.4 XMEM address regions

Input always starts at `0x000000`.  Subsequent regions are placed
dynamically by the Python harness (`_compute_addresses`) to avoid overlap
as input size grows with channels and spatial dimensions:

```
kernel_base  = align(input_end,  256)
mask_base    = align(kernel_end, 256)
output_base  = align(mask_end,   256)
```

Standard conv uses a fixed large kernel offset (`0x200000`) because its
kernel can be much larger (out_channels × in_channels/8 × 128 bytes).
Depthwise, pointwise, and the stride-2 variants compute all offsets
dynamically.

---

## 5. Standard Convolution — Internals

### 5.1 CR register parameters

| CR | Value | Description |
|---|---|---|
| cr0 | `INPUT_BASE_ADDR` | Input base address |
| cr1 | `KERNEL_BASE_ADDR` | Kernel base address |
| cr2 | `OUTPUT_BASE_ADDR` | Output base address |
| cr3 | `MASK_BASE_ADDR` | Mask base address |
| cr4 | `cols` | Spatial width |
| cr5 | `num_chunks` | = rows × cols / 128 |
| cr6 | `in_group_stride` | = in_channels × 128 |
| cr7 | `1024` | Channel group size (8 × 128, constant) |
| cr8 | `total_kernel_bytes` | = out_channels × (in_channels/8) × 128 |

### 5.2 Assembly structure (3 sections)

The assembly has three sections because border chunks require special handling:

1. **Section 1 (g0):** Top-border chunk (chunk 0).
   - Only loads S1 (current) and S2 (next) into the cyclic register.
   - S0 (previous) is implicitly zero (cyclic register initializes to zeros).
   - kr=-1 taps read from zero-padded S0 — no masking needed.

2. **Section 2 (main loop):** Middle chunks (1 through N-2).
   - Loads all three neighbors: S0, S1, S2.
   - All 9 taps use normal border masks (left/right only).

3. **Section 3 (gN):** Bottom-border chunk (last chunk).
   - Only loads S0 and S1 (skips S2 — no chunk below).
   - kr=+1 taps use bottom-border masks (slots 3/4/5) to zero out
     stale data from the cyclic register.

### 5.3 Cyclic register usage

The cyclic register (512 bytes, wrapping) holds 3 neighboring chunks at
fixed offsets:

```
S0 (previous chunk):  index 0
S1 (current chunk):   index 128
S2 (next chunk):      index 256
```

Vertical neighbor access uses a universal formula that works for all `cols`:

```
kr = -1:  cyclic_base = 128 - cols
kr =  0:  cyclic_base = 128
kr = +1:  cyclic_base = 128 + cols
```

Horizontal shift adds kc offset (-1 or +1) to the cyclic base.

### 5.4 Filter loop structure

For each chunk, the assembly iterates over all output filters:

```
for each output filter (outer loop via filter_loop):
    load kernel block into r0
    reset accumulator
    for each input channel group (8 channels, inner loop via ch_loop):
        load S0, S1, S2 into cyclic register for this channel
        multiply-accumulate all 9 taps (3 kr × 3 kc)
        if more channel groups in this kernel block: continue
        else: reload next kernel block, continue
    store accumulator to output
```

### 5.5 Mask scheme

Six 128-bit mask slots (set bit = zero that element):

| Slot | Purpose | Used by |
|---|---|---|
| 0 | No masking (all zeros) | kc=0 (center column) |
| 1 | Left border (zero col 0) | kc=-1 |
| 2 | Right border (zero last col) | kc=+1 |
| 3 | Bottom row only | gN section, kr=+1, kc=0 |
| 4 | Left + bottom | gN section, kr=+1, kc=-1 |
| 5 | Right + bottom | gN section, kr=+1, kc=+1 |

---

## 6. Depthwise Convolution — Internals

### 6.1 CR register parameters

| CR | Value | Description |
|---|---|---|
| cr0 | `INPUT_BASE_ADDR` | Input base address |
| cr1 | `KERNEL_BASE_ADDR` | Kernel base address |
| cr2 | `OUTPUT_BASE_ADDR` | Output base address |
| cr3 | `MASK_BASE_ADDR` | Mask base address |
| cr4 | `cols` | Spatial width |
| cr5 | `num_chunks` | = rows × cols / 128 |
| cr6 | `group_stride` | = channels × 128 |
| cr7 | `1024` | Channel group size (8 × 128, constant) |

### 6.2 Key difference from standard conv

In depthwise convolution, each output channel depends on only one input
channel (no cross-channel mixing).  This means:
- The kernel is much smaller: `channels × 9` bytes (vs `out × in × 9`).
- There is no separate "filter loop" — the channel loop and output loop
  are fused: after computing all 9 taps for one channel, the result is
  stored immediately.
- The accumulator is reset per-channel (inside `ch_loop`), not per-filter.

### 6.3 Assembly structure

Same 3-section structure as standard conv (top border, main, bottom border),
with the same cyclic register scheme and mask slots.  The inner loop
processes 8 channels per kernel group (one `ldr_mult_reg` load), doing
9 multiply-accumulates and one store per channel.

---

## 7. Pointwise Convolution — Internals

### 7.1 CR register parameters

| CR | Value | Description |
|---|---|---|
| cr0 | `INPUT_BASE_ADDR` | Input base address |
| cr1 | `KERNEL_BASE_ADDR` | Kernel base address |
| cr2 | `MASK_BASE_ADDR` | Mask base address (all zeros) |
| cr3 | `OUTPUT_BASE_ADDR` | Output base address |
| cr4 | `oc_per_reg` | = 128 / in_channels |
| cr5 | `row_groups` | = rows × cols / 128 |
| cr6 | `pipeline_limit` | = in_channels - 5 |
| cr7 | `out_channels` | Total output channels |
| cr8 | `row_group_stride` | = in_channels × 128 |

### 7.2 Key differences from 3×3 convolutions

- **No spatial neighbors**: Each output pixel depends only on the same
  spatial position across input channels.  No cyclic register neighbor
  tricks, no border masking.
- **No 3-section structure**: There are no border chunks to handle specially.
- **Pipelined cyclic register**: Instead of loading 3 spatial neighbors,
  the cyclic register holds 4 input channels (S0-S3) simultaneously,
  forming a software pipeline.

### 7.3 Pipeline strategy

The cyclic register (512 bytes = 4 × 128-byte slots) is used as a
4-stage pipeline buffer:

```
S0 (index   0): input channel k
S1 (index 128): input channel k+1
S2 (index 256): input channel k+2
S3 (index 384): input channel k+3
```

The pipeline loop has 4 VLIW words per iteration:
- **Word A**: Load next channel → S0, multiply from S1
- **Word B**: Load next channel → S1, multiply from S2
- **Word C**: Load next channel → S2, multiply from S3
- **Word D**: Load next channel → S3, multiply from S0 + branch

This achieves 1 multiply-accumulate per cycle (the load and multiply
overlap because they use different slots in the VLIW word).

### 7.4 Dual-register kernel loading

Each kernel group loads two mult-stage registers (r0 and r1), each
holding `oc_per_reg` output channels worth of kernel weights.  The
assembly processes all OCs from r0 first (loop A), then all from r1
(loop B), before loading the next kernel group.

### 7.5 Output pointer advancement

The output pointer is advanced by `+128` (one int8 AAQ record) after each
`xmem.store_aaq_result` in a separate VLIW word, following the same
pattern as standard and depthwise conv (see Known Pitfalls §10.1).

---

## 8. Fixed-Size Apps

These apps target specific spatial configurations and expose flexible channel
counts via CR registers.  They share the same memory layout conventions as the
universal apps (chunked input/output, same XMEM address regions).

### 8.1 `conv_8x8` — Standard 3×3 convolution, 8×8 spatial

Harness: `Conv8x8App(inst_path, input_path, kernel_path, output_path, in_channels, out_channels)`

- Spatial: fixed 8 × 8 (64 elements per channel, fits in one 128-byte chunk pair)
- Channels: any multiple of 2 for `out_channels`; `in_channels` flexible
- Output: 128 bytes int8 per OC pair (AAQ-quantized)
- Use when you need exactly 8×8 spatial and the universal app overhead matters

### 8.2 `conv_first_layer` — First-layer 3×3 stride-2 convolution

Harness: `ConvFirstLayerApp(inst_path, input_path, kernel_path, output_path)`

- Hardcoded: 256×256×3 input → 128×128×16 output, stride 2
- Designed for the first layer of a MobileNet-style network (3 RGB channels in)
- No flexible channel parameters — everything is compiled into the asm

### 8.3 `depthwise_8x8` — Depthwise 3×3 convolution, 8×8 spatial

Harness: `Depthwise8x8App(inst_path, input_path, kernel_path, output_path, channels)`

- Spatial: fixed 8 × 8
- Channels: any multiple of 2
- Processes channels in pairs (channel A in lanes 0–63, channel B in lanes 64–127 of each 128-byte output chunk)
- Output: 128 bytes int8 per channel pair

### 8.4 `depthwise_conv_stride2` — Depthwise 3×3 stride-2, large spatial

Harness: `DepthwiseConvStride2App(inst_path, input_path, kernel_path, output_path, dtype, rows, cols, channels)`

- Input: `rows × cols` spatial (designed for 128×128), flexible channels
- Output: `(rows//2) × (cols//2)` spatial (stride-2 downsampling)
- Output addresses computed dynamically via `app.output_base` (no module-level constant)
- Border handling: last output row computed from only 6 taps when S2 row is out of bounds

### 8.5 `depthwise_conv_stride2_small` — Depthwise 3×3 stride-2, small spatial

Harness: `DepthwiseConvStride2SmallApp(inst_path, input_path, kernel_path, output_path, dtype, rows, cols, channels)`

- Input: 32×32 or 64×64 spatial, flexible channels
- Assembly is a **Jinja2 template** rendered at runtime with the specific spatial parameters — the `.asm` file contains `{{ rows }}`, `{{ cols }}` placeholders
- Otherwise identical semantics to `depthwise_conv_stride2`

### 8.6 `pointwise_8x8` — Pointwise 1×1 convolution, 8×8 spatial

Harness: `Pointwise8x8App(inst_path, input_path, kernel_path, output_path, in_channels, out_channels)`

- Spatial: fixed 8 × 8
- Channels: flexible `in_channels` and `out_channels`
- Output: 128 bytes int8 per OC pair

### 8.7 `residual_add` — Element-wise addition

Harness: `ResidualAddApp(inst_path, input_a_path, input_b_path, output_path, num_chunks)`

- Adds two equally-shaped int8 tensors element-wise (residual connection)
- `num_chunks`: total number of 128-byte chunks across both inputs

---

## 9. Common Techniques

### 9.1 VLIW execution order

Within one cycle, slots execute in this order:

```
LR → XMEM → MULT → ACC → AAQ → COND
```

This means:
- An `incr` or `set` (LR slot) takes effect before `ldr`/`str` (XMEM slot)
  in the **same** VLIW word.
- `mult.ve` (MULT slot) sees LR writes from the same cycle.
- `acc` (ACC slot) accumulates the current cycle's `mult_res`.
- `blt`/`beq` (COND slot) reads from the **snapshot** taken at cycle start.

### 9.2 Cyclic register for spatial neighbors

For 3×3 convolutions, the cyclic register stores 3 consecutive chunks at
indices 0, 128, and 256.  The vertical offset formula `128 ± cols` works
because:
- Each chunk has `rows_per_chunk = 128/cols` spatial rows.
- Moving ±cols in the cyclic register shifts by exactly one spatial row.
- The cyclic register wraps at 512, so offset 128-cols accesses the
  previous chunk's last rows (kr=-1) and offset 128+cols accesses the
  next chunk's first rows (kr=+1).

### 9.3 Mask-based border handling

Instead of conditional branches for border pixels, the IPU uses a 128-bit
mask register.  Each mask slot has one bit per element in the SIMD lane.
A set bit **zeros** the corresponding element in the multiply result.

This eliminates branches entirely — the same 9-tap multiply sequence runs
for every pixel, with masks zeroing out-of-bounds contributions.

### 9.4 Channel group processing

Input channels are processed in groups of 8 (one `ldr_mult_reg` loads
128 bytes = 8 channels × 9 taps + padding for 3×3 convs).  The channel
group size constant `1024 = 8 × 128` is used as the loop bound increment.

---

## 10. Known Pitfalls

### 10.1 `xmem.store_aaq_result` + `add`/`incr` in the same VLIW word

**Never** put `xmem.store_aaq_result lr7 cr2` and an LR write to `lr7` in
the same VLIW word.  Because LR executes before XMEM, the store sees the
**post-incremented** address, writing the output 128 bytes too late.

Correct pattern (used in all working code):
```asm
aaq;;
xmem.store_aaq_result lr7 cr2;;   # Store (reads current lr7)

add lr7 lr7 cr12;;                # Then advance output pointer
```

The same hazard applied to the legacy `str_acc_reg` instruction (which
stored 512-byte int32 records); `xmem.store_aaq_result` uses 128-byte
int8 records but the ordering rule is identical.

### 10.2 `blt` reads from snapshot

Branch conditions (`blt`, `beq`) read registers from the **snapshot** taken
at the start of the cycle.  If you `incr` a register and `blt` on it in the
same VLIW word, the branch sees the old value.  Put the `incr` in a
previous cycle.

### 10.3 `lr15` gets trashed by S2 index computation

In the main loop of standard and depthwise conv, `lr15` is temporarily
overwritten with `256` (= lr4 + lr4) for the S2 cyclic load.  After the
filter/channel loop finishes, lr15 must be restored to the chunk loop limit
(`cr5 - 1`) before the chunk loop branch.

### 10.4 Cyclic register initializes to zero

The cyclic register starts as all zeros.  The top-border section exploits
this: it skips loading S0, and the kr=-1 taps naturally read zeros (correct
zero-padding for the top border).

---

## 11. Legacy Specialized Apps

The universal apps replace all of the following specialized per-configuration
apps.  These still exist in `ipu_apps/` but are no longer needed for new
work:

**Standard convolution:**
`conv_8x8_16to16`, `conv_8x8_64to64`, `conv_8x8_160to160`,
`conv_16x16x64`, `conv_16x16x128`, `conv_16x16_256to128`,
`conv_32x32x16`, `conv_32x32x64`, `conv_32x32x96`, `conv_32x32_192to96`,
`conv_64x64_8to16`,
`conv_128x128_1to8`, `conv_128x128_4to8`, `conv_128x128_8to16`

**Depthwise convolution:**
`depthwise_conv`, `depthwise_conv_128x128x8`, `depthwise_conv_128x128x64`,
`depthwise_conv_64x64x1`, `depthwise_conv_64x64x256`,
`depthwise_conv_8x8x160`

**Pointwise convolution:**
`pointwise_conv`, `pointwise_conv_32x32x16`, `pointwise_conv_32x32x32`,
`pointwise_conv_128x128_16to64`, `pointwise_conv_128x128_64to16`,
`pointwise_conv_8x8x160`
