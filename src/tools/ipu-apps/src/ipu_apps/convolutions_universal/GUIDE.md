# `convolutions_universal/` — Guide

Convolution and residual-add apps for the IPU emulator, organized by family.

```
convolutions_universal/
├── conv/                # standard 3×3 convolutions
├── depthwise/           # depthwise 3×3 convolutions
├── pointwise/           # pointwise (1×1) convolutions
├── residual_add/        # element-wise residual addition
└── profiling/           # XMEM-access profilers (no math)
```

Each family contains one **universal** app (runtime-parameterized via CR
registers — no recompilation needed for new shapes) plus, where useful,
**fixed-size** apps optimized for specific shapes where the universal
overhead matters.

| App | Family | Kind | Kernel | Notes |
|---|---|---|---|---|
| `conv_universal` | conv | universal | 3×3 | Standard cross-channel conv |
| `conv_8x8` | conv | fixed | 3×3 | Spatial 8×8, flexible channels |
| `conv_first_layer` | conv | fixed | 3×3 | 256×256×3 → 128×128×16, stride 2 |
| `depthwise_conv_universal` | depthwise | universal | 3×3 | Per-channel conv |
| `depthwise_8x8` | depthwise | fixed | 3×3 | Spatial 8×8, flexible channels |
| `depthwise_conv_stride2` | depthwise | fixed | 3×3 | Large spatial → stride-2 downsample |
| `depthwise_conv_stride2_small` | depthwise | fixed | 3×3 | 32–64 spatial → stride-2 downsample |
| `pointwise_conv_universal` | pointwise | universal | 1×1 | Cross-channel 1×1 |
| `pointwise_8x8` | pointwise | fixed | 1×1 | Spatial 8×8, flexible channels |
| `residual_add` | residual_add | — | — | Element-wise int8 add |

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Supported Configurations](#2-supported-configurations)
3. [File Structure](#3-file-structure)
4. [Memory Layout](#4-memory-layout)
5. [`conv/` family](#5-conv-family)
6. [`depthwise/` family](#6-depthwise-family)
7. [`pointwise/` family](#7-pointwise-family)
8. [`residual_add/`](#8-residual_add)
9. [`profiling/`](#9-profiling)
10. [Common Techniques](#10-common-techniques)
11. [Known Pitfalls](#11-known-pitfalls)

---

## 1. Quick Start

All commands assume you are at the repository root.

```bash
# Set up PYTHONPATH (no Bazel needed)
PYPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"
```

### Assemble

```bash
PYTHONPATH="$PYPATH" python -m ipu_as.cli assemble \
    --input src/tools/ipu-apps/src/ipu_apps/convolutions_universal/conv/conv_universal/conv_universal.asm \
    --output /tmp/conv_universal.bin --format bin
```

Same pattern for any other app — substitute its family and name in the path:

- `convolutions_universal/depthwise/depthwise_conv_universal/depthwise_conv_universal.asm`
- `convolutions_universal/pointwise/pointwise_conv_universal/pointwise_conv_universal.asm`

### Generate test data

```bash
# Standard conv: 32×32 spatial, 16 in-channels, 16 out-channels
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.conv.conv_universal.gen_test_data \
    --rows 32 --cols 32 --in-channels 16 --out-channels 16 \
    --output-dir /tmp/test_data/conv_32x32_16to16

# Depthwise conv: 64×64 spatial, 256 channels
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal.gen_test_data \
    --rows 64 --cols 64 --channels 256 \
    --output-dir /tmp/test_data/dw_64x64x256

# Pointwise conv: 32×32 spatial, 16 in-channels, 32 out-channels
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.pointwise.pointwise_conv_universal.gen_test_data \
    --rows 32 --cols 32 --in-channels 16 --out-channels 32 \
    --output-dir /tmp/test_data/pw_32x32_16to32
```

### Run

```bash
# Standard convolution
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.conv.conv_universal \
    --inst /tmp/conv_universal.bin \
    --input /tmp/test_data/conv_32x32_16to16/int8/input_int8.bin \
    --kernel /tmp/test_data/conv_32x32_16to16/int8/kernel_int8.bin \
    --output /tmp/output.bin \
    --rows 32 --cols 32 --in-channels 16 --out-channels 16

# Depthwise and pointwise follow the same pattern (see --help for each).
```

### Python API

```python
from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp

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
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import DepthwiseConvUniversalApp

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
from ipu_apps.convolutions_universal.pointwise.pointwise_conv_universal import PointwiseConvUniversalApp

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
| `in_channels` | Multiple of 8, ≥ 8 |
| `out_channels` | ≥ 1 |
| `dtype` | INT8, FP8_E4M3, FP8_E5M2 |

### Depthwise convolution (`depthwise_conv_universal`)

| Parameter | Constraint |
|---|---|
| `cols` | Power of 2: 16, 32, 64, or 128 |
| `rows` | Any, such that `rows * cols >= 256` (at least 2 chunks) |
| `channels` | Multiple of 8, ≥ 8 |
| `dtype` | INT8, FP8_E4M3, FP8_E5M2 |

### Pointwise convolution (`pointwise_conv_universal`)

| Parameter | Constraint |
|---|---|
| `rows` | Power of 2: 16, 32, 64, or 128 |
| `cols` | Power of 2: 16, 32, 64, or 128 |
| `in_channels` | Must divide 128, ≥ 4 (i.e. 4, 8, 16, 32, 64, 128) |
| `out_channels` | Must be divisible by `2 * (128 / in_channels)` |
| `dtype` | INT8, FP8_E4M3, FP8_E5M2 |

The `out_channels` constraint for pointwise arises from the pipeline strategy:
each kernel-group load fills two 128-byte mult-stage registers (R0 and R1),
and each register holds `128 / in_channels` output channels.

---

## 3. File Structure

```
convolutions_universal/
├── __init__.py                       # Shared helpers (dtype, masks, output dump)
├── weights.py                        # Weight-packing utilities
├── GUIDE.md                          # This file
│
├── conv/
│   ├── conv_universal/
│   │   ├── __init__.py               # ConvUniversalApp harness
│   │   ├── __main__.py               # CLI entry point
│   │   ├── conv_universal.asm
│   │   ├── gen_test_data.py
│   │   └── benchmark/benchmark.py
│   ├── conv_8x8/
│   │   ├── __init__.py               # Conv8x8App harness
│   │   ├── conv_8x8.asm
│   │   └── benchmark/benchmark.py
│   └── conv_first_layer/
│       ├── __init__.py               # ConvFirstLayerApp
│       ├── conv_first_layer.asm
│       └── benchmark/benchmark.py
│
├── depthwise/
│   ├── depthwise_conv_universal/
│   │   ├── __init__.py               # DepthwiseConvUniversalApp
│   │   ├── __main__.py
│   │   ├── depthwise_conv_universal.asm
│   │   ├── gen_test_data.py
│   │   └── benchmark/benchmark.py
│   ├── depthwise_8x8/
│   │   ├── __init__.py               # Depthwise8x8App
│   │   ├── depthwise_8x8.asm
│   │   └── benchmark/benchmark.py
│   ├── depthwise_conv_stride2/
│   │   ├── __init__.py               # DepthwiseConvStride2App
│   │   ├── depthwise_conv_stride2.asm
│   │   └── benchmark/benchmark.py
│   └── depthwise_conv_stride2_small/
│       ├── __init__.py               # DepthwiseConvStride2SmallApp
│       ├── depthwise_conv_stride2_small.asm   # Jinja template
│       └── benchmark/benchmark.py
│
├── pointwise/
│   ├── pointwise_conv_universal/
│   │   ├── __init__.py               # PointwiseConvUniversalApp
│   │   ├── __main__.py
│   │   ├── pointwise_conv_universal.asm
│   │   ├── gen_test_data.py
│   │   └── benchmark/benchmark.py
│   └── pointwise_8x8/
│       ├── __init__.py               # Pointwise8x8App
│       ├── pointwise_8x8.asm
│       └── benchmark/benchmark.py
│
├── residual_add/
│   ├── __init__.py                   # ResidualAddApp
│   └── residual_add.asm
│
└── profiling/
    ├── _utils.py
    ├── profile_*.py                  # one per app
    ├── run_mobilevit_sweep.py        # full MobileViT-S layer sweep
    └── results/                      # CSV outputs
```

Each harness class inherits from `ipu_apps.base.IpuApp` and implements:

- `__init__()` — validates parameters, computes derived constants
- `setup(state)` — loads data into XMEM, sets CR registers
- `teardown(state)` — dumps output from XMEM to disk

---

## 4. Memory Layout

XMEM is **2 MB** byte-addressable (matching the C model). All app address
constants fit comfortably inside this window — see §11.3 for the exact
addresses each app uses.

### 4.1 Chunk-based input layout (standard & depthwise)

The IPU processes data in 128-byte chunks. Multiple spatial rows of the
**same channel** pack into one chunk when `cols < 128`:

```
rows_per_chunk = 128 / cols
```

For a 32×32 image: `rows_per_chunk = 4`, so 4 spatial rows fit in one 128 B chunk.
For a 128×128 image: `rows_per_chunk = 1`, one row per chunk.

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

Same interleaving as input. Each output record is **128 bytes of int8**
produced by the AAQ unit, which quantizes the 128-element int32 accumulator
to int8 by clamping each lane to `[−128, 127]`:

```
output_offset = (chunk_index * num_out_channels + filter) * 128 + element
```

> **AAQ semantics — important.** `aaq` is a plain clamp now:
> `aaq_result[i] = clamp(r_acc[i], -128, 127)`. There is **no** `>>24`
> right shift. All assembly in this directory assumes the accumulator
> already sits in int8 range, so kernels and activations are sized for
> that. The `aaq` instruction fires in the same VLIW cycle as the final
> `acc` for each output record, so quantization adds zero extra cycles.

### 4.3 Kernel layout

- **Standard conv:** `kernel[filter * in_channels * 9 + ic * 9 + dr * 3 + dc]`.
  Packed into FPB=28 super-blocks (256 bytes each, half-loaded into R0 + R1)
  via [`weights.py:pack_conv_weights_dense`](weights.py).
- **Depthwise conv:** `kernel[channel * 9 + dr * 3 + dc]`. Packed into
  groups of 128 bytes (8 channels × 9 taps = 72 bytes + 56 padding).
- **Pointwise conv:** `kernel[oc * in_channels + ic]`. Padded to a multiple
  of 128 bytes; the assembly reads `in_channels` bytes per output channel
  directly from the mult-stage register.

### 4.4 XMEM address regions

Each universal app uses a fixed region layout sized to fit in 2 MB:

```
conv_universal:                     depthwise_conv_universal:        pointwise_conv_universal:
  INPUT   0x000000                    INPUT   0x000000                  INPUT   0x000000
  KERNEL  0x100000                    KERNEL  0x110000                  KERNEL  0x110000
  MASK    0x180000                    MASK    0x120000                  MASK    0x120000
  ZERO    0x180080                    ZERO    0x120080                  OUTPUT  0x130000
  OUTPUT  0x1C0000                    OUTPUT  0x130000
```

Fixed-size apps use even tighter layouts (see each app's `__init__.py`).
`depthwise_conv_stride2` computes most regions dynamically via an `align()`
helper.

---

## 5. `conv/` family

### 5.1 `conv_universal` — runtime-parameterized standard 3×3

#### CR register parameters

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
| cr8 | `total_kernel_bytes` | = out_channels × ceil(in_ch/28) × 256 |

#### Assembly structure — 3 sections

Border chunks need special handling, so the asm has three sections:

1. **g0 (top border):** Loads only S1 (current) and S2 (next) into the
   cyclic register. S0 (previous) is implicitly zero because the cyclic
   register initializes to zeros — `kr=-1` taps therefore read zero-padded
   data with no masking needed.
2. **Main loop (middle chunks):** Loads all three neighbours S0, S1, S2.
   All 9 taps use normal left/right border masks.
3. **gN (bottom border):** Loads S0 and S1 only (no chunk below). `kr=+1`
   taps use bottom-border masks (slots 3/4/5) to zero stale data from the
   cyclic register.

#### Cyclic register usage

The 512-byte cyclic register holds three neighbour chunks at fixed offsets:

```
S0 (previous chunk):  index 0
S1 (current chunk):   index 128
S2 (next chunk):      index 256
```

Vertical neighbour access uses one universal formula for all valid `cols`:

```
kr = -1:  cyclic_base = 128 - cols
kr =  0:  cyclic_base = 128
kr = +1:  cyclic_base = 128 + cols
```

Horizontal shift adds the `kc` offset (-1 or +1) to the cyclic base.

#### Filter loop

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

#### Mask scheme

Six 128-bit mask slots (set bit = zero that element in `mult_res`):

| Slot | Purpose | Used by |
|---|---|---|
| 0 | No masking (all zeros) | kc=0 (centre column) |
| 1 | Left border (zero col 0) | kc=-1 |
| 2 | Right border (zero last col) | kc=+1 |
| 3 | Bottom row only | gN section, kr=+1, kc=0 |
| 4 | Left + bottom | gN section, kr=+1, kc=-1 |
| 5 | Right + bottom | gN section, kr=+1, kc=+1 |

### 5.2 `conv_8x8` — standard 3×3, 8×8 spatial

Harness: `Conv8x8App(inst_path, input_path, kernel_path, output_path, in_channels, out_channels)`.

- Spatial: fixed 8×8 (64 elements per channel, fits in one 128-byte chunk pair)
- Channels: any multiple of 2 for `out_channels`; `in_channels` flexible
- Output: 128 bytes int8 per OC pair (AAQ-quantized)

Use when you need exactly 8×8 spatial and the universal app overhead matters.

### 5.3 `conv_first_layer` — first-layer 3×3 stride-2

Harness: `ConvFirstLayerApp(inst_path, input_path, kernel_path, output_path)`.

- Hardcoded: 256×256×3 input → 128×128×16 output, stride 2
- Designed for the first layer of a MobileNet-style network (3 RGB channels)
- No flexible channel parameters — everything is compiled into the asm

---

## 6. `depthwise/` family

### 6.1 `depthwise_conv_universal` — runtime-parameterized

#### CR register parameters

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

#### Key differences from `conv_universal`

In depthwise convolution, each output channel depends on only one input
channel (no cross-channel mixing). Concretely:

- The kernel is much smaller: `channels × 9` bytes (vs `out × in × 9`).
- There is no separate "filter loop" — the channel loop and output loop
  are fused: after computing all 9 taps for one channel, the result is
  stored immediately.
- The accumulator is reset per-channel (inside `ch_loop`), not per-filter.

Same 3-section structure as `conv_universal` (top, main, bottom borders),
same cyclic register scheme, same mask slots. The inner loop processes
8 channels per kernel group (one `ldr_mult_reg` load), doing 9
multiply-accumulates and one store per channel.

### 6.2 `depthwise_8x8` — 8×8 spatial

Harness: `Depthwise8x8App(inst_path, input_path, kernel_path, output_path, channels)`.

- Spatial: fixed 8×8
- Channels: any multiple of 2
- Processes channels in pairs (channel A in lanes 0–63, channel B in lanes
  64–127 of each 128-byte output chunk)
- Output: 128 bytes int8 per channel pair

### 6.3 `depthwise_conv_stride2` — large spatial, stride-2

Harness: `DepthwiseConvStride2App(inst_path, input_path, kernel_path, output_path, dtype, rows, cols, channels)`.

- Input: `rows × cols` spatial (designed for 128×128), flexible channels
- Output: `(rows//2) × (cols//2)` spatial (stride-2 downsampling)
- Output addresses computed dynamically via `app.output_base` (no
  module-level constant)
- Border handling: the last output row is computed from only 6 taps when
  the S2 row is out of bounds

### 6.4 `depthwise_conv_stride2_small` — small spatial, stride-2

Harness: `DepthwiseConvStride2SmallApp(inst_path, input_path, kernel_path, output_path, dtype, rows, cols, channels)`.

- Input: 32×32 or 64×64 spatial, flexible channels
- The `.asm` is a **Jinja2 template** rendered at runtime with the
  specific spatial parameters — the file contains `{{ rows }}`, `{{ cols }}`
  placeholders
- Otherwise identical semantics to `depthwise_conv_stride2`

---

## 7. `pointwise/` family

### 7.1 `pointwise_conv_universal` — runtime-parameterized 1×1

#### CR register parameters

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

#### Key differences from 3×3 convolutions

- **No spatial neighbours**: each output pixel depends only on the same
  spatial position across input channels. No cyclic-register neighbour
  tricks, no border masking.
- **No 3-section structure**: no border chunks to handle specially.
- **Pipelined cyclic register**: instead of three spatial neighbours, the
  cyclic register holds **four** input channels (S0–S3) simultaneously,
  forming a software pipeline.

#### Pipeline strategy

The 512-byte cyclic register acts as a 4-stage pipeline buffer:

```
S0 (index   0): input channel k
S1 (index 128): input channel k+1
S2 (index 256): input channel k+2
S3 (index 384): input channel k+3
```

The pipeline loop has 4 VLIW words per iteration:

- **Word A**: load next channel → S0, multiply from S1
- **Word B**: load next channel → S1, multiply from S2
- **Word C**: load next channel → S2, multiply from S3
- **Word D**: load next channel → S3, multiply from S0 + branch

This achieves 1 multiply-accumulate per cycle (the load and multiply
overlap because they use different VLIW slots).

#### Dual-register kernel loading

Each kernel group loads two mult-stage registers (R0 and R1), each holding
`oc_per_reg` output channels worth of kernel weights. The assembly
processes all OCs from R0 first (loop A), then all from R1 (loop B),
before loading the next kernel group.

#### Output pointer advancement

The output pointer is advanced by `+128` (one int8 AAQ record) after each
`xmem.store_aaq_result` in a **separate** VLIW word — see §11.1.

### 7.2 `pointwise_8x8` — 8×8 spatial

Harness: `Pointwise8x8App(inst_path, input_path, kernel_path, output_path, in_channels, out_channels)`.

- Spatial: fixed 8×8
- Channels: flexible `in_channels` and `out_channels`
- Output: 128 bytes int8 per OC pair

---

## 8. `residual_add`

Harness: `ResidualAddApp(inst_path, input_a_path, input_b_path, output_path, num_chunks)`.

- Adds two equally-shaped int8 tensors element-wise (residual connection)
- `num_chunks`: total number of 128-byte chunks across both inputs

---

## 9. `profiling/`

XMEM-access profilers that run the program's control flow and memory
operations but skip all multiply/accumulate work. Useful for measuring
data-movement cost separately from compute.

Each app has a matching `profile_<app>.py` script that runs the harness
under [`ProfilingIpu`](../../../../ipu-emu-py/src/ipu_emu/profiling_ipu.py) (a subclass of
`Ipu` that no-ops `mult`/`acc`/`aaq` and records XMEM accesses) and
writes a CSV to [`profiling/results/`](profiling/results/).

```bash
# Run a single app
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.profiling.profile_conv_universal

# Sweep across all MobileViT-S layers
PYTHONPATH="$PYPATH" python -m ipu_apps.convolutions_universal.profiling.run_mobilevit_sweep
```

---

## 10. Common Techniques

### 10.1 VLIW execution order

Within one cycle, slots execute in this order:

```
LR → XMEM → MULT → ACC → AAQ → COND
```

Consequences:

- An `incr` or `set` (LR slot) takes effect **before** `ldr`/`str` (XMEM
  slot) in the **same** VLIW word.
- `mult.ve` (MULT slot) sees LR writes from the same cycle.
- `acc` (ACC slot) accumulates the current cycle's `mult_res`.
- `blt`/`beq` (COND slot) reads from the **snapshot** taken at cycle start.

### 10.2 Cyclic register for spatial neighbours

For 3×3 convolutions, the cyclic register stores 3 consecutive chunks at
indices 0, 128, and 256. The vertical offset formula `128 ± cols` works
because:

- Each chunk has `rows_per_chunk = 128/cols` spatial rows.
- Moving ±cols in the cyclic register shifts by exactly one spatial row.
- The cyclic register wraps at 512, so offset `128-cols` accesses the
  previous chunk's last rows (`kr=-1`) and offset `128+cols` accesses the
  next chunk's first rows (`kr=+1`).

### 10.3 Mask-based border handling

Instead of conditional branches for border pixels, the IPU uses a 128-bit
mask register. Each mask slot has one bit per element in the SIMD lane.
A **set** bit **zeros** the corresponding element in the multiply result.

This eliminates branches entirely — the same 9-tap multiply sequence runs
for every pixel, with masks zeroing out-of-bounds contributions.

### 10.4 Channel-group processing

Input channels are processed in groups of 8 (one `ldr_mult_reg` loads
128 bytes = 8 channels × 9 taps + padding for 3×3 convs). The channel
group size constant `1024 = 8 × 128` is used as the loop bound increment.

---

## 11. Known Pitfalls

### 11.1 `xmem.store_aaq_result` + LR write to the same register in one word

**Never** put `xmem.store_aaq_result lr7 cr2` and an LR write to `lr7` in
the same VLIW word. Because LR executes before XMEM, the store sees the
**post-incremented** address, writing the output 128 bytes too late.

Correct pattern (used throughout):

```asm
aaq;;
xmem.store_aaq_result lr7 cr2;;   # store (reads current lr7)
add lr7 lr7 cr12;;                # then advance output pointer
```

The same hazard applied to the legacy `str_acc_reg` instruction (which
stored 512-byte int32 records); `xmem.store_aaq_result` uses 128-byte
int8 records but the ordering rule is identical.

### 11.2 `blt` reads from snapshot

Branch conditions (`blt`, `beq`) read registers from the **snapshot**
taken at the start of the cycle. If you `incr` a register and `blt` on
it in the same VLIW word, the branch sees the **old** value. Put the
`incr` in a previous cycle.

### 11.3 `lr15` is trashed by the S2 index computation

In the main loop of standard and depthwise conv, `lr15` is temporarily
overwritten with `256` (= lr4 + lr4) for the S2 cyclic load. After the
filter/channel loop finishes, `lr15` must be restored to the chunk-loop
limit (`cr5 - 1`) **before** the chunk-loop branch.

### 11.4 Cyclic register initializes to zero

The cyclic register starts as all zeros. The top-border section exploits
this: it skips loading S0, and the `kr=-1` taps naturally read zeros
(correct zero-padding for the top border).

### 11.5 AAQ is a direct clamp — no `>>24`

`aaq_result[i] = clamp(r_acc[i], -128, 127)`. There is **no** arithmetic
right shift by 24 before clamping. All assembly in this directory assumes
this semantics; if you run against an older emulator that still shifts,
every output byte will be silently scaled by `>>24` and tests will fail
in confusing ways. See `Ipu.execute_aaq` in
[`ipu.py`](../../../../ipu-emu-py/src/ipu_emu/ipu.py).
