# Convolution Profiling Scripts

Memory access profiler for all `convolutions_universal` apps. Each script runs
the target app in profiling mode (math is no-op'd, only memory addresses are
tracked) and prints a table of **peak lookahead rows** per memory region.

## What is "peak lookahead"?

For each memory region (inputs, kernels, outputs, mask), the profiler tracks:

```
lookahead = max_row_seen_so_far_in_region - current_access_row
```

The reported value is the maximum of this quantity over the entire run. It
answers: *"how many rows back must this region stay buffered at peak?"*

A value of 0 means the region is accessed strictly sequentially (no buffering
needed beyond the current row). A value of N means the program read up to N
rows ahead of where it currently is, so at least N+1 rows must be live in
memory at once.

## Setup

No installation needed. Run directly with `PYTHONPATH` set from the repo root:

```bash
PYPATH="src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src"
```

Required Python packages: `numpy`

## Running a Script

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.<script_name>
```

Or directly:

```bash
PYTHONPATH=$PYPATH python src/tools/ipu-apps/src/ipu_apps/convolutions_universal/profiling/<script_name>.py
```

## Scripts

### `profile_pointwise_universal.py`
**App:** `pointwise_conv_universal` — 1×1 convolution, flexible spatial and channels.  
**Regions:** inputs, kernels, mask, outputs  
**Binary:** assembled on first run from `pointwise_conv_universal/pointwise_conv_universal.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_pointwise_universal
```

Expected pattern: `inputs = in_channels - 1`, `outputs = 0` (written sequentially, 128 B/record int8).

---

### `profile_conv_universal.py`
**App:** `conv_universal` — 3×3 convolution, flexible spatial and channels.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** assembled on first run from `conv_universal/conv_universal.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_conv_universal
```

Expected pattern: `inputs = 3 * in_channels - 1` (3-row sliding window). Output: 128 B/record int8.

---

### `profile_depthwise_universal.py`
**App:** `depthwise_conv_universal` — 3×3 depthwise convolution, flexible spatial and channels.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** assembled on first run from `depthwise_conv_universal/depthwise_conv_universal.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_universal
```

Expected pattern: `inputs = 2 * channels - 1`, `kernels = channels/8 - 1`. Output: 128 B/record int8.

---

### `profile_depthwise_stride2.py`
**App:** `depthwise_conv_stride2` — stride-2 depthwise 3×3, cols fixed at 128.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** assembled on first run from `depthwise_conv_stride2/depthwise_conv_stride2.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_stride2
```

---

### `profile_depthwise_stride2_small.py`
**App:** `depthwise_conv_stride2_small` — stride-2 depthwise 3×3, cols ∈ {32, 64}.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** the assembly is a Jinja template parameterized by `cols`. A separate
binary is assembled and cached for each `cols` value on first run
(`depthwise_conv_stride2_small_cols32.bin`, `depthwise_conv_stride2_small_cols64.bin`).

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_stride2_small
```

---

### `profile_conv_8x8.py`
**App:** `conv_8x8` — 3×3 convolution, 8×8 spatial, flexible channels.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** assembled on first run from `conv_8x8/conv_8x8.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_conv_8x8
```

---

### `profile_depthwise_8x8.py`
**App:** `depthwise_8x8` — 3×3 depthwise convolution, 8×8 spatial, flexible channels.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** assembled on first run from `depthwise_8x8/depthwise_8x8.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_8x8
```

---

### `profile_pointwise_8x8.py`
**App:** `pointwise_8x8` — 1×1 convolution, 8×8 spatial, flexible channels.  
**Regions:** inputs, kernels, outputs, mask  
**Binary:** assembled on first run from `pointwise_8x8/pointwise_8x8.asm`, cached as `.bin`.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_pointwise_8x8
```

---

### `profile_conv_first_layer.py`
**App:** `conv_first_layer` — hardcoded 256×256×3 → 128×128×16, stride 2.  
**Regions:** input_ch0, input_ch1, input_ch2, kernels, mask, outputs, temp  
**Binary:** assembled on first run from `conv_first_layer/conv_first_layer.asm`, cached as `.bin`.

This app has a fixed configuration so only one run is performed.

```bash
PYTHONPATH=$PYPATH python -m ipu_apps.convolutions_universal.profiling.profile_conv_first_layer
```

---

## Customising Configs

Each script has a `CONFIGS` list near the top defining which
(rows, cols, channels, ...) combinations to sweep. Edit it freely — invalid
configurations will be caught and printed as `ERROR: ...` inline without
aborting the rest of the sweep.

## Implementation Notes

The profiler lives in `ipu_emu`:
- `profiling_xmem.py` — `ProfilingXMem`: XMem subclass that records peak
  lookahead per CR region
- `profiling_ipu.py` — `ProfilingIpu`: Ipu subclass that no-ops mult/acc/aaq
  and intercepts xmem instructions to record addresses; `run_profile` entry
  point

Shared helpers for these scripts are in `_utils.py`.
