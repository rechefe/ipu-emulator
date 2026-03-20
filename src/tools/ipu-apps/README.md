# ipu-apps

IPU application test harnesses — Python ports of the C test harnesses.

## Framework

Subclass `IpuApp`, write `setup` and `teardown`, call `run`:

```python
from ipu_apps import IpuApp

class MyApp(IpuApp):
    def setup(self, state):
        load_binary_to_xmem(state, self.data_path, 0x0000, 128)
        state.regfile.set_cr(0, 0x0000)

    def teardown(self, state):
        if self.output_path:
            dump_xmem_to_binary(state, self.output_path, 0x1000, 128, 1)
```

Extra `__init__` kwargs are stored as attributes automatically:

```python
app = MyApp(inst_path="program.bin", data_path="data.bin", output_path="out.bin")
state, cycles = app.run()
```

Each app lives in its own subpackage under `ipu_apps/` with a consistent structure:

| File | Purpose |
|------|---------|
| `__init__.py` | App harness class (subclass of `IpuApp`) |
| `*.asm` | VLIW assembly kernel |
| `__main__.py` | Debug CLI runner |
| `gen_test_data.py` | Golden reference data generator |
| `test_data_format/` | Generated test data directory |

## Application catalog

### Fully connected

128 inputs, 64 outputs, 10 samples.

```python
from ipu_apps.fully_connected import FullyConnectedApp
app = FullyConnectedApp(
    inst_path="fc.bin", inputs_path="inputs.bin",
    weights_path="weights.bin", output_path="output.bin", dtype="INT8",
)
```

### Depthwise convolutions (3x3)

Per-channel 3x3 convolution (no cross-channel mixing).

| App | Spatial | Channels | Rows/chunk |
|-----|---------|----------|------------|
| `depthwise_conv` | 128x128 | 1 | 1 |
| `depthwise_conv_64x64x1` | 64x64 | 1 | 2 |
| `depthwise_conv_128x128x8` | 128x128 | 8 | 1 |

### Standard convolutions (3x3)

3x3 spatial convolution with cross-channel mixing.

| App | Spatial | In ch -> Out ch | Rows/chunk |
|-----|---------|-----------------|------------|
| `conv2ch` | 128x128 | 2 -> 1 | 1 |
| `conv2ch2ch` | 128x128 | 2 -> 2 | 1 |
| `conv_128x128_1to8` | 128x128 | 1 -> 8 | 1 |
| `conv_128x128_4to8` | 128x128 | 4 -> 8 | 1 |
| `conv_128x128_8to16` | 128x128 | 8 -> 16 | 1 |
| `conv_64x64_8to16` | 64x64 | 8 -> 16 | 2 |

### Pointwise convolutions (1x1)

1x1 convolution (pure channel mixing, no spatial kernel).

| App | Spatial | In ch -> Out ch | Rows/chunk |
|-----|---------|-----------------|------------|
| `pointwise_conv` | 128x128 | 8 -> 8 | 1 |
| `pointwise_conv_32x32x16` | 32x32 | 16 -> 16 | 4 |
| `pointwise_conv_32x32x32` | 32x32 | 32 -> 32 | 4 |
| `pointwise_conv_128x128_16to64` | 128x128 | 16 -> 64 | 1 |
| `pointwise_conv_128x128_64to16` | 128x128 | 64 -> 16 | 1 |

## Code generation

### Pointwise convolution assembly generator

`gen_pointwise_conv_asm.py` generates fully-pipelined VLIW assembly for
arbitrary pointwise convolutions without writing assembly by hand.

```bash
python -m ipu_apps.gen_pointwise_conv_asm \
    --rows 128 --cols 128 \
    --in-channels 64 --out-channels 16 \
    --output pointwise_conv_128x128_64to16.asm
```

Or print to stdout:

```bash
python -m ipu_apps.gen_pointwise_conv_asm \
    --in-channels 16 --out-channels 32
```

#### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--rows` | 128 | Spatial height (power of 2, 16-128) |
| `--cols` | 128 | Spatial width (power of 2, 16-128) |
| `--in-channels` | required | Input channels (must divide 128, >= 4) |
| `--out-channels` | required | Output channels |
| `--output`, `-o` | stdout | Output `.asm` file path |

#### Constraints

- `in_channels` must divide 128 (so whole output channels fit per r register)
- `in_channels` >= 4 (required for the 4-slot r_cyclic pipeline)
- `out_channels` must be divisible by `2 * (128 / in_channels)`

#### How it works

The generator produces the same pipelined VLIW assembly that the
hand-written apps use:

1. **Outer loop** over row groups (`rows * cols / 128` iterations)
2. **Kernel-group loop** loads r0/r1 register pairs (256 bytes each)
3. **Inner loop** per output channel:
   - Pre-loads 4 input channels into r_cyclic slots S0-S3
   - Pipelines remaining loads: while mult.ve reads slot S[k], the next
     cycle loads new data into that slot
   - Reloads ich0-3 at the end for the next output channel
   - Stores accumulator and advances output pointer

The total VLIW words per output channel = `in_channels + 4`
(e.g., 20 for 16 ic, 68 for 64 ic).

#### Register allocation

The generated code uses a fixed register map that works across all
valid parameter combinations:

| Register | Value | Purpose |
|----------|-------|---------|
| lr0 | dynamic | Input row-group base address |
| lr1 | dynamic | Output pointer (pre-offset by -512) |
| lr2 | dynamic | Row-group counter |
| lr3 | dynamic | Output channel counter |
| lr4 | dynamic | Kernel byte index (0..in_channels-1 per oc) |
| lr5 | 512 | Output stride (128 elements x 4 bytes) |
| lr6 | 128 | Cyclic S1 offset / address stride |
| lr7 | 0 | Cyclic S0 offset / zero constant |
| lr8 | 256 | Cyclic S2 offset / kernel group stride |
| lr9 | 384 | Cyclic S3 offset |
| lr10 | 128/in_ch | Output channels per kernel half |
| lr11 | row_groups | Row-group loop limit |
| lr12 | dynamic | Kernel memory offset (advances per group) |
| lr13 | dynamic | Inner loop oc limit |
| lr14 | dynamic | Temp (address computation) |
| lr15 | out_ch | Total output channel limit |

#### Memory layout

The generated assembly expects CR registers to be set by the Python harness:

| CR | Content | Size |
|----|---------|------|
| cr0 | Input base | `row_groups * in_channels * 128` bytes |
| cr1 | Kernel base | `in_channels * out_channels` bytes |
| cr2 | Mask base | 128 bytes (all zeros) |
| cr3 | Output base | `row_groups * out_channels * 512` bytes |

Input and output are interleaved by row-group:
- Input: `rg0_ch0(128B), rg0_ch1(128B), ..., rg0_chN(128B), rg1_ch0, ...`
- Output: `rg0_oc0(512B), rg0_oc1(512B), ..., rg0_ocM(512B), rg1_oc0, ...`
- Kernel: `kernel[oc * in_channels + ic]` (linear, row-major by output channel)

## Running without Bazel

```bash
cd /path/to/ipu-emulator
PYPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"

# Assemble
PYTHONPATH="$PYPATH" python -m ipu_as.cli assemble \
    --input app.asm --output app.bin --format bin

# Generate test data (example for pointwise_conv_128x128_64to16)
PYTHONPATH="$PYPATH" python -m ipu_apps.pointwise_conv_128x128_64to16.gen_test_data

# Run tests
PYTHONPATH="$PYPATH" \
    PCONV128_64TO16_INST_BIN=/tmp/app.bin \
    PCONV128_64TO16_DATA_DIR=src/tools/ipu-apps/src/ipu_apps/pointwise_conv_128x128_64to16/test_data_format \
    python -m pytest src/tools/ipu-apps/test/test_pointwise_conv_128x128_64to16.py -v

# Generate assembly with codegen
PYTHONPATH="$PYPATH" python -m ipu_apps.gen_pointwise_conv_asm \
    --in-channels 64 --out-channels 16 --output app.asm
```

Required pip packages: `click lark jinja2 pytest ml-dtypes numpy`
