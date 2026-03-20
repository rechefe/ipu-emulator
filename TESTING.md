# Running Tests

This guide walks through running the IPU emulator tests, using the
**fully-connected (FC)** application as an example. The same pattern applies to
every other application — only the file paths and environment variable names
change.

## Prerequisites

- Python 3.10+
- Git (to clone the repo)

## 1. Install Python dependencies

Open a terminal in VS Code (`Ctrl+`` `) and run:

```bash
pip install click lark jinja2 pytest numpy ml-dtypes
```

These are the only external packages needed across all four sub-projects:

| Package     | Used by          |
|-------------|------------------|
| click       | ipu-as-py (CLI)  |
| lark        | ipu-as-py (parser)|
| jinja2      | ipu-as-py        |
| pytest      | all tests        |
| numpy       | ipu-emu-py, ipu-apps |
| ml-dtypes   | ipu-emu-py (FP8 support) |

## 2. Choose your method

### Method A — Bazel (CI-style, Linux/Mac)

If you have Bazel (or Bazelisk) installed:

```bash
bazel test //...  --test_output=errors
```

This assembles all `.asm` files, sets environment variables, and runs every
test automatically. **No further setup needed.**

> Bazel may not work on Windows due to application-control policies. Use
> Method B instead.

---

### Method B — Direct Python (works everywhere)

This is a 3-step process: set `PYTHONPATH`, assemble the `.asm` file, then
run pytest with the right environment variables.

All commands below assume your terminal is at the **repository root**.

#### Step 1 — Set PYTHONPATH

This tells Python where to find the four sub-packages. Run this once per
terminal session.

**Git Bash / WSL / Linux / Mac:**
```bash
export PYPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"
```

**PowerShell:**
```powershell
$env:PYPATH = "$(Get-Location)\src\tools\ipu-emu-py\src;$(Get-Location)\src\tools\ipu-common\src;$(Get-Location)\src\tools\ipu-apps\src;$(Get-Location)\src\tools\ipu-as-py\src"
```

#### Step 2 — Assemble the `.asm` file

Each application has an assembly source that must be compiled to a `.bin`
before tests can use it. For the FC app:

**Git Bash / WSL / Linux / Mac:**
```bash
PYTHONPATH="$PYPATH" python -m ipu_as.cli assemble \
    --input src/tools/ipu-apps/src/ipu_apps/fully_connected/fully_connected.asm \
    --output /tmp/fc.bin \
    --format bin
```

**PowerShell:**
```powershell
$env:PYTHONPATH = $env:PYPATH
python -m ipu_as.cli assemble `
    --input src/tools/ipu-apps/src/ipu_apps/fully_connected/fully_connected.asm `
    --output $env:TEMP/fc.bin `
    --format bin
```

#### Step 3 — Run the test

Each app test reads two environment variables: one pointing to the assembled
binary, and one pointing to the directory with golden test data (inputs +
expected outputs).

**Git Bash / WSL / Linux / Mac:**
```bash
PYTHONPATH="$PYPATH" \
FC_INST_BIN=/tmp/fc.bin \
FC_DATA_DIR=src/tools/ipu-apps/src/ipu_apps/fully_connected/test_data_format \
python -m pytest src/tools/ipu-apps/test/test_fully_connected.py -v
```

**PowerShell:**
```powershell
$env:PYTHONPATH = $env:PYPATH
$env:FC_INST_BIN = "$env:TEMP/fc.bin"
$env:FC_DATA_DIR = "src/tools/ipu-apps/src/ipu_apps/fully_connected/test_data_format"
python -m pytest src/tools/ipu-apps/test/test_fully_connected.py -v
```

### What to expect

The test runs the FC kernel in the emulator for three data types (int8,
fp8_e4m3, fp8_e5m2) and compares the output against pre-computed golden
binaries. A passing run looks like:

```
test_fully_connected.py::test_fc[int8-INT8-out_int8_acc_int32.bin]       PASSED
test_fully_connected.py::test_fc[fp8_e4m3-FP8_E4M3-out_fp8_e4m3_acc_fp32.bin] PASSED
test_fully_connected.py::test_fc[fp8_e5m2-FP8_E5M2-out_fp8_e5m2_acc_fp32.bin] PASSED
```

## 3. Running emulator unit tests (no env vars needed)

The emulator's own tests don't require assembly or environment variables:

```bash
PYTHONPATH="$PYPATH" python -m pytest src/tools/ipu-emu-py/test/ -v
```

PowerShell:
```powershell
$env:PYTHONPATH = $env:PYPATH
python -m pytest src/tools/ipu-emu-py/test/ -v
```

## 4. Running assembler tests

```bash
PYTHONPATH="$PYPATH" python -m pytest src/tools/ipu-as-py/test/ -v
```

## Quick-reference: environment variables per app

Every app test expects `<PREFIX>_INST_BIN` (path to assembled binary) and
`<PREFIX>_DATA_DIR` (path to test data). The table below lists the prefix and
assembly source for each app:

| App                          | Env prefix         | Assembly source (under `src/tools/ipu-apps/src/ipu_apps/`) |
|------------------------------|--------------------|------------------------------------------------------------|
| fully_connected              | `FC`               | `fully_connected/fully_connected.asm`                      |
| depthwise_conv               | `DCONV`            | `depthwise_conv/depthwise_conv.asm`                        |
| depthwise_conv_64x64x1       | `DCONV64`          | `depthwise_conv_64x64x1/depthwise_conv_64x64x1.asm`       |
| depthwise_conv_128x128x8     | `DCONV8`           | `depthwise_conv_128x128x8/depthwise_conv_128x128x8.asm`   |
| conv2ch                      | `CONV2CH`          | `conv2ch/conv2ch.asm`                                      |
| conv2ch2ch                   | `CONV2CH2CH`       | `conv2ch2ch/conv2ch2ch.asm`                                |
| pointwise_conv               | `PCONV`            | `pointwise_conv/pointwise_conv.asm`                        |
| pointwise_conv_32x32x16      | `PCONV16`          | `pointwise_conv_32x32x16/pointwise_conv_32x32x16.asm`     |
| pointwise_conv_32x32x32      | `PCONV32`          | `pointwise_conv_32x32x32/pointwise_conv_32x32x32.asm`     |
| conv_128x128_1to8            | `CONV1TO8`         | `conv_128x128_1to8/conv_128x128_1to8.asm`                  |
| conv_128x128_4to8            | `CONV4TO8`         | `conv_128x128_4to8/conv_128x128_4to8.asm`                  |
| conv_128x128_8to16           | `CONV8TO16`        | `conv_128x128_8to16/conv_128x128_8to16.asm`                |

The data dir is always `<app_dir>/test_data_format` relative to the ipu-apps
source root.
