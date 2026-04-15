# Developer Workflow Guide

This guide walks you through the day-to-day workflow: writing assembly,
assembling it, running tests, and debugging. It covers both direct Python
(works everywhere) and Bazel (Linux/Mac CI).

---

## 1. Environment Setup

### Prerequisites

- Python 3.10+
- Git

### Install Python dependencies

```bash
pip install click lark jinja2 pytest numpy ml-dtypes
```

| Package   | Used by              |
|-----------|----------------------|
| click     | ipu-as-py (CLI)      |
| lark      | ipu-as-py (parser)   |
| jinja2    | ipu-as-py (templates)|
| pytest    | all tests            |
| numpy     | ipu-emu-py, ipu-apps |
| ml-dtypes | ipu-emu-py (FP8)     |

### Set PYTHONPATH

Run this once per terminal session from the **repository root**:

**Git Bash / WSL / Linux / Mac:**
```bash
export PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"
```

**PowerShell:**
```powershell
$env:PYTHONPATH = "$(Get-Location)\src\tools\ipu-emu-py\src;$(Get-Location)\src\tools\ipu-common\src;$(Get-Location)\src\tools\ipu-apps\src;$(Get-Location)\src\tools\ipu-as-py\src"
```

> **Why?** The project has four Python packages (`ipu_common`, `ipu_as`,
> `ipu_emu`, `ipu_apps`) that need to find each other. PYTHONPATH tells Python
> where they live.

---

## 2. Write Assembly

Assembly files live alongside their Python harness:

```
src/tools/ipu-apps/src/ipu_apps/
  └── my_app/
      ├── my_app.asm          ← your assembly code
      ├── __init__.py          ← Python app class (setup/teardown)
      ├── __main__.py          ← debug runner (optional)
      └── test_data_format/    ← golden input/output binaries
```

The assembly syntax reference is in `docs/assembly-syntax.md`. Key rules:

- `;;` ends a VLIW word (one cycle)
- `;` separates sub-instructions within a word
- `break ;;` inserts a debug breakpoint

Example (fully connected inner loop):

```asm
element_loop:
    ldr_mult_reg        mem_bypass lr4 cr1;
    incr                lr4 128;
    incr                lr5 1;
    mult.ve             mem_bypass lr5 lr15 lr15;
    acc;
    blt                 lr5 lr6 element_loop;;
```

For the full guide on building a new application from scratch (app class,
`__main__.py`, test data, BUILD rules), see `docs/building-applications.md`.

---

## 3. Assemble

The assembler compiles `.asm` files into `.bin` binaries that the emulator
can execute.

### Direct Python

```bash
python -m ipu_as.cli assemble \
    --input src/tools/ipu-apps/src/ipu_apps/fully_connected/fully_connected.asm \
    --output /tmp/fc.bin \
    --format bin
```

**PowerShell:**
```powershell
python -m ipu_as.cli assemble `
    --input src\tools\ipu-apps\src\ipu_apps\fully_connected\fully_connected.asm `
    --output $env:TEMP\fc.bin `
    --format bin
```

Replace the paths for your app. The general pattern is:

```
python -m ipu_as.cli assemble \
    --input src/tools/ipu-apps/src/ipu_apps/<APP_NAME>/<APP_NAME>.asm \
    --output /tmp/<APP_NAME>.bin \
    --format bin
```

### Bazel (Linux/Mac)

```bash
bazel build //src/tools/ipu-apps:assemble_fully_connected
```

The assembled binary lands in `bazel-bin/`.

---

## 4. Run Tests

Tests compare emulator output against pre-computed golden reference files.
Each app test needs two environment variables:

| Variable            | Points to                                 |
|---------------------|-------------------------------------------|
| `<PREFIX>_INST_BIN` | The assembled `.bin` from step 3           |
| `<PREFIX>_DATA_DIR` | Directory with golden input/output binaries|

### Direct Python

**Git Bash / WSL / Linux / Mac:**
```bash
FC_INST_BIN=/tmp/fc.bin \
FC_DATA_DIR=src/tools/ipu-apps/src/ipu_apps/fully_connected/test_data_format \
python -m pytest src/tools/ipu-apps/test/test_fully_connected.py -v
```

**PowerShell:**
```powershell
$env:FC_INST_BIN = "$env:TEMP\fc.bin"
$env:FC_DATA_DIR = "src\tools\ipu-apps\src\ipu_apps\fully_connected\test_data_format"
python -m pytest src\tools\ipu-apps\test\test_fully_connected.py -v
```

### Bazel (Linux/Mac)

Bazel assembles, sets env vars, and runs the test in one command:

```bash
bazel test //src/tools/ipu-apps:test_fully_connected --test_output=errors
```

Or run everything:

```bash
bazel test //... --test_output=errors
```

### Expected output

A passing run looks like:

```
test_fully_connected.py::test_fc[int8-INT8-out_int8_acc_int32.bin]       PASSED
test_fully_connected.py::test_fc[fp8_e4m3-FP8_E4M3-out_fp8_e4m3_acc_fp32.bin] PASSED
test_fully_connected.py::test_fc[fp8_e5m2-FP8_E5M2-out_fp8_e5m2_acc_fp32.bin] PASSED
```

### Unit tests (no env vars needed)

The emulator and assembler have their own unit tests that don't require
assembly or environment variables:

```bash
# Emulator tests
python -m pytest src/tools/ipu-emu-py/test/ -v

# Assembler tests
python -m pytest src/tools/ipu-as-py/test/ -v
```

### Environment variable prefixes per app

| App                          | Prefix              | Assembly source (relative to `ipu_apps/`)     |
|------------------------------|---------------------|------------------------------------------------|
| fully_connected              | `FC`                | `fully_connected/fully_connected.asm`          |
| depthwise_conv               | `DCONV`             | `depthwise_conv/depthwise_conv.asm`            |
| depthwise_conv_64x64x1       | `DCONV64`           | `depthwise_conv_64x64x1/depthwise_conv_64x64x1.asm` |
| depthwise_conv_128x128x8     | `DCONV8`            | `depthwise_conv_128x128x8/depthwise_conv_128x128x8.asm` |
| conv2ch                      | `CONV2CH`           | `conv2ch/conv2ch.asm`                          |
| conv2ch2ch                   | `CONV2CH2CH`        | `conv2ch2ch/conv2ch2ch.asm`                    |
| pointwise_conv               | `PCONV`             | `pointwise_conv/pointwise_conv.asm`            |
| pointwise_conv_32x32x16      | `PCONV16`           | `pointwise_conv_32x32x16/pointwise_conv_32x32x16.asm` |
| pointwise_conv_32x32x32      | `PCONV32`           | `pointwise_conv_32x32x32/pointwise_conv_32x32x32.asm` |
| conv_128x128_1to8            | `CONV1TO8`          | `conv_128x128_1to8/conv_128x128_1to8.asm`      |
| conv_128x128_4to8            | `CONV4TO8`          | `conv_128x128_4to8/conv_128x128_4to8.asm`      |
| conv_128x128_8to16           | `CONV8TO16`         | `conv_128x128_8to16/conv_128x128_8to16.asm`    |

The test data directory is always `<app_dir>/test_data_format`.

---

## 5. Debug

The emulator has an interactive debugger. It pauses at `break` instructions
in your assembly and drops you into a prompt where you can inspect and
modify state.

### Add breakpoints to your assembly

```asm
break ;;                  # always stop here
break.ifeq lr5 10 ;;     # stop only when lr5 == 10
```

### Run with the debug CLI

Each app has a `__main__.py` that enables interactive debugging. Run it
directly (not via `bazel test`, which captures stdout):

**Git Bash / WSL / Linux / Mac:**
```bash
FC_INST_BIN=/tmp/fc.bin \
FC_DATA_DIR=src/tools/ipu-apps/src/ipu_apps/fully_connected/test_data_format \
python -m ipu_apps.fully_connected --dtype INT8
```

**PowerShell:**
```powershell
$env:FC_INST_BIN = "$env:TEMP\fc.bin"
$env:FC_DATA_DIR = "src\tools\ipu-apps\src\ipu_apps\fully_connected\test_data_format"
python -m ipu_apps.fully_connected --dtype INT8
```

**Bazel (Linux/Mac):**
```bash
bazel run //src/tools/ipu-apps:fully_connected -- --dtype INT8
```

### Debug prompt commands

When the emulator hits a `break`, you get an interactive prompt:

```
========================================
IPU Debug - Break at PC=3
========================================
=== LR Registers ===
  lr 0 =          0 (0x00000000)
  lr 1 =       1280 (0x00000500)
  ...

debug >>>
```

**Navigation:**

| Command        | What it does                          |
|----------------|---------------------------------------|
| `continue` / `c` | Run until the next break           |
| `step`         | Execute one instruction, then stop    |
| `quit` / `q`   | Stop execution and exit               |

**Inspect registers:**

| Command   | What it does                                |
|-----------|---------------------------------------------|
| `regs`    | Print all registers                         |
| `lr`      | Print loop registers (lr0–lr15)             |
| `cr`      | Print control registers (cr0–cr15)          |
| `pc`      | Print program counter                       |
| `acc`     | Print accumulator (128 × 32-bit words)      |
| `r`       | Print R registers (mult stage)              |
| `rcyclic` | Print R cyclic register (512 bytes)         |
| `rmask`   | Print R mask register                       |

**Read specific values:**

```
debug >>> get lr0              # single register
debug >>> get r0 0 32          # 32 bytes from r0 starting at offset 0
debug >>> getw acc 0 16        # first 16 words (32-bit) of accumulator
debug >>> getw rcyclic 32 4    # 4 words from word offset 32
```

**Modify state:**

```
debug >>> set lr0 100
debug >>> set cr5 0x8000
debug >>> set pc 10
```

**Other:**

| Command   | What it does                                |
|-----------|---------------------------------------------|
| `disasm`  | Disassemble the current instruction         |
| `save <file>.json` | Save all register state to a JSON file |

### Debugging tips

- Use `break.ifeq lr5 10 ;;` to stop at a specific loop iteration instead
  of breaking every time.
- Use `getw acc 0 16` (not `get acc`) for the accumulator — it stores
  32-bit values.
- Step through a loop a few times watching a register:
  `step` → `get lr5` → `step` → `get lr5`.
- Save state at key points for offline comparison:
  `save before_mult.json` → `continue` → `save after_mult.json`.

---

## Quick Reference: Full Workflow Example

Complete example for the `fully_connected` app on Git Bash:

```bash
# 1. Set PYTHONPATH (once per terminal)
export PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src"

# 2. Assemble
python -m ipu_as.cli assemble \
    --input src/tools/ipu-apps/src/ipu_apps/fully_connected/fully_connected.asm \
    --output /tmp/fc.bin \
    --format bin

# 3. Test
FC_INST_BIN=/tmp/fc.bin \
FC_DATA_DIR=src/tools/ipu-apps/src/ipu_apps/fully_connected/test_data_format \
python -m pytest src/tools/ipu-apps/test/test_fully_connected.py -v

# 4. Debug (interactive)
FC_INST_BIN=/tmp/fc.bin \
FC_DATA_DIR=src/tools/ipu-apps/src/ipu_apps/fully_connected/test_data_format \
python -m ipu_apps.fully_connected --dtype INT8
```

---

## Project Structure at a Glance

```
src/tools/
├── ipu-common/    Shared types, instruction spec, register schema
├── ipu-as-py/     Assembler (.asm → .bin)
├── ipu-emu-py/    Emulator + debug CLI
└── ipu-apps/      Applications (one subdir per app)
    ├── src/ipu_apps/
    │   ├── fully_connected/
    │   │   ├── fully_connected.asm
    │   │   ├── __init__.py         (app class)
    │   │   ├── __main__.py         (debug runner)
    │   │   └── test_data_format/   (golden data)
    │   ├── depthwise_conv/
    │   ├── pointwise_conv/
    │   └── ...
    └── test/
        ├── test_fully_connected.py
        ├── test_depthwise_conv.py
        └── ...
```

## Further Reading

- `docs/assembly-syntax.md` — Complete assembly language reference
- `docs/building-applications.md` — How to create a new app from scratch
- `docs/debugging.md` — Full debug CLI reference
- `docs/setup.md` — Bazel/Docker/Dev Container setup
