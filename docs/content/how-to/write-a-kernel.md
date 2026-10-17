# Write a kernel

Write a kernel, the program and its declaration, then run, test and debug it.
The example is `scale`, which multiplies a matrix by a scalar: `y = x * s`. It
is the kernel in `kernels/elementwise/scale/`, so every listing below is a file
you can open. Every command runs from the repository root.

A kernel is a folder with three files the kernel harness reads, plus its Bazel file:

| File | What it holds |
|---|---|
| `scale.asm` | The program: an assembly template, rendered for each case. |
| `kernel.toml` | Everything else: parameters, regions, registers, cases, tolerances. |
| `reference.py` | What the outputs should be, in PyTorch. |
| `BUILD.bazel` | One line, `ipu_kernel()`, which declares the targets. |

The harness does the rest. From `kernel.toml` it checks the parameters,
places each region in XMEM, writes the inputs in the element format of the
run, programs the CRs, runs the program and compares the outputs with the
reference. A kernel has no Python harness and no hand-written CR setup.

## Before you start

- Pick the family: a folder under `kernels/`, such as `elementwise`. A new
  family is a folder with a `BUILD.bazel` of
  `load("//kernels:defs.bzl", "ipu_kernel_family")` and `ipu_kernel_family()`.
- Read the [assembly syntax](../reference/isa/assembly-syntax.md) if you have
  not written IPU assembly before, and keep the
  [instruction reference](../reference/isa/instructions.md) open: step 2 uses
  nine instructions. Both pages are generated in the docs build.

## 1. Create the folder

Create `kernels/elementwise/scale/` and its `BUILD.bazel`:

```python
load("//kernels:defs.bzl", "ipu_kernel")

ipu_kernel()
```

Then add the kernel to its family in `kernels/families.bzl`:

```python
KERNELS = {
    "elementwise": ["scale"],
}
```

Bazel cannot find subpackages on its own, so `//kernels:test_families` fails
until the folder is listed.

## 2. Write the program

`scale.asm` loads one row of `x` (128 elements), multiplies it by `s`, and
stores the product, once per row:

<!-- asm-kernel: kernels/elementwise/scale -->
```asm
{#- scale.asm -- y = x * s, one XMEM row (128 elements) per iteration.

    Design notes
    - MULT.VE multiplies the row in R0 by the element scalar in {{ cr.s }}.
      The scalar is decoded in the active element format (Q49), so the same
      program runs in FP32, INT8 and E4.
    - The product goes through ACC (first add), and ST.ACC stores the
      accumulator vector: y is an accumulator output, FP32 in FP32 and E4,
      INT32 in INT8. One accumulator vector is one row in FP32 and four rows
      in an 8-bit format, hence acc_rows.
    - Region bases and the row count are in rows, not bytes.
-#}
{%- set acc_rows = 1 if format in ("fp32", "int32") else 4 %}
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ST.ACC LR2 {{ cr.y }} ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 {{ acc_rows }} ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
```

An instruction word is one line ending in `;;`; instructions joined by `;`
issue together in that word. Line by line:

| Word | What it does |
|---|---|
| `SET LR0 CR0 ; SET LR1 CR0 ; SET LR2 CR0` | Zero three loop registers: `LR0` (an offset that stays 0), `LR1` (the row of `x`) and `LR2` (the row of `y`). `CR0` always reads 0. |
| `LD.MULT R0 LR1 {{ cr.x }}` | Load XMEM row `x + LR1` into `R0`. A row is 128 elements in every format. |
| `MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ; ACC.ADD.FIRST` | Multiply `R0`, from element `LR0` (0), by the scalar in `{{ cr.s }}`, with mask slot 0 and no mask shift (every lane, as `R_MASK` resets), and the dstructure in `CR15`. `ACC.ADD.FIRST` in the same word writes that product to the accumulator. |
| `ST.ACC LR2 {{ cr.y }}` | Store the accumulator to row `y + LR2`. |
| `ADD LR1 LR1 CR1 ; INC LR2 {{ acc_rows }}` | Step to the next row of `x` (`CR1` always reads 1) and the next accumulator vector of `y`. |
| `BLT.S LR1 {{ cr.rows }} row_loop` | Loop while `LR1` is less than `rows`. |
| `HALT` | End the run. |

`LD.MULT` has its own word because the multiply does not see a load from the
same word; `ACC.ADD.FIRST` does see the same word's product. The
[timing model](../explanation/timing-model.md) gives every such rule.

Three things make it a kernel program rather than a fixed one:

- **The header** is a `{#- ... -#}` comment: the first line says what the
  program computes, then its design notes. The kernel's generated page shows
  it, so write the notes there, not in a separate file.
- **`{{ cr.x }}`** is the register `kernel.toml` binds to `x`. The program
  never names a CR for a value; it asks for the value by name, and renders as
  `CR2` here. A name the `[abi]` table does not bind is an error at the line
  that uses it.
- **Template values**: every parameter (`{{ rows }}`), the element format of
  the run (`format`, such as `"e4"`) and each tensor's shape (`shapes.x`).
  `scale` uses `format` to step over one accumulator vector: 128 32-bit
  elements, which is one row in FP32 (512 bytes a row) and four rows in an
  8-bit format (128 bytes a row).

`CR0` reads 0, `CR1` reads 1 and `CR15` holds the dstructure (by default all
128 elements are valid). Mnemonics and register names are upper case, and
operands are separated by spaces.

## 3. Declare it in `kernel.toml`

```toml
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = { type = "int", min = 1, max = 256 }
s    = { type = "float" }                 # element scalar: encoded in the active format

[inputs.x]
shape = ["rows", 128]                     # automatic placement; add row = N to pin

[outputs.y]
shape = ["rows", 128]
store = "accumulator"                     # ST.ACC: FP32 (INT32 in INT8)

[abi]                                     # name -> CR; templates see {{ cr.x }} == "CR2"
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
reference = "reference.py:reference"
tolerance = { fp32 = "EXACT", int32 = "EXACT" }   # y, an accumulator output, is compared as FP32 (INT32 in INT8)

[cases.default]
rows = 4
s = 2.0

[cases.one_row]
rows = 1
s = -3.0

[cases.sweep]                             # one case per value: sweep[rows=1], sweep[rows=7], ...
rows = [1, 7, 33, 128]
s = 1.5                                   # exact in E4 and FP32, not in INT8
formats = ["fp32", "e4"]                  # per-case format limit

[cases.mine]                              # user data instead of the generator / reference
rows = 2
s = 2.0
inputs = { x = "data/x.npy" }             # small integers in [-8, 8]: exact in fp32, int8 and e4
expected = { y = "data/y.npy" }           # so y = 2x is exact in every format too
```

Section by section:

- **`[kernel]`**: `op` is the operation the kernel computes, the name routing
  answers a query with. `formats` lists the element formats it runs in; each
  case runs in every one of them unless it narrows the list.
- **`[params]`**: each parameter's type and domain. A case or a query outside
  the domain is refused before anything runs, with the reason. A `float`
  parameter bound to a CR is an element scalar: the harness encodes it in the
  format of the run, and a value that format cannot hold exactly (1.5 in
  INT8) is an error that names the nearest values. That is why `sweep` lists
  its own `formats`.
- **`[inputs.x]` and `[outputs.y]`**: each region's shape, in elements, with
  expressions over parameters (`"rows"`, `"ceil(k / 128)"`). The framework
  places them in XMEM; `row = N` pins one. `store = "accumulator"` says the
  program stores `y` from the accumulator (`ST.ACC`), so `y` is read back as
  FP32 (INT32 in INT8).
- **`[abi]`**: which register holds each value: a region's base row, an
  integer, or an element scalar. Every region must be bound, since the program
  can reach it only through a register.
- **`[golden]`**: the reference function and the tolerance tier per format
  (step 7).
- **`[cases.*]`**: named sets of parameter values. `[cases.default]` is
  required. A list makes a sweep: `sweep` runs as `sweep[rows=1]`,
  `sweep[rows=7]` and so on. `mine` replaces the generated input and the
  reference with files from `data/`.

Every key is listed in the [`kernel.toml` reference](../reference/kernel-toml.md).

## 4. Write the reference

`reference.py` takes the inputs as tensors, already decoded from the format
of the run, and every parameter as a keyword. It returns each output by name:

```python
import torch


def reference(x: torch.Tensor, *, rows: int, s: float) -> dict[str, torch.Tensor]:
    """y = x * s. The framework passes inputs decoded from the active format
    and rounds the result to that format before comparing."""
    return {"y": x * s}
```

The harness rounds the result to the format each output is stored in before
it compares: the format of the run, or for an accumulator output like `y` its
accumulator format (FP32, or INT32 in INT8). An operation other kernels share
has one reference in `ipu_kernels.reference.ops` instead; the kernel names it
with `[golden] op = "<name>"` and needs no `reference.py`.

## 5. Run its tests

```bash
bazel test //kernels/elementwise/scale
```

The target runs every case in every format it runs in, as one test each:
`test_case[default-fp32]`, `test_case[one_row-e4]`, `test_case[sweep[rows=7]-fp32]`.
A failing test names the case, the format, the output, the worst element and
the tier it failed at. A case with `tier = "nightly"` runs in
`//kernels/elementwise/scale:scale_nightly` instead, which only the nightly run
(`--config=full`) runs.

## 6. Run one case

`bazel run` on the same target runs one case through the kernel runner and
prints one report per format: the result, the cycle count and each output's
comparison.

```bash
bazel run //kernels/elementwise/scale -- --case one_row --format e4
bazel run //kernels/elementwise/scale -- --list-cases
bazel run //kernels/elementwise/scale -- --case default --format all
```

Options to look inside a run:

| To | Add |
|---|---|
| Use your own input | `--input x=x.npy` |
| Keep the outputs | `--output out` (one `.npy` per output) |
| Record every cycle | `--trace trace.jsonl` |
| Step through it in the debugger | `--debug` |
| Count the ISA idioms it uses | `--profile-aliases` |

For example:

```bash
bazel run //kernels/elementwise/scale -- --case one_row --format e4 --debug
```

Relative paths are read from where you run the command. Every option is in
the [command-line reference](../reference/cli.md#kernel-runner-run_kernel), and
[Debug a kernel](debug-a-kernel.md) walks through the debugger.

## 7. Choose a tolerance tier

`[golden] tolerance` names the tier each format is compared at; a format not
listed is compared at `EXACT`. The key is the format an output is compared
in: the format of the run for an element output, its accumulator format for an
accumulator output. So `scale` names `fp32` and `int32`, and a key no output is
compared in (`e4` here) is an error that says so:

| Tier | For |
|---|---|
| `EXACT` | Data movement, or one correctly rounded operation. |
| `FP32_ORDER` | FP32 sums whose order differs from the reference's. |
| `FP32_LONG` | Long reductions (K of 256 or more), reciprocal and square-root chains. |
| `TRANSCENDENT` | Exponentials, logarithms and other functions the hardware approximates. |
| `INT8_LSB` | One least significant bit after rounding to nearest. |
| `FP8_ULP` | One code step in an 8-bit float format. |

Start from the strictest tier that passes, and say why. A tier's numbers can
be overridden for one format, with the reason. For a kernel whose output is
stored in E4 (an element output):
`e4 = { tier = "FP8_ULP", ulps = 2, why = "two roundings: the product and the store" }`.

## When the kernel needs more

| The kernel needs | Use |
|---|---|
| An input or output in a layout other than row-major, such as channel-major data or packed weights | A pack transform: `pack = "<name>"` on the tensor, defined in the folder's `pack.py`, the family's `ipu_kernels.<family>.pack`, or the shared `ipu_kernels.pack`. |
| A test the generated cases cannot express, such as a property over many runs | A `test_*.py` in the folder. `bazel test` runs it too, with the fixtures `kernel`, `cases` and `run_case`. |
| Several programs that run one after another on the same XMEM | A stages hook: `[kernel] stages = "stages.py:run"`. |
| The same program at other shapes | A variant: a [`registry.toml` entry](../reference/kernel-toml.md#registrytoml) of four lines, not a new folder. |
| A region only some cases have, or a region placed before another | `when = "<bool param>"` on the tensor, or `after = "<tensor>"`. |
