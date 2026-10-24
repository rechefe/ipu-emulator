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
places each region in XMEM, writes the inputs in the variant of the run (in
NX8, each row with its metadata word), programs the CRs, runs the program and compares the outputs with the
reference. A kernel has no Python harness and no hand-written CR setup.
Every input is a tensor declared in `kernel.toml`: a kernel never supplies
its XMEM contents ready-made. To see the raw rows a run lays out, open the run
in the debugger and read the XMEM pane ([Debug a kernel](debug-a-kernel.md),
step 5).

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
    - MULT.VE multiplies the row in R0 by the MULT constant in {{ cr.s }}
      (float32 bits in FP32; ADR 0007), reading R0 with the QF half in
      {{ cr.qf }}.L (FP32 ignores it).
    - The product goes through ACC (first add), and ACTIVATE.EXACT identity
      stores the accumulator vector: one FP32 row per iteration.
    - Region bases and the row count are in rows, not bytes.
-#}
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.EXACT identity CR15.L {{ cr.y }} LR2 ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
```

An instruction word is one line ending in `;;`; instructions joined by `;`
issue together in that word. Line by line:

| Word | What it does |
|---|---|
| `SET LR0 CR0 ; SET LR1 CR0 ; SET LR2 CR0` | Zero three loop registers: `LR0` (an offset that stays 0), `LR1` (the row of `x`) and `LR2` (the row of `y`). `CR0` always reads 0. |
| `LD.MULT R0 LR1 {{ cr.x }}` | Load XMEM row `x + LR1` into `R0`. A row is 128 elements in both modes. |
| `MULT.VE LR0 {{ cr.s }} 0 LR0 CR15.L {{ cr.qf }}.L ; ACC.ADD.FIRST` | Multiply `R0`, from element `LR0` (0), by the MULT constant in `{{ cr.s }}`, with mask slot 0 and no mask shift (every lane, as `R_MASK` resets), the DS half `CR15.L` and the QF half `{{ cr.qf }}.L` (which FP32 ignores). `ACC.ADD.FIRST` in the same word writes that product to the accumulator. |
| `ACTIVATE.EXACT identity CR15.L {{ cr.y }} LR2` | Store the accumulator, unchanged, as one FP32 row at `y + LR2`. |
| `ADD LR1 LR1 CR1 ; INC LR2 1` | Step to the next row of `x` (`CR1` always reads 1) and of `y`. |
| `BLT.S LR1 {{ cr.rows }} row_loop` | Loop while `LR1` is less than `rows`. |
| `HALT` | End the run. |

`LD.MULT` has its own word because the multiply does not see a load from the
same word; `ACC.ADD.FIRST` does see the same word's product, and
`ACTIVATE.EXACT` the same word's accumulator. The
[timing model](../explanation/timing-model.md) gives every such rule.

Three things make it a kernel program rather than a fixed one:

- **The header** is a `{#- ... -#}` comment: the first line says what the
  program computes, then its design notes. The kernel's generated page shows
  it, so write the notes there, not in a separate file.
- **`{{ cr.x }}`** is the register `kernel.toml` binds to `x`. The program
  never names a CR for a value; it asks for the value by name, and renders as
  `CR2` here. A name the `[abi]` table does not bind is an error at the line
  that uses it.
- **Template values**: every parameter (`{{ rows }}`), the variant of the run
  (`format`, such as `"e4"` or `"fp32"`) and each tensor's shape
  (`shapes.x`). Every load and store moves one row in both modes, so `scale`
  needs none of them.

`CR0` reads 0, `CR1` reads 1 and `CR15.L` holds the DS half (by default all
128 elements are valid). `scale` stores with `ACTIVATE.EXACT`, the
emulator-only exact store, so it runs in the FP32 mode only; an NX8 kernel
stores with `ACTIVATE.QUANTIZE`, which quantizes each row to the case's QF
([`ACTIVATE.QUANTIZE`, `ACTIVATE.EXACT` and `LD.LUT`](../reference/isa/activate-quantize.md)). Mnemonics and register names are upper case, and
operands are separated by spaces.

## 3. Declare it in `kernel.toml`

The listing is the kernel's own file, included in the docs build, so it is
always the one the tests run:

```toml
--8<-- "kernels/elementwise/scale/kernel.toml"
```

Section by section:

- **`[kernel]`**: `op` is the operation the kernel computes, the name routing
  answers a query with. `formats` lists the variants it runs in: `fp32` for
  the emulator-only FP32 mode, an NX8 type (`e1` to `e7`, `dynamic`,
  `s<sign>e<fe>w<bits>`) for the hardware mode. Each case runs in every one
  of them unless it narrows the list.
- **`[params]`**: each parameter's type and domain. A case or a query outside
  the domain is refused before anything runs, with the reason. A `float`
  parameter bound to a CR is a MULT constant: its float32 bits in FP32, and
  in NX8 one element with its own scale and type; a value NX8 cannot hold
  exactly is an error that names the nearest values.
- **`[inputs.x]` and `[outputs.y]`**: each region's shape, in elements, with
  expressions over parameters (`"rows"`, `"ceil(k / 128)"`). The framework
  places them in XMEM; `row = N` pins one. Every store writes one row of the
  mode, so an output is read back in the variant of the run.
- **`[derived]`**: values computed from the case; `case_qf` is the QF half of
  the case's variant (0 in FP32), and `ds(...)`, `qf(...)` and
  `constant(...)` build the other CR halves and constants.
- **`[abi]`**: which register holds each value: a region's base row, an
  integer, or an element scalar. Every region must be bound, since the program
  can reach it only through a register: by itself, or through a bound
  `[derived]` value that reads its `base()`.
- **`[golden]`**: the reference function and the tolerance tier per variant
  (step 7).
- **`[cases.*]`**: named sets of parameter values. `[cases.default]` is
  required. A list makes a sweep: `sweep` runs as `sweep[rows=1]`,
  `sweep[rows=7]` and so on. `mine` replaces the generated input and the
  reference with files from `data/`.

Every key is listed in the [`kernel.toml` reference](../reference/kernel-toml.md).

## 4. Write the reference

`reference.py` takes the inputs as tensors, already decoded from the variant
of the run, and every parameter as a keyword. It returns each output by name:

```python
--8<-- "kernels/elementwise/scale/reference.py"
```

The harness rounds the result to the variant of the run before it compares:
in NX8 it quantizes each golden row as `ACTIVATE.QUANTIZE` does. An operation other kernels share
has one reference in `ipu_kernels.reference.ops` instead; the kernel names it
with `[golden] op = "<name>"` and needs no `reference.py`. The second worked
example, `kernels/reshape/identity/` (`y = x`), does that: its `kernel.toml`
has `op = "identity"`, and its cases keep their data in a `gen.py` generator.

### Test against your own data

A case can name its own files instead: `inputs` replaces the generated input,
and `expected` replaces the reference for that output. `scale`'s `mine` case
reads both from `data/`, saved with `numpy.save` in the tensor's shape
(`[rows, 128]`). Each file is rounded to the variant of the run, so pick values
it holds exactly, as `mine` does (small integers), or the comparison
sees the rounding. To try other data without editing `kernel.toml`, pass
`--input x=my_x.npy` to a run (step 6).

## 5. Run its tests

```bash
bazel test //kernels/elementwise/scale
```

The target runs every case in every variant it runs in, as one test each:
`test_case[default-fp32]`, `test_case[one_row-fp32]`, `test_case[sweep[rows=7]-fp32]`.
A failing test reports the case, the variant, the output, the worst element and
the tier it failed at, in the same line a run prints. For example, `mine` run
with one element of `x` changed (`--input`, step 6):

```text
scale/mine/fp32: FAIL  cycles=12  MAC util=16.7%  y: FAIL at EXACT (max|err|=2 max rel=0.166667); worst at [1, 5]: got 14.0, want 12.0
```

Row 1, lane 5 of `y` is 14 where `2 * x` is 12. A run that faults instead
names the fault, the PC and the source line of the word.

A case with `tier = "nightly"` runs in
`//kernels/elementwise/scale:scale_nightly` instead, which only the nightly run
(`--config=full`) runs.

## 6. Run one case

`bazel run` on the same target runs one case through the kernel runner and
prints one report per variant: the result, the cycle count and each output's
comparison.

```bash
bazel run //kernels/elementwise/scale -- --case one_row --format fp32
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
bazel run //kernels/elementwise/scale -- --case one_row --format fp32 --debug
```

Relative paths are read from where you run the command. Every option is in
the [command-line reference](../reference/cli.md#kernel-runner-kernel-targets-and-run_kernel).

When an output is wrong, run the failing case with `--debug`: the debugger
stops before the first word, and you step through the program watching the
registers, the accumulator and XMEM, with the source line of each word.
[Debug a kernel](debug-a-kernel.md) walks through it, and the
[Cycle Inspector](step-through-a-kernel-run.md) shows a recorded run in the browser.

## 7. Choose a tolerance tier

`[golden] tolerance` names the tier each variant is compared at; a variant not
listed is compared at `EXACT`. The key is a variant name (`fp32`, `e4`) or the
mode `nx8` for every NX8 variant. A key the kernel does not run in is an error
that says so. The tiers:

| Tier | For |
|---|---|
| `EXACT` | Data movement, or one correctly rounded operation. |
| `FP32_ORDER` | FP32 sums whose order differs from the reference's. |
| `FP32_LONG` | Long reductions (K of 256 or more), reciprocal and square-root chains. |
| `TRANSCENDENT` | Exponentials, logarithms and other functions the hardware approximates. |
| `NX8_STEP` | One step of the NX8 golden row's grid: the scale and the element type set the step. |

Start from the strictest tier that passes, and say why. A tier's numbers can
be overridden for one variant, with the reason. For a kernel whose e4 rows
round twice:
`e4 = { tier = "NX8_STEP", ulps = 2, why = "two roundings: the product and the store" }`.

## Add an NX8 variant to an FP32 kernel

A kernel written for the FP32 mode runs on the hardware only once it runs in
NX8. Adding the variants is its own change. The worked examples are
`kernels/conformance/vector_tour` (e3, e4, e5, dynamic) and
`kernels/linear/fully_connected` (e4, e5, dynamic):

1. **Store with the hardware instruction.** `ACTIVATE.EXACT` is FP32 only, and
   the harness refuses it for an NX8 case. Replace each store with
   `ACTIVATE.QUANTIZE` and an activation it has (`identity`, `relu`, `relu6`,
   `reciprocal`, `rsqrt`, `exp2`, `generic`). A kernel that needs another
   function stays FP32: the projections store with SiLU, so they list `fp32`
   only, and `kernels/projections/test_formats.py` pins that an NX8 variant is
   refused.
2. **Bind the QF.** Every MULT and `ACTIVATE.QUANTIZE` names a QF half:
   `[derived] qf = "case_qf"`, a CR for it in `[abi]`, and `{{ cr.qf }}.L` in
   the program.
3. **Give the inputs a range for the type.** `range = { nx8 = [-4, 4] }`
   keeps the data inside what a row's grid holds well; the harness writes
   each input row with its metadata.
4. **Add the variants and their tier.** Add them to `[kernel] formats` and
   name the tier in `[golden] tolerance` (`nx8 = "NX8_STEP"`), then run every
   case: `bazel run //kernels/linear/fully_connected -- --case default --format all`
   prints each output's largest error. Record what you measured beside the
   tier.

## When the kernel needs more

| The kernel needs | Use |
|---|---|
| An input or output in a layout other than row-major, such as channel-major data or packed weights | A pack transform: `pack = "<name>"` on the tensor, defined in the kernel folder's `pack.py` (one kernel's), the family folder's `kernels/<family>/pack.py` (shared by the family's kernels), or the shared `ipu_kernels.pack` (every family's). A family's transforms go in `kernels/<family>/pack.py`. Only when other Python imports names from them does the family put them in `ipu_kernels.<family>.pack` instead: the harness loads a folder's `.py` files by path, so one cannot import another (softmax's `scratch.py` imports `partition_size`). |
| A test the generated cases cannot express, such as a property over many runs | A `test_*.py` in the folder. `bazel test` runs it too, with the fixtures `kernel`, `cases` and `run_case`. |
| Several programs that run one after another on the same XMEM | A stages hook: `[kernel] stages = "stages.py:run"`. |
| The same program at other shapes | A variant: a [`registry.toml` entry](../reference/kernel-toml.md#registrytoml) of four lines, not a new folder. |
| A region only some cases have, or a region placed before another | `when = "<bool param>"` on the tensor (or `"not <bool param>"`, or terms joined by `and` and `or`, `and` binding tighter), or `after = "<tensor>"`. |
