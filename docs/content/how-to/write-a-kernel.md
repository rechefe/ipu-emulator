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

The listing is the kernel's own file, included in the docs build, so it is
always the one the tests run:

```toml
--8<-- "kernels/elementwise/scale/kernel.toml"
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
  can reach it only through a register: by itself, or through a bound
  `[derived]` value that reads its `base()`.
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
--8<-- "kernels/elementwise/scale/reference.py"
```

The harness rounds the result to the format each output is stored in before
it compares: the format of the run, or for an accumulator output like `y` its
accumulator format (FP32, or INT32 in INT8). An operation other kernels share
has one reference in `ipu_kernels.reference.ops` instead; the kernel names it
with `[golden] op = "<name>"` and needs no `reference.py`. The second worked
example, `kernels/reshape/identity/` (`y = x`), does that: its `kernel.toml`
has `op = "identity"`, and its cases keep their data in a `gen.py` generator.

### Test against your own data

A case can name its own files instead: `inputs` replaces the generated input,
and `expected` replaces the reference for that output. `scale`'s `mine` case
reads both from `data/`, saved with `numpy.save` in the tensor's shape
(`[rows, 128]`). Each file is rounded to the format of the run, so pick values
the format holds exactly, as `mine` does (small integers), or the comparison
sees the rounding. To try other data without editing `kernel.toml`, pass
`--input x=my_x.npy` to a run (step 6).

## 5. Run its tests

```bash
bazel test //kernels/elementwise/scale
```

The target runs every case in every format it runs in, as one test each:
`test_case[default-fp32]`, `test_case[one_row-e4]`, `test_case[sweep[rows=7]-fp32]`.
A failing test reports the case, the format, the output, the worst element and
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
the [command-line reference](../reference/cli.md#kernel-runner-kernel-targets-and-run_kernel).

When an output is wrong, run the failing case with `--debug`: the debugger
stops before the first word, and you step through the program watching the
registers, the accumulator and XMEM, with the source line of each word.
[Debug a kernel](debug-a-kernel.md) walks through it, and the
[Cycle Inspector](step-through-a-kernel-run.md) shows a recorded run in the browser.

## 7. Choose a tolerance tier

`[golden] tolerance` names the tier each format is compared at; a format not
listed is compared at `EXACT`. The key is the format an output is compared
in: the format of the run for an element output, its accumulator format for an
accumulator output. So `scale` names `fp32` and `int32`, and a key no output is
compared in (`e4` here) is an error that says so. The tiers:

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

## Add a format to a migrated kernel

A kernel migrated from the old Python harness (E07) ran in FP32 first, on its
old data, so its outputs matched the old harness's. Adding a format afterwards is its own change.
The worked example is the projections (`kernels/projections/`), whose
`proj_p4` template runs in FP32 and INT8:

1. **Check that the program does not depend on the format.** Region bases
   and strides in CRs count rows, not bytes; the program holds no byte
   offsets; each output's `store` matches its store instruction (`element`
   for `ST.AAQ`, `accumulator` for `ST.ACC`, which takes 4 rows a vector in an
   8-bit format). If one fails, fix the program in its own change first, and
   show that the FP32 outputs are unchanged.
2. **Keep the old data in FP32 only.** The case generator that reproduces
   the old data returns nothing in another format, so the harness's default
   generator makes those inputs. The projections' `gen.py`:

   ```python
   def seeded_uniform(plan, rng):
       if plan.format is not ElementFormat.FP32:
           return {}
       ...
   ```

3. **Give the inputs a range for the format.** The default range of an
   integer format is all of it, and a long dot product of full-range INT8
   values saturates almost every output, so the test would pass whatever the
   program did. Choose a range that keeps fewer than 1% of the outputs at the
   rails, and test that it does:

   ```toml
   [inputs.b]
   shape = ["n_streams", "n_tg", "k", "n_tok"]
   pack = "channel_major"
   range = { int8 = [-2, 2] }   # at most 0.36% of y saturates (K = 480)
   ```

4. **Add the format and its tier.** Add it to `[kernel] formats` and name its
   tier in `[golden] tolerance`, then run the kernel and every registry entry
   over it: `bazel run //kernels/projections:proj_qkv_192_p4 -- --format int8`
   prints each output's largest error. Record what you measured beside the
   tier.
5. **Leave out what the emulator cannot run yet, and say why.** The
   projections do not run in E4, because `ACTIVATE.QUANTIZE` refuses an 8-bit
   float format (OQ-20), and the two-group templates do not run in INT8,
   because `ST.AAQ` stores four rows in an 8-bit format and their store order
   overwrites rows already stored (OQ-35). A test that fails when the
   emulator changes (`kernels/projections/test_formats.py`) records each one.

## When the kernel needs more

| The kernel needs | Use |
|---|---|
| An input or output in a layout other than row-major, such as channel-major data or packed weights | A pack transform: `pack = "<name>"` on the tensor, defined in the kernel folder's `pack.py` (one kernel's), the family folder's `kernels/<family>/pack.py` (shared by the family's kernels), or the shared `ipu_kernels.pack` (every family's). A family's transforms go in `kernels/<family>/pack.py`. Only when other Python imports names from them does the family put them in `ipu_kernels.<family>.pack` instead: the harness loads a folder's `.py` files by path, so one cannot import another (softmax's `scratch.py` imports `partition_size`). |
| A test the generated cases cannot express, such as a property over many runs | A `test_*.py` in the folder. `bazel test` runs it too, with the fixtures `kernel`, `cases` and `run_case`. |
| Several programs that run one after another on the same XMEM | A stages hook: `[kernel] stages = "stages.py:run"`. |
| The same program at other shapes | A variant: a [`registry.toml` entry](../reference/kernel-toml.md#registrytoml) of four lines, not a new folder. |
| A region only some cases have, or a region placed before another | `when = "<bool param>"` on the tensor (or `"not <bool param>"`, or terms joined by `and` and `or`, `and` binding tighter), or `after = "<tensor>"`. |
