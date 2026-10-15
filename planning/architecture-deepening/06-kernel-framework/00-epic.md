# Epic: Kernel framework (`ipu_harness`): a kernel is an `.asm`, a `kernel.toml` and an optional reference

ID E06 · Phase: after E03, E04, E05 · Followed by E07 (kernel migration)

## Goal

Writing and testing a kernel should take the assembly program, a short
declaration and, optionally, a PyTorch reference. Nothing else:

```
kernels/<family>/<name>/
  <name>.asm        # program + design notes in its header (Q79)
  kernel.toml       # op, formats, params, inputs/outputs, abi, cases, golden
  reference.py      # optional PyTorch reference (or a shared one, selected by op)
  data/*.npy        # optional user inputs / expected outputs
  BUILD.bazel       # one line: ipu_kernel()
```

```bash
bazel test //kernels/<family>/<name>                                   # every case x every format
bazel run  //kernels/<family>/<name> -- --case one_row --format e4 --debug
```

The framework (`ipu_harness`) is one deep module behind that small interface.
It validates the parameters once, places regions in XMEM, packs each tensor for
the active element format, programs the CRs, assembles the program with the
case parameters as template values, runs it to `HALT` on the E04 reference
model, unpacks the outputs, and compares them with the reference (rounded to the
format) or with user data, using a named tolerance tier. It reports cycles and
MAC utilisation and can write a trace.

## Background

Measured at HEAD 694a8a8. Paths are relative to `src/tools/ipu-apps/src/ipu_apps/`.

- **Volume.** A kernel today is 5 files: `app.py`, `cases.py`, `test.py`,
  `__init__.py` and `<name>.asm`. There are 89 kernels, 99 `app.py`, 99
  `cases.py`, 96 `test.py` and 102 `__init__.py`. The kernel tree holds 20,525
  lines of Python against 18,054 lines of assembly.
- **Hand-written harness work.** There are 512 `set_cr(` calls in kernel
  `app.py` files. `LANES`/`ROW_BYTES`/`ELEM_BYTES` are redefined in most
  harnesses, and every harness writes its own loaders and teardown. See
  `kernels/projections/app.py:164-245` for the cleanest of them.
- **Validation.** A query is validated 3 times on the `run_case` path:
  `kernel_registry/cases.py:137`, `registry.py:54`, and the constructor
  (`memory.py:62`, `kernels/softmax/app.py:108`).
- **Spec lookup and the import cycle.** The spec reaches the harness through
  MRO reflection (`registry.py:67-80`), via lazy imports that form a cycle
  (`base.py:95`, `memory.py:54`).
- **Wide FP32 only.** Every kernel runs in wide FP32 only
  (`ExecutionConfig(mode="fp32")`, e.g. `kernels/projections/app.py:382`), even
  though Q44 says kernels are not wide-only.
- **Debug and profiling in the harness.** `IpuApp.run` reads `IPU_DEBUG_TUI`
  (`base.py:91`) and writes alias profiles into state (`base.py:103-105`).
  Every case runs 3 times: `case_tests`, then baseline and profiled runs in
  `test/test_alias_profile_kernels.py:24-28`.
- **Whole-image kernels.** 15 kernels hand the harness a complete pre-built
  XMEM image (`kernel_registry/memory.py:1-5`, `MemoryApp`), so their
  `cases.py` own the layout.
- **Build graph and global state.** One `py_library` globs every kernel
  (`src/tools/ipu-apps/BUILD.bazel:13-22`), so one `cases.py` edit reruns all
  206 ipu-apps tests. The registry state lives in module globals
  (`registry.py:29` `_CACHE`, `layers.py:42` `_ADAPTERS`). 17 mobilevit test
  files `sys.path.insert` to import their fixtures.

## Decisions applied

| ID | Decision | Where |
|----|----------|-------|
| Q80 | Kernel = `.asm` + `kernel.toml` + optional `reference.py`. `bazel test` runs every case x format; `bazel run ... -- --case X --format e4 --debug`. Escape hatches: `pack.py`, `test_*.py`, stages hook. | all |
| Q81 | Automatic XMEM placement; optional `row =` pin | E06-04 |
| Q82 | Cases may name `.npy` inputs/expected; `--input x=path.npy` | E06-08, E06-10 |
| Q83 | Delete MemoryApp and the whole-image path; `pack.py` transforms one declared tensor | E06-05, E06-20 |
| Q47, Q50 | One element-format enum; 128 elements per vector, only width changes; packing per format | E06-05 |
| Q49 | CR scalars are decoded in the active format | E06-06 |
| Q57 | One LayerNorm reference matching the kernel (no eps); revisit with AaQ | E06-09 |
| Q69, Q70 | Packages `ipu_harness` + `ipu_kernels`; tests stay in kernel folders | E06-01, E06-13 |
| Q73 | Enormous tier out of the default run; nightly + `--config=full` | E06-13 |
| Q74, Q74b, Q75b | PyTorch reference (test-only dependency); keep comparing against the user's own data; little code beyond the assembly | E06-09, E06-10 |
| Q76 | Family `registry.toml` for shape variants (replaces L12) | E06-17 |
| Q79 | Design notes only in the `.asm` header; shown on the generated kernel page | E06-21 |
| Q62 | One-way dependencies, enforced by Bazel deps + import-graph test | E06-01, E06-18 |
| D3 | Refactors are output-identical; behaviour changes are separate PRs | all |
| L11-L25 | Accepted plan defaults for the kernel/registry clusters. Explicitly: L15 exact input sizes, L16 byte-unit kernels marked `unit = "bytes"`, L13 mobilevit experiments promoted or deleted (E07) | E06-05, E06-06 |

## Worked example and acceptance target: the `scale` kernel

`y = x * s`. Input `x` is `[rows, 128]`. The ABI is `x = CR2`, `y = CR3`,
`rows = CR4`, `s = CR5`. E06 is done when this kernel, written as below and with
no other file, passes `bazel test //kernels/elementwise/scale` in FP32, INT8 and
E4.

`kernels/elementwise/scale/scale.asm`:

```asm
{#- scale.asm -- y = x * s, one XMEM row (128 elements) per iteration.

    Design notes
    - MULT.VE multiplies the row in R0 by the element scalar in {{ cr.s }}.
      The scalar is decoded in the active element format (Q49), so the same
      program runs in FP32, INT8 and E4.
    - The product goes through ACC (first add) and ACTIVATE.QUANTIZE identity,
      then ST.AAQ stores one output row.
    - Region bases and the row count are in rows, not bytes.
-#}
    SET LR0 CR0 ;
    SET LR1 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15 ;
    ST.AAQ LR1 {{ cr.y }} ;;
    ADD LR1 LR1 CR1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
```

`kernels/elementwise/scale/kernel.toml`:

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

[abi]                                     # name -> CR; templates see {{ cr.x }} == "CR2"
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
reference = "reference.py:reference"      # default; or op = "scale" for a shared reference
tolerance = { fp32 = "EXACT", int8 = "EXACT", e4 = "FP8_ULP" }

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
inputs = { x = "data/x.npy" }
expected = { y = "data/y.npy" }
```

`kernels/elementwise/scale/reference.py`:

```python
import torch


def reference(x: torch.Tensor, *, s: float) -> dict[str, torch.Tensor]:
    """y = x * s. The framework passes inputs decoded from the active format
    and rounds the result to that format before comparing."""
    return {"y": x * s}
```

`kernels/elementwise/scale/BUILD.bazel`:

```python
ipu_kernel()
```

Expected developer experience:

```text
$ bazel test //kernels/elementwise/scale
//kernels/elementwise/scale:scale   PASSED   (7 cases, 17 case x format runs)

$ bazel run //kernels/elementwise/scale -- --case one_row --format e4
scale/one_row/e4: PASS  cycles=9  MAC util=…%  max|err|=0 (tier FP8_ULP)

$ bazel run //kernels/elementwise/scale -- --case default --format int8 --input x=/tmp/x.npy --debug
```

`sweep` expands to four cases, so the kernel has 7 cases. `sweep` limits
itself to FP32 and E4 because `s = 1.5` has no INT8 encoding; without the
`formats` line, validation (E06-03) rejects the case for INT8 with a located
error instead of silently rounding `s`.

## Scope

- New packages `ipu_harness` (framework) and `ipu_kernels` (kernel tree, shared
  family Python, shared references), with Bazel libraries and an import-graph
  test.
- The `kernel.toml` schema, its loader and its validation.
- Placement, per-format packing, CR programming, assembly, cases, the PyTorch
  reference, tolerance tiers, the runner, and the generated tests.
- The stages hook, in-memory chaining, the registry value, family
  `registry.toml`, per-family Bazel libraries, and the mobilevit fixtures library.
- Deleting MemoryApp and the whole-image path (after E07 moves its 15 users).
- The "write a kernel" how-to page.

## Out of scope

- Moving the existing kernels (E07).
- Element-format codecs (E03), the reference-model API, trace writer and
  format-independent handlers (E04), and the template render API and assembly
  session (E05).
- The AaQ model (Q25, Q35). Its tiers stay provisional.
- Generating the kernel docs site page itself (E08 `ipu_gen`, E09). E06-21
  provides the data.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|----|------|-------|------|------------|-------------------|
| E06-01 | [01-package-skeleton.md](01-package-skeleton.md) | Create `ipu_harness` and `ipu_kernels` packages, Bazel libraries and import rules | S | E03 (`ipu_isa`), E01 (import-graph test) | none |
| E06-02 | [02-kernel-toml-schema.md](02-kernel-toml-schema.md) | `kernel.toml` schema, loader and read-only typed view | M | E06-01, E03 (element formats) | none |
| E06-03 | [03-params-validate-once.md](03-params-validate-once.md) | Parameter domain and a single validation seam (`plan_case`) | S | E06-02 | none |
| E06-04 | [04-region-placement.md](04-region-placement.md) | Automatic XMEM region placement with optional row pins | S | E06-03 | none |
| E06-05 | [05-format-packing.md](05-format-packing.md) | Per-format packing, unpacking and named `pack` transforms | M | E06-04, E03 (codecs) | none |
| E06-06 | [06-cr-programming.md](06-cr-programming.md) | Program CRs from `[abi]`: region bases, integers, element scalars | S | E06-04, E06-05, E04 (Q49 behaviour) | none |
| E06-07 | [07-assembly-with-case-params.md](07-assembly-with-case-params.md) | Assemble with case parameters as template values (E05 session) | S | E06-03, E05 (render + session) | none |
| E06-08 | [08-cases.md](08-cases.md) | Cases: defaults, sweeps, generator escape hatch, user `.npy` data | M | E06-03 | none |
| E06-09 | [09-pytorch-reference.md](09-pytorch-reference.md) | PyTorch reference loading and the shared reference module | M | E06-02, E01 (lockfile) | none |
| E06-10 | [10-golden-compare-tiers.md](10-golden-compare-tiers.md) | Golden comparison: rounding to format and named tolerance tiers | S | E06-05, E06-08, E06-09 | none |
| E06-11 | [11-runner.md](11-runner.md) | Runner and CLI: run a case end to end, report cycles/MAC, trace | M | E06-05..E06-10, E04 (reference API, trace) | none |
| E06-12 | [12-runner-owns-debug-profile.md](12-runner-owns-debug-profile.md) | The runner owns the debug TUI launch and alias profiling | S | E06-11 | none |
| E06-13 | [13-generated-kernel-tests.md](13-generated-kernel-tests.md) | `ipu_kernel()` Bazel macro, generated case x format tests, and the `scale` kernel | M | E06-11, E06-12 | none (new kernel) |
| E06-14 | [14-stages-hook.md](14-stages-hook.md) | Stages hook for multi-stage kernels | S | E06-11 | none |
| E06-15 | [15-in-memory-chaining.md](15-in-memory-chaining.md) | Chain kernels in memory (one kernel's output region is the next one's input) | S | E06-11 | none |
| E06-16 | [16-registry-value.md](16-registry-value.md) | Routing registry as a value built from `kernel.toml` files | M | E06-03 | none |
| E06-17 | [17-family-registry-toml.md](17-family-registry-toml.md) | Family `registry.toml`: shape variants over a template | M | E06-13, E06-16 | none |
| E06-18 | [18-per-family-bazel-libraries.md](18-per-family-bazel-libraries.md) | Per-family Bazel libraries, `FAMILY_DEPS` and the family import-graph test | M | E06-13 | none |
| E06-19 | [19-mobilevit-fixtures-library.md](19-mobilevit-fixtures-library.md) | MobileViT fixtures as a `py_library` (no `sys.path` hacks) | S | none | none |
| E06-20 | [20-delete-memoryapp.md](20-delete-memoryapp.md) | Delete MemoryApp and the whole-image path | S | E07 (the 15 MemoryApp kernels migrated) | none |
| E06-21 | [21-kernel-howto-docs.md](21-kernel-howto-docs.md) | How-to "Write a kernel", `kernel.toml` reference, kernel-page data | S | E06-13, E06-17, E09 (docs skeleton) | none |

## Suggested order

```
E06-01 ──▶ E06-02 ──▶ E06-03 ──┬──▶ E06-04 ──▶ E06-05 ──▶ E06-06 ──┐
                                ├──▶ E06-07 ───────────────────────┤
                                ├──▶ E06-08 ──┐                     │
E06-02 ──▶ E06-09 ──────────────┴─────────────┴──▶ E06-10 ─────────┴──▶ E06-11 ──▶ E06-12 ──▶ E06-13
                                └──▶ E06-16 ─────────────────────────────────────────────────────┐
E06-13 ──┬──▶ E06-14                                                                               │
         ├──▶ E06-15                                                                               ▼
         ├──▶ E06-18                                                                  E06-17 ◀────┘
         └──▶ E06-21
E06-19 (independent, can land first)
E07 (all MemoryApp kernels) ──▶ E06-20
```

E06-13 is the first issue a kernel author sees: once it lands, `scale` runs in
all three formats. E07's pilot (projections) can start once E06-13 and E06-17
are merged.

## Open questions

- **Location of the kernel tree.** Q80 names the label
  `//kernels/<family>/<name>`. Q69 puts each package in one directory under
  `src/tools/`. These drafts assume (a): a top-level `kernels/` tree of Bazel
  packages (sources, `kernel.toml`, `reference.py`, data), with
  `src/tools/ipu-kernels/` holding the `ipu_kernels` Python package (shared
  family code, shared references, pack transforms). The alternative (b) puts
  the tree under `src/tools/ipu-kernels/kernels/` and gives up the short label.
  Decide in E06-01.
