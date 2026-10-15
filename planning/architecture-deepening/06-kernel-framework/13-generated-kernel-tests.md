# `ipu_kernel()` Bazel macro, generated case x format tests, and the `scale` kernel

Part of [Epic: Kernel framework](00-epic.md) · ID E06-13 · Size M · Depends on: E06-11, E06-12 · Behaviour change: none (adds a new kernel)

## Goal

`bazel test //kernels/<family>/<name>` runs every case in every format, and
`bazel run //kernels/<family>/<name> -- ...` runs one, with no `__init__.py`,
no `test.py` and no `app.py` in the kernel folder. This issue lands the epic's
acceptance kernel, `scale`.

## Background

- Every kernel today carries a boilerplate `test.py`
  (`kernels/projections/proj_ffn1_192_p4/test.py`, 3 lines) and an empty
  `__init__.py`. There are 96 and 102 of them.
- `ipu_app.bzl:10-41` declares one `py_test` per kernel. The test switches
  between "run the app" and "run pytest" through `bazel_entry.py:46-61`, based
  on `BUILD_WORKING_DIRECTORY`. Its data is every `.asm` in the family
  (`ipu_app.bzl:103-107`).
- Q70: tests stay in kernel folders. Q73: the enormous tier stays out of the
  default run, and runs nightly and under `--config=full`.

## Implementation

1. **Macro** `kernels/defs.bzl`:
   ```python
   def ipu_kernel(name = None, extra_tests = None, size = "large"):
       """Declare :<name> (bazel test: all cases x formats; bazel run: one case),
       :<name>_nightly (cases with tier = "nightly", tags = ["manual", "nightly"]),
       :assemble_<name> (default-case binary) and :<name>_files (asm, toml, py, npy)."""
   ```
   - `name` defaults to the package's last path component.
   - `srcs` hold only the shared `ipu_harness/kernel_test.py` entry. The data
     is the kernel directory's files, the family's `*.asm.j2` macros, and the
     shared `asm_lib`.
   - Dependencies are the family library (E06-18), `ipu_harness_lib` and
     `requirement("torch")`.
   - `extra_tests` defaults to `glob(["test_*.py"])`. These escape-hatch tests
     run in the same target and can use the fixtures `kernel`, `run_case` and
     `cases`.
2. **Entry** `ipu_harness/kernel_test.py`:
   - Under `bazel test`, it runs pytest over a generated parametrisation:
     `test_case[<case>-<format>]` for each expanded case (E06-08) and each
     format allowed for it, followed by the extra tests.
   - Under `bazel run`, it calls the CLI (E06-11). The
     `BUILD_WORKING_DIRECTORY` dispatch moves here from `bazel_entry.py`.
   - Assembly is shared across the parametrisation through one session.
3. **Sharding.** Set `shard_count` from the run count when it exceeds a
   threshold (for example 3 formats x 20 cases). This keeps per-target wall time
   inside the E01 CI budget.
4. **The `scale` kernel.** Add `kernels/elementwise/scale/` exactly as in the
   [epic](00-epic.md#worked-example-and-acceptance-target-the-scale-kernel):
   `scale.asm`, `kernel.toml`, `reference.py`, `data/x.npy`, `data/y.npy`, and
   a one-line `BUILD.bazel`. `data/*.npy` is produced once, and the snippet that
   produced it goes in the PR description. The kernel folder keeps exactly the
   files the epic lists.

## Tests

- [ ] `bazel test //kernels/elementwise/scale` passes 17 parametrised runs: `default`, `one_row` and `mine` x 3 formats, plus `sweep[...]` x 4 x 2 formats.
- [ ] `bazel run //kernels/elementwise/scale -- --case one_row --format e4` prints one PASS line with cycles and MAC utilisation.
- [ ] A planted wrong-output `.asm` (for example `ADD` instead of `MULT.VE`) fails with the case, the format, the output index and the tier in the message.
- [ ] A planted `test_extra.py` in the kernel folder runs as part of the same target.
- [ ] A case with `tier = "nightly"` is absent from `:scale` and present in `:scale_nightly`, which is tagged `manual`.

## Acceptance Criteria

- [ ] The `scale` kernel consists of `scale.asm`, `kernel.toml`, `reference.py`, `data/` and a one-line `BUILD.bazel`, and passes in FP32, INT8 and E4.
- [ ] No `__init__.py`, `test.py`, `app.py` or `cases.py` is needed in a kernel folder.
- [ ] `bazel test //...` is green, and the new target fits the CI time budget.
