# `fully_connected` onto `ipu_harness` with its legacy data

Part of [Epic: Kernel migration](00-epic.md) · ID E07-21 · Size M · Depends on: E07-02 · Behaviour change: none

## Goal

Migrate `fully_connected`, the only kernel that already runs in more than one
format: INT8 narrow, INT8 data with wide INT32 arithmetic, E4 and E5. Keep its
current input data and goldens, so binaries and outputs are proven identical.
Replacing the goldens is E07-22.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/linear/fully_connected/`.

- **Shapes and addresses.** The shapes are fixed by the program: 10 samples x
  128 inputs -> 64 outputs (`app.py:39-47`). Byte base addresses `0x0000`,
  `0x20000` and `0x40000` are converted to rows for the CRs (`app.py:49-54`).
- **Output store.** The output is stored with `STR_ACC_REG`, which writes the
  full 512-byte accumulator. In narrow INT8 that is a 4-row stride
  (`app.py:55-61`).
- **Format handling.** The format comes from `parse_dtype` (`app.py:65-88`),
  which accepts `fp8_e4m3`/`fp8_e5m2` as aliases of E4/E5 (Q34 renames them).
  Execution is `mode="int32" if wide_mode else "native"` (`app.py:241-243`).
  INT8 rows are expanded to INT32 by hand in wide mode (`app.py:91-115`).
- **Cases.** The cases (`cases.py:15-44`) are `default` (INT8 data, wide
  INT32), `int8`, `fp8_e4m3` and `fp8_e5m2`. All are compared byte-exact
  against 9 `.bin` files in `test_data_format/`: an input, a weight and a
  golden per format. The wide and narrow INT8 paths share one golden.
- **Tests.** `test.py` skips on a missing fixture
  (`MissingInputFixture`, `test.py:12-18`). It re-runs every case for MAC and
  alias accounting (`test.py:21-28`). The constructor guards its own spec
  (`app.py:167`).

## Implementation

- **Folder kernel** `kernels/linear/fully_connected/`:
  - `formats = ["int8", "int32", "e4", "e5"]`, where `int32` is the EMULATOR
    wide-integer format from Q47 and replaces `wide_mode`;
  - inputs `x` of shape `[10, 128]` and `w` of shape `[64, 128]` (logical, not
    transposed); a family `pack.py` transform reproduces the transpose done by
    `_load_and_transpose_weights`;
  - output `y` of shape `[10, 64]` with `store = "accumulator"` (E06-04 gives
    the INT8 4-row stride);
  - `[abi]` from today's CR writes. Byte-unit CRs, if any remain, are marked
    `unit = "bytes"` (L16).
- **Legacy data:** a `gen.py:legacy_bin` generator decodes today's `.bin`
  inputs per format, and a `legacy` case uses it with the matching `.bin`
  golden as `expected`, through a small adapter that decodes the golden bytes to
  an array.
  - The format mapping is `default` -> `int32`, `int8` -> `int8`,
    `fp8_e4m3` -> `e4` and `fp8_e5m2` -> `e5`. This applies the Q34 rename,
    since format names come from E03.
- **Tests.** The MAC and alias accounting asserts (`mult_lane_ops`, and
  `alias_hits == {"A16_STR_ACC_REG": 10}`) move into a `test_accounting.py`
  that uses the profiled single run (E06-12). There is no second run. Delete
  the skip-on-missing-fixture behaviour and its test: a missing `.npy` or
  `.bin` is now a load error (cluster plan 06, step 1).
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` (the XMEM image per format, including the INT32 expansion) and `compare_legacy` are identical for all 4 old cases.
- [ ] The accounting asserts pass on the single profiled run.
- [ ] A missing data file fails the test. It is not skipped.

## Acceptance Criteria

- [ ] `fully_connected` runs from `kernel.toml` in 4 formats, with outputs identical to today.
- [ ] No `parse_dtype`, `wide_mode` or `MissingInputFixture` remains.
- [ ] `bazel test //...` is green.
