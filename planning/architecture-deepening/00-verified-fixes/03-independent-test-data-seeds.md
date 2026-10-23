# Test data: independent draws for convolution cases, stable seeds for mobilevit fillers

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-03 · Size S · Depends on: none · Behaviour change: yes (test inputs only: convolution weights and biases, and mobilevit filler streams, get new values; the emulator is unchanged)

## Goal

Every random test input is reproducible across processes, and two tensors of one case never share the same random values. A kernel that reads weights where it should read inputs, or bias where it should read weights, must fail its test.

## Background

Verified at 694a8a8 (`src/tools/ipu-apps/`):

- `kernel_registry/case_support.py:7-8`:
  ```python
  def random_values(shape):
      return np.random.default_rng(7).standard_normal(shape).astype('<f4')
  ```
  Each call reseeds, so every call returns the same leading values.
- `kernels/convolutions/cases.py:29-31` draws the input, the weights and the bias of one case with three calls:
  ```python
  x = random_values(params['shape'])
  weights = random_values((out_channels, channels, k, k)) / 4
  bias = random_values((out_channels,))
  ```
  So `bias` equals the first `out_channels` values of `x` (flattened), and `weights * 4` starts with the same values as `x`. A kernel that loads the wrong region (for example `x` where it should load the bias) can still produce the expected output for small cases such as `single_pixel`. This affects `conv1x1`, `conv3x3_relu` and `conv3x3_relu_cin1` (`make_cases`, used by the memory-layout kernels).
- The other callers of `random_values` draw one tensor per case: `pooling/cases.py:19, 44`, `detect/channel_peak/cases.py:13`, `detect/score_threshold/cases.py:14`, `normalize/l2_normalize_channels/cases.py:11`. They are not affected and their inputs must not change.
- `test/mobilevit/test_full_layer_l4.py:235, 375` and `test/mobilevit/test_full_layer_l5.py:198, 339` seed filler streams with `np.random.RandomState(hash((tag, "filler")) & 0xFFFFFFFF)`. `hash()` of a tuple of strings is salted per process (`PYTHONHASHSEED`), so each run uses different filler data. A failure cannot be reproduced from its log. These `test_full_*` targets are `manual` and `enormous` (`ipu_app.bzl:179-183`).

## Implementation

1. `case_support.py`: give `random_values` an optional generator and keep today's default:
   ```python
   def random_values(shape, rng: np.random.Generator | None = None):
       """Standard-normal float32 values. Pass one ``rng`` per case to draw
       several independent tensors; without it, each call restarts seed 7."""
       rng = np.random.default_rng(7) if rng is None else rng
       return rng.standard_normal(shape).astype('<f4')
   ```
   Single-tensor callers do not change, so their inputs stay bit-identical.
2. `convolutions/cases.py` `make_cases.prepare`: create `rng = np.random.default_rng(7)` once per case and pass it to all three draws. `x` keeps its current values (first draw of seed 7). `weights` and `bias` change.
3. Mobilevit: replace the four `hash(...)` seeds with a stable digest:
   ```python
   def _stable_seed(*parts: str) -> int:
       return zlib.crc32("/".join(parts).encode())
   rng = np.random.RandomState(_stable_seed(tag, "filler"))
   ```
   Put `_stable_seed` in the shared mobilevit helper module the two files already import from, or define it once in each file if they share none.
4. Grep for any other `hash(` used as a seed under `src/tools` and fix it the same way. At HEAD these four are the only ones.

## Tests

- [ ] Unit test in `test_kernel_registry.py` (or a new `test_case_support.py`): with one `rng`, the three convolution draws share no leading values; without an `rng`, `random_values` returns the same values as before this change (compare against a hard-coded prefix of seed-7 values).
- [ ] Unit test: `_stable_seed("a", "filler")` is the same in two subprocesses started with different `PYTHONHASHSEED` values.
- [ ] `bazel test //src/tools/ipu-apps:convolutions` passes with the new weights and biases. If a case fails, that is a real kernel or reference bug: fix it in a separate PR before this one merges, and link it.
- [ ] Pooling, detect and l2-normalize targets pass unchanged.
- [ ] The mobilevit `test_full_*` targets are manual and take hours. Run them once by name and record the result in the PR, or state that they were not run.

## Acceptance Criteria

- [ ] No two tensors of one convolution case come from a restarted generator.
- [ ] No test seeds a generator from `hash()`.
- [ ] Inputs of single-tensor callers are bit-identical to HEAD.
- [ ] The PR is labelled as a behaviour change (test inputs) and names the affected targets.

## Risks

- E06 replaces `cases.py` with `kernel.toml` cases and generated data. This fix still matters until then, and the "one generator per case" rule should carry over (E06 owns that).
