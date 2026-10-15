# PyTorch reference loading and the shared reference module

Part of [Epic: Kernel framework](00-epic.md) · ID E06-09 · Size M · Depends on: E06-02, E01 (lockfile workflow) · Behaviour change: none

## Goal

Let a kernel state "correct" as a few lines of PyTorch. That is either its own
`reference.py`, or a shared reference selected by `op` from
`ipu_kernels.reference`. PyTorch is a test-only dependency (Q74b). The shared
module holds exactly one reference per operation. LayerNorm is the first one
that matters: it has no eps and matches what the kernels compute (Q57).

## Background

- References are copied across the repo: LayerNorm 12 times (with two
  semantics, eps and no eps), SiLU 9 times and softmax 9 times. The tolerance
  literals take 17 different values across 48 files. See cluster plan 07c.
  Examples: `kernels/projections/cases.py:21-32` (`silu`, `reference`),
  `kernels/normalize/cases.py` (LayerNorm without eps), and
  `test/mobilevit/test_full_layer_l4.py` (with eps=1e-5).
- `attention/cases.py` `agg_fold`/`acc_fold` mirror the emulator's rounding at
  rtol=atol=0. They are datapath models, not math references. A bug shared by
  the model and the emulator goes unseen.
- Q74/Q74b: references use PyTorch, as a test-only dependency, and users keep
  the option of testing against their own data. Q57: one LayerNorm reference
  matching the kernel (no eps), to be revisited with AaQ.
- Python dependencies come from each package's `pyproject.toml` through
  `generate_requirements` and `pip.parse` (`MODULE.bazel:19-71`).

## Implementation

1. **Dependency.**
   - Add `torch` (the CPU wheel) to the `dev`/test optional dependencies of
     `ipu-kernels`, never to a runtime dependency list. Regenerate the lock
     through the E01 workflow.
   - Only kernel test targets (E06-13) and `ipu_kernels.reference` tests depend
     on `requirement("torch")`. The `ipu_harness_lib` library does not.
2. **Loader** (`ipu_harness/reference.py`):
   ```python
   ReferenceFn = Callable[..., Mapping[str, "torch.Tensor"]]

   def load_reference(manifest: KernelManifest) -> ReferenceFn | None:
       """[golden] reference = "reference.py:fn" -> that function (loaded by file path, no
       __init__.py needed); [golden] op = "layernorm" -> ipu_kernels.reference.ops.layernorm;
       neither, and reference.py exists -> reference.py:reference; else None."""

   def evaluate(fn: ReferenceFn, inputs: Mapping[str, np.ndarray], params) -> dict[str, np.ndarray]:
       """Call fn with float64 torch tensors (inputs already decoded from the format)
       and keyword params; return float64 NumPy arrays keyed by output name."""
   ```
   `torch` is imported inside `evaluate` only, so `bazel run` with `--input` and
   `expected` files works without loading PyTorch.
3. **Shared module** (`ipu_kernels/reference/ops.py`): `matmul`, `linear`
   (with optional activation), `silu`, `softmax`, `layernorm` (`eps = 0.0`, with
   rsqrt(0) treated the way the kernel does today; the docstring cites Q57 and
   the AaQ open question), `conv2d` (bias and activation optional),
   `maxpool2d` and `residual_add`. Each is a few lines of `torch.nn.functional`.
   They are filled in as E07 migrates the families. This issue adds `matmul`,
   `silu`, `softmax` and `layernorm`.
4. **Datapath mirrors.** These move to `ipu_kernels/reference/datapath.py`,
   labelled as mirrors. A golden that uses a mirror at `EXACT` must also pass the
   matching `ops.*` reference at an FP32 tier. That rule lands with attention in
   E07.

## Tests

- [ ] `load_reference` resolves all four forms: explicit `reference.py:fn`, `op`, implicit `reference.py:reference`, and none.
- [ ] `ops.layernorm` matches a hand-computed 3x2 example. On a constant row it returns what the kernel returns (no eps).
- [ ] `ops.softmax` rows sum to 1, and `ops.silu` matches the closed form at 5 points.
- [ ] The import-graph test (E06-01) shows `ipu_kernels.reference` does not import `ipu_emu`, which after Q31 also holds the emulator's activation functions.
- [ ] `ipu_harness_lib` builds and its unit tests pass without torch installed.

## Acceptance Criteria

- [ ] PyTorch is a test-only dependency, pinned in the lockfile.
- [ ] One shared reference per op. New kernels use `op = ...` or a local `reference.py`, never a copy.
- [ ] `bazel test //...` is green.

## Risks

- **Wheel size and CI time.** The torch CPU wheel is about 200 MB. Measure the
  cold-cache CI time in the PR against the E01 5-10 min budget. If it does not
  fit, use the CPU-only index URL and cache the wheel in CI.
- **Python version.** Check the wheel for the hermetic Python 3.10
  (`MODULE.bazel:12`).
