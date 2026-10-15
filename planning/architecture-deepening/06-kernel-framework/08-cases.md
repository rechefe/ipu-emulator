# Cases: defaults, sweeps, generator escape hatch, user `.npy` data

Part of [Epic: Kernel framework](00-epic.md) · ID E06-08 · Size M · Depends on: E06-03 · Behaviour change: none

## Goal

Declare a kernel's cases as data in `[cases]`. A case is parameter values, plus
an optional format limit, user inputs/expected files, or a generator. The
framework produces the input tensors from a seeded, per-tensor generator unless
the case or the command line supplies them (Q82).

## Background

- 99 `cases.py` files write input files into a workspace, build a
  `PreparedCase` and a `check` closure. See `kernels/projections/cases.py:44-93`
  and `kernel_registry/cases.py:16-42`.
- Sweeps are a separate mechanism (`kernel_registry/testing.py:21-33`, the
  `case_tests(..., sweep=[...])` argument).
- RNG problems:
  - `case_support.random_values` reseeds `default_rng(7)` on every call
    (`kernel_registry/case_support.py:8`). As a result, convolution weights equal a
    prefix of the inputs (`kernels/convolutions/cases.py:29-31`).
  - `RandomState` is used 72 times, and `default_rng` 6 times.
  - Mobilevit filler data is seeded with `hash((tag, "filler"))`, which depends
    on PYTHONHASHSEED (`test/mobilevit/test_full_layer_l4.py:235,375`).
- Q82: a case may name `.npy` inputs and expected outputs, relative to the
  kernel folder, and `--input x=path.npy` works on the command line. The output
  is compared to the file if one is given, otherwise to the reference.

## Implementation

Module `ipu_harness/cases.py`:

```python
@dataclass(frozen=True)
class Case:
    name: str                       # "default", "sweep[rows=7]"
    values: Mapping[str, object]    # scalars only (sweep expanded)
    formats: tuple[ElementFormat, ...]
    inputs: Mapping[str, Path]
    expected: Mapping[str, Path]
    generator: str | None
    tier: str

def expand(manifest: KernelManifest) -> tuple[Case, ...]:
    """List-valued keys are sweep axes: one case per point of their cartesian product,
    named "<case>[k=v,...]". Values not given fall back to [params] defaults."""

def inputs_for(case: Case, plan: CasePlan, *, overrides: Mapping[str, Path] = {}) -> dict[str, np.ndarray]:
    """Priority: --input override > case `inputs` file > case `generator` > default generator."""
```

- **Default generator.** For each input tensor it uses
  `rng(kernel, case, tensor)` = `default_rng(SeedSequence(crc32("kernel/case/tensor")))`.
  The seed is stable across processes, and every tensor gets an independent
  stream. Values are uniform in `[-1, 1)` for float formats and uniform over
  the full INT8 range for INT8. An input can set `range = [lo, hi]`, or a
  per-format table `range = { int8 = [-3, 3], e4 = [-1, 1] }`. This keeps the
  outputs out of saturation, which would otherwise make INT8 tests pass
  trivially. Values are
  generated as float64 and then rounded to the format, so the reference sees
  exactly what XMEM holds.
- **Generator escape hatch.** `generator = "gen:make"` names `gen.py:make` in the
  kernel folder. Its signature is `make(plan, rng) -> dict[str, np.ndarray]`,
  for inputs that need structure, such as sorted, sparse or masked data. It
  returns logical tensors and never bytes.
- **User data.** `inputs = { x = "data/x.npy" }` loads with `np.load(allow_pickle=False)`.
  The shape must match exactly (L15). `expected = { y = "data/y.npy" }` is
  handed to the comparer (E06-10).
- **Tier.** `tier = "nightly"` keeps a heavy case out of the default `bazel test`
  run (Q73). E06-13 places it in a separate target.

## Tests

- [ ] `scale` expands to 7 cases with the names `default`, `one_row`, `sweep[rows=1]`, ..., `mine`.
- [ ] A two-axis sweep produces the cartesian product in a stable order.
- [ ] `rng("k", "c", "x")` is equal in two subprocesses with different PYTHONHASHSEED, and differs from `rng("k", "c", "w")`.
- [ ] Priority order: `--input` beats a case file, and a case file beats the generator.
- [ ] A `.npy` with the wrong shape or dtype `object` is rejected.
- [ ] A generator that returns bytes, or a tensor that is not declared, is rejected.

## Acceptance Criteria

- [ ] A kernel needs no Python to declare cases, sweeps or user data.
- [ ] No per-call reseeding and no `hash(` seeding in `ipu_harness`.
- [ ] `bazel test //...` is green.
