# Golden comparison: rounding to format and named tolerance tiers

Part of [Epic: Kernel framework](00-epic.md) · ID E06-10 · Size S · Depends on: E06-05, E06-08, E06-09 · Behaviour change: none

## Goal

Decide pass or fail one way for every kernel. The expected value is the
reference result rounded to the output's format, or the user's expected file.
The comparison uses a **named** tolerance tier chosen per format in
`[golden] tolerance`. A failure reports where and by how much.

## Background

- There are 17 distinct `rtol`/`atol` literals across 48 files (cluster plan
  07c), for example `kernels/projections/cases.py:17` `RTOL, ATOL = 1e-4, 1e-3`.
  None of them states a reason.
- Some checks are byte-exact (`kernel_registry/cases.py:97-106`
  `check_output_bytes`, used by `reshape/identity/cases.py` and
  `linear/fully_connected/cases.py`). Others are `assert_allclose` loops, as in
  `kernels/projections/cases.py:59-67`.
- Q80: "PyTorch reference rounded to format, compare with format tolerance
  tier". Q82: if the case gives an expected file, compare against the file.

## Implementation

Module `ipu_harness/tolerance.py`:

```python
class Tier(NamedTuple):
    name: str
    rtol: float
    atol: float          # in output units; for INT8 in LSBs
    ulps: int | None     # for 8-bit float formats: allowed code distance
    why: str

TIERS = {
    "EXACT":        Tier("EXACT", 0, 0, 0, "data movement or a single correctly rounded op"),
    "FP32_ORDER":   Tier("FP32_ORDER", 1e-5, 1e-6, None, "FP32 accumulation order differs from the reference"),
    "FP32_LONG":    Tier("FP32_LONG", 1e-4, 1e-5, None, "long reductions (K >= 256) or reciprocal/rsqrt chains"),
    "TRANSCENDENT": Tier("TRANSCENDENT", 1e-3, 1e-4, None, "libm vs future AaQ LUT headroom (Q25: provisional)"),
    "INT8_LSB":     Tier("INT8_LSB", 0, 1, None, "one LSB after round-to-nearest"),
    "FP8_ULP":      Tier("FP8_ULP", 0, 0, 1, "one code step in the 8-bit float format"),
}
```

Module `ipu_harness/compare.py`:

```python
@dataclass(frozen=True)
class Comparison:
    output: str
    tier: Tier
    passed: bool
    max_abs: float
    max_rel: float
    worst_index: tuple[int, ...]
    got: float
    want: float

def expected_for(case, plan, ref_out: Mapping[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """User expected file (rounded to the output format) if given, else the reference
    output rounded to the output's format (element format or accumulator format)."""

def compare(got, want, tier) -> Comparison: ...
```

Rules:
- The reference sees inputs **decoded from the format**. These are the values
  actually in XMEM after packing, not the float64 originals. Its float64 result
  is rounded once, to the output format, with the E03 codec (saturation
  included). The "rounding to format after" rule (Q80) means quantization error
  never counts against the kernel.
- User input files are rounded to the format when packed, like generated
  inputs. If rounding changed any element, the report says how many. A user
  `expected` file computed from the unrounded inputs then fails in a way that
  can be diagnosed, not mysteriously.
- The tier comes from `[golden] tolerance[format]`. An unknown tier name is a
  load-time error (E06-02). A kernel-local override is
  `tolerance = { fp32 = { tier = "FP32_ORDER", rtol = 3e-5, why = "measured on case X" } }`.
  The `why` is required, so no anonymous literal can appear.
- With no reference and no expected file, there is no comparison. That is
  allowed only for `bazel run` (a smoke run). `bazel test` fails with "no
  golden".

## Tests

- [ ] `scale` E4: a reference value between two E4 codes rounds to the nearest-even code, and the kernel passes at `EXACT` when the emulator rounds the same way.
- [ ] INT8 saturation: `x = 100`, `s = 2` gives the expected 127.
- [ ] The `Comparison` for a planted 1-element error names the output, the index, both values and the tier.
- [ ] An override without `why` is rejected at load.
- [ ] An expected `.npy` file is compared instead of the reference, even when `reference.py` exists.

## Acceptance Criteria

- [ ] All kernels compare through `compare()`. There are no `rtol=`/`atol=` literals in `kernels/` or `ipu_kernels` except tier overrides with a `why`.
- [ ] Tier values are provisional. The E07 pilot measures them in FP32, INT8 and E4 and records the numbers in the tier docstrings.
- [ ] `bazel test //...` is green.

## Open question

- The final tier names and values are not decided. The set above follows cluster
  plan 07c, and the FP8 tier depends on how AaQ rounds FP8 outputs (Q25/Q35).
  The pilot (E07-03) proposes the final values.
