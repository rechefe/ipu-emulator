# Chain kernels in memory (one kernel's output region is the next one's input)

Part of [Epic: Kernel framework](00-epic.md) · ID E06-15 · Size S · Depends on: E06-11 · Behaviour change: none

## Goal

Let a test run kernel B on the machine state kernel A left behind, with B's
input region bound to A's output region, without a file round-trip and without
monkeypatching harness modules. The MobileViT layer chains (ported in E07) are
the first users.

## Background

- `test/mobilevit/fixture_kernel_support.py:121` (`skip_load_data`) disables a
  harness module's `_load_data` by monkeypatching, so a producer's raw output
  can stay in XMEM. Lines 67-71 and 105-114 read `module.OUTPUT_BASE` and
  `module.ROW_BYTES` globals. The harness module's globals *are* the interface
  the chain goes through.
- `test_full_layer_l{4,5}_packed.py` copy each consumer's CR map, at about 43
  `set_cr` calls per file.
- Q83 deletes the whole-image path. Raw-image debugging moves to the debugger
  XMEM view, so chaining needs its own explicit API.

## Implementation

`ipu_harness/chain.py`:

```python
@dataclass(frozen=True)
class Resident:
    """A tensor already in XMEM: where it is and how it is laid out."""
    base_row: int
    rows: int
    pack: str | None       # transform name it was written with
    format: ElementFormat

class Chain:
    def __init__(self, fmt: ElementFormat, *, options: RunOptions = RunOptions()): ...
    def run(self, kernel: str | KernelManifest, params: Mapping[str, object],
            inputs: Mapping[str, np.ndarray | Resident]) -> ChainStep:
        """Plan, place (Resident inputs are pinned at their rows), pack only the
        np.ndarray inputs, write CRs, assemble, run on the chain's ONE model."""

@dataclass(frozen=True)
class ChainStep:
    result: RunResult
    resident: Mapping[str, Resident]   # this step's outputs, ready to feed the next step
```

- A `Resident` input becomes a row pin in placement (E06-04). Overlap with
  other regions is checked as usual.
- Layout compatibility is checked: a `Resident`'s `pack` name and `format` must
  equal the consumer's declaration, or the call fails before running. This
  replaces "poison the region and hope" (`fixture_kernel_support.py:67,105`).
- Comparison per step is optional (`compare=False` for intermediate steps). The
  chain test compares the final output against a chained reference.

## Tests

- [ ] Chain `scale` into `scale`: step 2's `x` is step 1's `y` as `Resident`, and the final output equals `x * s1 * s2` rounded per step.
- [ ] A format or `pack` mismatch between producer and consumer fails before running.
- [ ] A `Resident` overlapping the consumer's other regions is a placement error.
- [ ] No file I/O happens between steps (checked with a patched `open`).

## Acceptance Criteria

- [ ] Chaining needs no module globals, no monkeypatching and no CR numbers.
- [ ] `bazel test //...` is green.
