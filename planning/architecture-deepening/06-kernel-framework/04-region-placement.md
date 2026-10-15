# Automatic XMEM region placement with optional row pins

Part of [Epic: Kernel framework](00-epic.md) · ID E06-04 · Size S · Depends on: E06-03 · Behaviour change: none

## Goal

Place every declared tensor in XMEM automatically, in rows, for the active
format. An optional `row = N` in `kernel.toml` pins a region (Q81). No kernel
computes base addresses by hand.

## Background

- Every harness computes its own bases. For example,
  `kernels/projections/app.py:141-156` (`data_base_row`, `weights_base_row`,
  `output_base_row`) packs regions back to back. `memory.py:33-40` checks
  capacity for the whole-image kernels only.
- Q50: wide mode is XMEM 4x wider, with 128 elements per vector, and only the
  width changes. So a region's size in rows is the same in every format, while
  its size in bytes is `rows x 128 x element_bytes(format)`. Base rows written
  to CRs are therefore format-independent for element tensors. Capacity in
  rows depends on the format, because XMEM bytes are fixed.
- One exception: an accumulator store writes the full accumulator width.
  `kernels/linear/fully_connected/app.py:55-61` explains why INT8 output
  vectors need `OUTPUT_STRIDE_ROWS = 4`: `STR_ACC_REG` writes 512 B, which is 4
  narrow rows. An output declared `store = "accumulator"` therefore takes
  `accumulator_bytes(format) x 128 / row_bytes(format)` rows per vector.

## Implementation

Module `ipu_harness/placement.py`:

```python
@dataclass(frozen=True)
class Region:
    name: str
    role: Literal["input", "output", "scratch"]
    base_row: int
    rows: int                      # rows the packed tensor occupies (E06-05 says how many)

@dataclass(frozen=True)
class Placement:
    regions: Mapping[str, Region]
    rows_used: int
    def region(self, name: str) -> Region: ...

def place(plan: CasePlan, row_counts: Mapping[str, int]) -> Placement:
    """Pinned regions first, then the rest in declaration order
    (inputs, outputs, scratch), each at the lowest free row."""
```

Rules:
- Placement is deterministic: the same plan gives the same bases. Declaration
  order matters so that migrated kernels can reproduce their current layout
  exactly. E07 proves this against the E02 digests.
- Pins that overlap each other, or that run past capacity, are configuration
  errors naming both regions.
- Capacity is `XMEM bytes / (128 x element_bytes(format))` rows. It comes from
  `ipu_isa` geometry (E03), not a local constant.
- Scratch regions (`[scratch.<name>] shape = [...]`) are placed and zeroed.
  They get an `[abi]` base like any tensor.

## Tests

- [ ] `scale` with rows = 4: `x` at row 0 (4 rows), `y` at row 4.
- [ ] A pinned region keeps its row, and the rest fill around it.
- [ ] Overlapping pins and a capacity overflow each give a located error.
- [ ] The same element tensor needs the same row count in FP32 and E4, while the capacity in rows differs (4x).
- [ ] An `accumulator` output of 2 vectors takes 2 rows in FP32 and 8 rows in INT8.
- [ ] Placement for `projections` layouts matches `ProjectionLayout` bases (`kernels/projections/app.py:141-156`) for all 12 kernels, using declaration order and no pins (guards E07-02).

## Acceptance Criteria

- [ ] No harness or kernel computes a base address. Bases reach the program only through `[abi]` (E06-06).
- [ ] `bazel test //...` is green.
