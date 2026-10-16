# Per-format packing, unpacking and named `pack` transforms

Part of [Epic: Kernel framework](00-epic.md) · ID E06-05 · Size M · Depends on: E06-04, E03 (element formats and codecs) · Behaviour change: none

## Goal

Turn a logical tensor (a NumPy array of real values) into XMEM bytes for the
active element format, and turn output bytes back into a logical tensor. One
default layout covers most kernels. A kernel that needs a different layout
names a transform that reshapes **one declared tensor** (Q83). Element sizes
come from the E03 element format, never from a local constant.

## Background

- `LANES`, `ROW_BYTES` and `ELEM_BYTES` are redefined in most harnesses, for
  example `kernels/projections/app.py:83-85`. The review counted about 50 copies
  of each.
- Loaders are hand-written per kernel: `_load_data` x12, `_load_weights` x12.
  For example, `kernels/projections/app.py:186-211` does a transpose plus
  padding to rows, which is a channel-major layout and output-major weights.
- Input-size policy differs between kernels. `kernels/projections/app.py:189-195`
  requires the exact size. `kernels/matmul/matmul_144x288_x128/app.py:76`
  truncates oversize input (`raw[:expected]`). L15 makes exact size the rule.
- Q47/Q50: one element-format enum. Each vector has 128 elements; element bytes
  are 4 for FP32/INT32 and 1 for INT8/E1..E7. Q48: codecs encode and decode
  lanes with format rounding and saturation.
- `fully_connected` already packs per format by hand
  (`kernels/linear/fully_connected/app.py:91-115`, `_expand_int8_row`).

## Implementation

Module `ipu_harness/packing.py`:

```python
class PackTransform(Protocol):
    """Reshape ONE logical tensor into rows of 128 elements, and back."""
    def rows(self, shape: tuple[int, ...]) -> int: ...
    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray: ...     # -> (rows, 128) float64
    def from_rows(self, r: np.ndarray, plan: CasePlan, shape) -> np.ndarray: ...

DEFAULT = RowMajor()   # last dim -> lanes (zero-padded to 128; >128 splits into ceil(n/128) rows)

def pack(tensor: TensorDecl, a: np.ndarray, plan: CasePlan) -> bytes:
    """Exact shape check (L15) -> transform.to_rows -> E03 encode(format) -> bytes."""

def unpack(tensor: TensorDecl, raw: bytes, plan: CasePlan) -> np.ndarray:
    """bytes -> E03 decode(format or accumulator format) -> transform.from_rows."""
```

- **Transforms are named.** `pack = "channel_major"` resolves first to
  `pack.py:<name>` in the kernel folder (the escape hatch), then to the family's
  `ipu_kernels/<family>/pack.py`, then to the shared library
  `ipu_kernels/pack/`. The initial shared set is extracted from today's
  loaders: `channel_major` (projections/matmul input), `output_major_weights`
  (projections/matmul weights), and `chunk_interleaved` (convolutions). Each
  transform takes its parameters from the `CasePlan` (for example `n_tg`).
- **A transform cannot see other tensors, XMEM, or CRs.** Its signature
  enforces this. It is how Q83's "pack.py only transforms one declared tensor"
  is made structural rather than a convention.
- **Formats.** `pack` encodes with the E03 codec for `plan.format`. `unpack`
  decodes `store = "element"` outputs with the same format, and
  `store = "accumulator"` outputs with the format's accumulator format (FP32
  for FP32/E*, INT32 for INT8). A `unit = "bytes"` kernel (L16) is a CR
  concern (E06-06), not a packing one.
- **Exact sizes.** `pack` requires `a.shape == plan.shapes[name]`. User `.npy`
  data goes through the same check. There is no truncation path.

## Tests

- [ ] Round trip `unpack(pack(a)) == round_to_format(a)` for each format in {fp32, int8, e4, e5} and each shared transform.
- [ ] FP32 `channel_major` and `output_major_weights` produce byte-identical XMEM images to `ProjectionP4App._load_data`/`_load_weights` for all 12 projection layouts (golden for E07-02).
- [ ] INT8 packing of a `[2, 128]` tensor is 256 bytes, and FP32 packing is 1024 bytes.
- [ ] A wrong-shape array, including oversize, is rejected with the tensor name and both shapes.
- [ ] A kernel-local `pack.py` transform overrides a shared one of the same name.
- [ ] Packing takes a `CasePlan` and re-checks nothing (E06-03's single seam).

## Acceptance Criteria

- [ ] `LANES`, `ROW_BYTES` and `ELEM_BYTES` are not defined in `ipu_harness` or `ipu_kernels`. Element and row sizes come from `ipu_isa`.
- [ ] Packing has no `if format == ...` branches. Format behaviour comes from the codec.
- [ ] `bazel test //...` is green.

## Risks

- Narrow-format accumulator outputs span several rows per vector (see E06-04).
  `unpack` must read the stride that `place` allocated, not assume one row per
  vector.
