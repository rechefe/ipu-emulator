# Element-format lane codecs with exhaustive bit-identity tests

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-13 · Size M · Depends on: E03 (element-format enum in `ipu_isa`), E00 (FP8 subnormal-carry encode fix) · Behaviour change: none

## Goal

Add the numerics module that handlers are written against (Q48): one **codec** per element format that decodes a row of lanes into numpy arrays, and encodes results back with the format's rounding and saturation. The codecs are vectorised (FP8 decode is a 256-entry table lookup) and must be **bit-identical** to today's scalar arithmetic, proven by an exhaustive test (L5). Pure Python and numpy (Q63).

This issue only adds the module and its tests. E04-15 and E04-16 move the handlers onto it.

## Background

- E03 adds one element-format enum to `ipu_isa` (Q47): `INT8`, `E1`..`E7` (hardware, runtime-selected), `FP32`, `INT32` (EMULATOR-tagged). It derives element bytes, row bytes and accumulator type.
- Today the arithmetic is scalar Python in `ipu_math.py`: `ipu_mult` (ipu_math.py:184-194) decodes each FP8 byte with `_fp8_to_float32_scalar` and multiplies Python floats; `ipu_add`/`ipu_sub` (ipu_math.py:197-217) wrap INT8 accumulation to int32 and add FP values as Python floats. Results are stored with `struct.pack_into("<f" | "<i")`.
- The wide path is already vectorised in `Ipu` (`_wide_lanes_from`/`_wide_store_lanes`, ipu.py:707-740) with its own `_wide_imult32` (ipu.py:688-691).
- `DType` (ipu_math.py:17-31) is the current narrow format enum.
- Plan 02 measured about 58 µs per narrow cycle, mostly in these per-lane loops.

## Implementation

New package `ipu_emu/numerics/`:

```python
class LaneCodec(Protocol):
    fmt: ElementFormat               # from ipu_isa
    element_bytes: int               # 1 for INT8/E1..E7, 4 for FP32/INT32
    acc_dtype: np.dtype              # int32 or float32 (accumulator storage)
    def decode(self, raw: memoryview, count: int = LANES) -> np.ndarray     # compute domain
    def decode_scalar(self, word: int) -> np.generic                        # CR / LR-indexed scalar
    def mult(self, a: np.ndarray, b: np.ndarray | np.generic) -> np.ndarray # -> accumulator domain
    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray
    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray
    def max(self, a: np.ndarray, b: np.ndarray) -> np.ndarray               # NaN rule as today
    def load_acc(self, raw: memoryview, count: int = LANES) -> np.ndarray
    def store_acc(self, out: memoryview, lanes: np.ndarray) -> None         # struct-equivalent overflow
    def one(self) -> bytes                                                  # one element of value 1

def codec_for(fmt: ElementFormat) -> LaneCodec      # cached; the only format switch in ipu_emu
```

1. `IntCodec` (INT8, INT32): products and sums in int64, wrapped to int32 exactly like `ipu_add`.
2. `Fp8Codec` (E1..E7): decode via a cached float64 table of 256 entries built from today's `_fp8_to_float32_scalar`; products and sums in float64, stored as float32 (matches Python-float arithmetic followed by `pack_into("<f")`).
3. `Fp32Codec` (FP32): moves `_wide_lanes_from`/`_wide_store_lanes` here.
4. `store_acc` keeps today's overflow behaviour exactly (a finite result too large for float32 raises `OverflowError` with struct's message, lanes before it written, the offending lane zeroed). E04-23 changes that, as a behaviour change.
5. `ipu_math.ipu_mult/ipu_add/ipu_sub/dtype_one_byte` become thin adapters over the codecs, kept until E04-17 removes `DType`.

## Tests

- [ ] Exhaustive (L5): for INT8 and each of E1..E7, all 256 × 256 byte pairs: `codec.mult` equals `ipu_mult` bit for bit (`np.float32(...).tobytes()` or int32 bytes).
- [ ] Exhaustive: all 256 codes decode to the same float32 bits as `_fp8_to_float32_scalar`.
- [ ] Random: 10^6 accumulator pairs per format (including ±0, ±inf, NaN, subnormals, values near the float32 maximum): `add`, `sub`, `max` match the scalar functions bit for bit, and overflow raises at the same lane.
- [ ] INT accumulate wraps at ±2^31 identically.
- [ ] The exhaustive suite runs in under 10 s (vectorised over a 256×256 grid).

## Acceptance Criteria

- [ ] `codec_for` is the only place in `ipu_emu` that branches on a format.
- [ ] Exhaustive tests are green; no handler changed yet.
- [ ] `numerics` imports only numpy, the standard library and `ipu_isa`.

## Open question

- Should encode/decode of element formats move into `ipu_isa` (pure numpy) so that `ipu_harness` (E06, per-format packing) and the conformance tooling can use it without importing `ipu_emu`? The decisions place the format enum in `ipu_isa` (Q47) but say nothing about where the codec lives.
