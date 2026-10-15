# Element-format enum with derived sizes and accumulator type

Part of [Epic E03: ISA package](00-epic.md) · ID E03-13 · Size M · Depends on: E03-07, E03-12 · Behaviour change: none (declares data; the emulator adopts it in E04-13 and E04-17) · `*_WIDTH` localparams: unchanged

## Goal

Define one element-format enum in `ipu_isa` (Q47): `INT8` and `E1` to `E7` (hardware, selected at run time) and `FP32`, `INT32` (emulator-only). Every size that depends on the format is derived from it: element bytes, row bytes, the accumulator type and the per-register element layout. This replaces, as a single source, the emulator's `DType` plus the three wide-vector flags; E04 does the replacement, E06 validates `kernel.toml` formats against it.

## Background

- `DType(IntEnum)` (`ipu_emu/ipu_math.py:17-32`): `INT8 = 0`, `E1`..`E7` = number of exponent bits; 1 sign + x exponent + (7 - x) mantissa bits.
- The emulator-only modes are three `IpuState` flags: `wide_vector_debug`, `wide_vector_arithmetic: WideVectorArithmetic` (`FP32`, `INT32`, `ipu_state.py:42-50`) and `wide_vector_quantize_output` (`ipu_state.py:65-77`).
- Element width is computed per call from the flags: `xmem_element_width_bytes` and `xmem_row_size_bytes` (`ipu_emu/ipu.py:89-103`). The wide-mode register copies `r_wide_debug` (512 B) and `r_cyclic_wide_debug` (2048 B) are separate entries in `REGISTER_DEFINITIONS` (`registers.py:58`, `:81`).
- XMEM rows the program may address: 16384 rows of 128 B in 8-bit formats (`NARROW_MAX_ROW`, `ipu.py:117`), `XMEM_SIZE_BYTES // 512` rows of 512 B in wide mode (`XMEM_ADDRESSABLE_ROWS`, `ipu.py:106`).
- GLOSSARY.md already defines *element format* and *accumulator type* (INT32 for INT8, FP32 for every float format). Q50: wide means XMEM rows 4x wider, 128 elements per vector, only the width changes. Q44: kernels are not wide-only.
- E06-02 expects lower-case names `int8`, `e1`..`e7`, `fp32`, `int32`.

## Implementation

1. `ipu_isa/formats.py`, authored as data and loaded into frozen records:
   ```python
   class ElementFormat(StrEnum):
       INT8 = "int8"; E1 = "e1"; E2 = "e2"; E3 = "e3"; E4 = "e4"; E5 = "e5"; E6 = "e6"; E7 = "e7"
       FP32 = "fp32"; INT32 = "int32"

   class AccumulatorType(StrEnum): INT32 = "int32"; FP32 = "fp32"

   @dataclass(frozen=True, slots=True)
   class FormatInfo:
       format: ElementFormat
       tag: Tag                      # HARDWARE: INT8, E1..E7; EMULATOR: FP32, INT32
       is_float: bool
       element_bits: int             # 8 or 32
       exp_bits: int                 # 0 for integers, x for Ex, 8 for FP32
       man_bits: int                 # stored mantissa bits: 7 - x for Ex, 23 for FP32, 0 for integers
       accumulator: AccumulatorType  # INT8 -> INT32, INT32 -> INT32, every float -> FP32
       @property
       def element_bytes(self) -> int        # element_bits // 8
       @property
       def row_bytes(self) -> int            # geometry.LANES * element_bytes

   FORMATS: Mapping[ElementFormat, FormatInfo]
   def register_layout(register: str, fmt: ElementFormat) -> tuple[int, int]   # (elements, element_bytes)
   ```
   `register_layout` derives R0/R1 (128 elements), R_CYCLIC (512 elements) and R_MASK from `REGISTER_DEFINITIONS` element counts and the format, and returns the accumulator width for MULT_RES and R_ACC. It describes what E04-17 must allocate; it does not change `REGISTER_DEFINITIONS` in this issue.
2. Schema rules: every format has exactly one tag; `exp_bits + man_bits + 1 == element_bits` for float formats except FP32 (IEEE layout, checked separately); the accumulator rule above holds.
3. Docs: generate a format table (name, tag, element bits, exponent and mantissa bits, accumulator) into the generated ISA reference (`gen_docs`). Numeric semantics (bias, specials, saturation, the FP8 special-value convention left open by Q34) stay out: they belong to the codecs in E04-13 and to the open-questions page (E09-05).
4. A mapping test pins the correspondence `DType(n) <-> ElementFormat` (`INT8`, `E1`..`E7`) so E04-17 can replace `DType` mechanically.

## Tests

- [ ] `ipu-isa/test/test_formats.py`: sizes for every format (`INT8.row_bytes == 128`, `FP32.row_bytes == 512`); tags; accumulator rule; `register_layout("r", FP32) == (128, 4)`; `register_layout("r_cyclic", E4) == (512, 1)`.
- [ ] The `DType` correspondence test.
- [ ] E02 baselines unchanged except the generated reference page gaining the format table.

## Acceptance Criteria

- [ ] One enum defines every element format with its tag; FP32 and INT32 are EMULATOR.
- [ ] Element bytes, row bytes, accumulator type and register element layout are derived from it, never restated.

## Open question

- Addressable XMEM rows per format: today 16384 rows in 8-bit formats and 1,048,576 rows in FP32/INT32. Q50 says only the row width changes between formats. Which row count applies to all formats is undecided; this issue records today's two values as data (`addressable_rows(fmt)`) and does not change them.
