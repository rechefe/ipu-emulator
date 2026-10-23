# Element formats

An element format says how the IPU reads the elements of a vector: how many bits
each element has and what number those bits stand for. A run is in one of two
**modes**, chosen when it starts, and each mode has one element format:
**NX8**, the hardware's, and **FP32**, the emulator's. The program is the same
in both modes; the mode decides what its loads, multiplies and stores compute
([ADR 0007](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0007-operational-aaq-and-nx8.md)).

## The formats

--8<-- "docs/generated/element-format-tables.md:formats"

The table is generated from `ipu_isa.formats`.

- **Row bytes.** An XMEM row holds one element per lane, 128 elements, and
  every load and store moves exactly one row: 128 bytes and a 16-bit metadata
  word in NX8, 512 bytes in FP32. `LD.MASK` and `LD.LUT` read the first 128
  bytes of a row in both modes.
- **Accumulator type.** `MULT_RES` and `R_ACC` hold `FP32` in both modes: in
  NX8 the multiplier expands each element to FP32 first.
- **Register element widths.** `R0`, `R1` and `R_CYCLIC` hold one row's
  elements per register (or `R_CYCLIC` slot), and in NX8 the row's metadata
  word beside them; `R_MASK` is the same in both modes.

**NX8 (hardware).** The IPU's only element format: 8-bit elements whose
meaning depends on a per-row scale and on a **QF** (quantization format), a
16-bit half of a CR that each instruction names. It is a family of types, not
one format: the QF chooses the type of every vector an instruction reads or
writes.

**FP32 (emulator only).** IEEE float32 elements, with no metadata, no scaling
and no quantization; every QF operand is ignored. It runs a kernel written for
NX8 at full precision, so its result can be compared with a reference such as
PyTorch. The assembly and its XMEM row numbers are the same as in NX8; each
element takes four bytes instead of one.

FP32 and NX8 are the only element formats. "E4" means the NX8 type e4.

## How NX8 encodes values

An NX8 type comes from a QF half ([Configuration
registers](../reference/isa/configuration-registers.md#qf-half)): `num_bits`
W (1–8), `exp_format` fe (1–7), `sign` (0 or 1) and `dynamic`. An element has
`sign` sign bits, fe exponent bits and fm = W − fe − sign mantissa bits, in
the top W bits of its byte.

A row's 16-bit **metadata word** holds `scale[7:0]` (an `e8m0` exponent, bias
127) and `dynamic_exponent[8]`. The element's exponent field is a distance
*below* the scale, so one byte means different values in different rows:

- An exponent field `e` below all ones is normal:
  (1 + m / 2<sup>fm</sup>) × 2<sup>scale − 127 − e</sup>.
- The all-ones exponent field is the subnormal range:
  (m / 2<sup>fm</sup>) × 2<sup>scale − 127 − (2<sup>fe</sup> − 2)</sup>; with
  mantissa 0 it is zero.
- There is no NaN and no infinity
  ([OQ-37](../requirements/open-questions.md#emulator-and-isa)).

**Dynamic** types are signed and 8 bits wide; each row chooses fe = 1 or 2
(`dynamic_exponent` 0 or 1 in its metadata) by the architecture spec's rule
(`ARCH-stage-aaq-str.md` §4.3.2.3).

The values each fixed type holds at unit scale (scale 127), computed from the
codec (`ipu_isa.nx8.decode_element`):

--8<-- "docs/generated/element-format-tables.md:encodings"

**Quantizing** a row (`ACTIVATE.QUANTIZE`, `ipu_isa.nx8.encode_row`):

- **Scale.** The row's scale is the largest FP32 exponent among its active
  finite elements. FP32 inputs are always treated as normalized, so an
  all-zero row gets scale 0 and reads back as 2<sup>-127</sup>
  ([open questions](../requirements/open-questions.md#emulator-and-isa)).
- **Rounding.** Each element rounds to the nearest value; a tie rounds up.
- **Saturation.** ±inf saturates to the largest magnitude and is left out of
  the scale; a negative input to an unsigned type encodes as 0. Each sets the
  `QUANT_SATURATION` [arithmetic event](../requirements/faults.md#arithmetic-events).
- **NaN.** A NaN to quantize is an emulator error: NX8 cannot hold it.
- **Inactive elements** (outside the DS half's active elements) are written as
  0 ([OQ-36](../requirements/open-questions.md#emulator-and-isa)).

**Expanding** an element for a multiply is the inverse: the MULT decodes each
element with the QF half it names and the metadata of the row the element was
loaded from (`ARCH-stage-mult.md` §3.2.1). `LD.MULT` and `LD.MULT.CYCLIC`
copy that metadata into `R0`, `R1` or the `R_CYCLIC` slot.

XMEM keeps the metadata in a side store, one word per 128-byte unit. Metadata
that was never written reads 0, like unwritten data.

## CR scalars

A CR holds 32 bits. `MULT.VE`, `MULT.EE` and `MULT.RC.VE` (with a CR as `src`)
use a CR as a scalar element, a **MULT constant**: they multiply a vector by it.

--8<-- "docs/generated/element-format-tables.md:cr-scalars"

In NX8 the constant is one element with its own scale and type, independent of
the vector's QF: `value[7:0]`, `scale[15:8]`, `fe[19:16]` and `sign[23:20]`,
always 8 bits wide and never dynamic (`ipu_isa.cr_halves.constant`). In FP32
it is the float32 bit pattern. `CR0` and `CR1` are the exceptions: as MULT
constants they read as 0.0 and 1.0 in both modes, so a multiply by `CR1` is an
identity multiply whatever the mode.

This program multiplies a vector of ones by `CR3`, so every element of
`MULT_RES` is the scalar as the mode decodes it:

```python
--8<-- "docs/examples/test_element_formats.py:program"
```

The test that runs it checks element 0 of `MULT_RES` in each mode:

```python
--8<-- "docs/examples/test_element_formats.py:expected"
```

## Selecting a mode

The mode is emulator configuration, not a CR value. Choose it when you
construct the state, or reset the state into the other mode:

```python
from ipu_emu.ipu_state import IpuState
from ipu_isa.formats import ElementFormat

state = IpuState(fmt=ElementFormat.NX8)  # the default

# Or return an existing state to reset, in the other mode.
state.reset(ElementFormat.FP32)
```

`state.fmt` cannot be assigned. An unknown format raises `ConfigurationError`
when the state is built, never while a program runs.

A kernel lists its variants in its `kernel.toml`'s `[kernel] formats`: `fp32`
for the FP32 mode, and for NX8 a named type (`e1` to `e7`, `dynamic`) or
`s<sign>e<fe>w<bits>`. A case can list a subset in its own `formats`;
otherwise it runs in each of the kernel's variants. The harness binds the
case's QF where the kernel asks for it (`case_qf`).

## XMEM layout per mode

An XMEM operand in assembly (`offset + base`) is a row number in both modes.
A row is 128 elements, so its width depends on the mode: row r starts at byte
r × 128 in NX8 and r × 512 in FP32. A program reaches the same row in both
modes. NX8 addresses only the first 16384 rows of XMEM.

- **`LD.MULT`** reads one row (128 elements, and its metadata) into `R0` or
  `R1`.
- **`LD.MULT.CYCLIC`** reads one row into one of the four slots of `R_CYCLIC`.
  Its `index` must be a slot boundary: `0`, `128`, `256` or `384`.
- **`ACTIVATE.QUANTIZE`** stores one row: 128 NX8 bytes and the row's metadata,
  or 128 FP32 values. `ACTIVATE.EXACT` (FP32 only) stores one FP32 row.
- **Register indices count elements.** `rc_idx`, `ra_idx` and the cyclic
  `index` are element indices in both modes.
- **Element data** in XMEM is little-endian `float32` in FP32 and one byte per
  element in NX8.

The kernel harness packs each tensor for the case's variant
(`ipu_harness.packing`): in NX8 it quantizes each row and writes its metadata,
and reads both back for an output.

## How FP32 computes

- **Multiply and accumulate.** IEEE float arithmetic, in both modes once the
  NX8 elements are expanded.
- **Masks.** The mask-and-shift on `MULT_RES` is the same in both modes: each
  of the 128 mask bits gates one element.
- **Stores.** `ACTIVATE.QUANTIZE` stores the activated FP32 row without
  quantizing it; `ACTIVATE.EXACT` applies the exact activation instead of the
  hardware list ([`ACTIVATE.QUANTIZE`, `ACTIVATE.EXACT` and
  `LD.LUT`](../reference/isa/activate-quantize.md)).

## Arithmetic events

An accumulator overflow or a NaN result is an arithmetic event, not a fault:
see [Arithmetic events](../requirements/faults.md#arithmetic-events). What the
emulator does on an `FP32` overflow is OQ-18 in
[Open questions](../requirements/open-questions.md#emulator-and-isa).
