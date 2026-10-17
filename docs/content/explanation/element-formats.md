# Element formats

An element format says how the IPU reads the elements of a vector: how many bits
each element has and what number those bits stand for. A kernel runs in one
format, chosen when it starts. The program is the same in every format; the
format decides what its loads, multiplies and accumulations compute.

## The formats

The format sets the element width. Everything else follows from it:

- **Row bytes.** An XMEM row holds one element per lane, 128 elements. Its
  width is 128 × the element bytes.
- **Accumulator type.** `MULT_RES` and `R_ACC` hold `INT32` for the integer
  formats and `FP32` for the float formats.
- **Register element widths.** `R0`, `R1` and `R_CYCLIC` hold elements of the
  format; `R_MASK` is the same in every format.

--8<-- "docs/generated/element-format-tables.md:formats"

The table is generated from `ipu_isa.formats`.

**Hardware formats.** `INT8` and `E1` to `E7` are `HARDWARE` formats: the IPU
has them, and the host selects one at run time.

**Emulator formats.** `FP32` and `INT32` are `EMULATOR` formats: they exist only
in the emulator. They run a kernel at full precision, with no 8-bit rounding
on the multiply path, so its result can be compared with a full-precision
reference such as PyTorch. The assembly and its XMEM row numbers are the same as
in the 8-bit formats; each element takes four bytes instead of one.

## How the 8-bit formats encode values

**`INT8`** is a two's-complement byte, -128 to 127.

**`E1` to `E7`** are 8-bit floats. `Ex` has 1 sign bit, x exponent bits and
7 - x mantissa bits, and its exponent bias is 2<sup>x-1</sup> - 1.

- An exponent of all ones is NaN, whatever the mantissa. There are no
  infinities.
- An exponent of zero is a subnormal value: 0.mantissa × 2<sup>1-bias</sup>.
- `E1` has no normal values: its only non-zero exponent is the NaN exponent.
- `E7` has no mantissa bits: every value is a power of two.

--8<-- "docs/generated/element-format-tables.md:encodings"

The table is computed from the element codec, `ipu_isa.formats.decode_element`.

Encoding a value into an 8-bit format:

- **Rounding.** It rounds to the nearest value, and a tie goes to the even code.
- **Overflow.** A value beyond the largest finite value, and an infinity,
  saturate to the largest finite value of that sign. `INT8` saturates to -128 or 127.
- **NaN.** NaN encodes as the canonical NaN: exponent all ones, mantissa 1
  (`0x7F` in `E7`, which has no mantissa bits).
  `INT8` has no NaN, so encoding NaN to `INT8` is an error.
- **Negative zero.** In `E1` to `E7`, `-0.0` keeps its sign bit (`0x80`).

!!! warning "Open question"
    The special values are not settled. The emulator reads every all-ones
    exponent as NaN, where IEEE-style decoders read mantissa 0 as an infinity,
    and its largest `E4` value is 240, not the 448 of OCP E4M3. See OQ-19 in
    [Open questions](../requirements/open-questions.md#emulator-and-isa).

## CR scalars

A CR holds 32 bits. `MULT.VE`, `MULT.EE` and `MULT.RC.VE` (with a CR as `src`)
use a CR as a scalar element: they multiply a vector by it. The scalar is the
CR's low element-width bits, decoded as an element of the active format:

--8<-- "docs/generated/element-format-tables.md:cr-scalars"

So the same CR value stands for a different number in each format. A host that
programs a scalar writes it in the kernel's format: the float32 bit pattern of
the value in `FP32`, its 8-bit code in `E4`.

`CR1` is the exception. It is the hardware constant 1, and as a scalar it reads
as 1 in every format (1.0 in a float format), so a multiply by `CR1` is an
identity multiply whatever the format. This rule holds from E04-28 on; before
it, every format read the CR's low byte.

This program multiplies a vector of ones by `CR3`, so every element of
`MULT_RES` is the scalar as the format decodes it:

```python
--8<-- "docs/examples/test_element_formats.py:program"
```

The test that runs it checks element 0 of `MULT_RES` in each format:

```python
--8<-- "docs/examples/test_element_formats.py:expected"
```

## Selecting a format

The format is emulator configuration, not a CR value. Choose it when you
construct the state, or reset the state into another format:

```python
from ipu_emu.ipu_state import IpuState
from ipu_isa.formats import ElementFormat

state = IpuState(fmt=ElementFormat.INT8)

# Or return an existing state to reset, in another format.
state.reset(ElementFormat.E4)
```

`state.fmt` cannot be assigned. An unknown format raises `ConfigurationError`
when the state is built, never while a program runs.

A kernel declares its formats:

- **A kernel with a `kernel.toml`** lists them in `[kernel] formats`. A case
  can list a subset in its own `formats`; otherwise it runs in each of the
  kernel's formats.
- **An `IpuApp` kernel** declares one format in its `SPEC` with
  `ExecutionConfig(fmt=...)`, and `app.make_state()` builds the state in it.

## XMEM layout per format

An XMEM operand in assembly (`offset + base`) is a row number in every format.
A row is 128 elements, so its width depends on the format: row r starts at byte
r × 128 in the 8-bit formats and r × 512 in `FP32` and `INT32`. A program reaches
the same row in every format. The 8-bit formats address only the first 16384
rows of XMEM.

- **`LD.MULT`** reads one row (128 elements) into `R0` or `R1`.
- **`LD.MULT.CYCLIC`** reads one row into one of the four slots of `R_CYCLIC`.
  Its `index` must be a slot boundary: `0`, `128`, `256` or `384`.
- **Register indices count elements.** `rc_idx`, `ra_idx` and the cyclic
  `index` are element indices in every format, so a program indexes its
  registers the same way whatever the format.
- **Element data** in XMEM is little-endian `float32` or `int32` in `FP32` and
  `INT32`, and one byte per element in the 8-bit formats.

The kernel harness packs each tensor for the case's format
(`ipu_harness.packing`). An accumulator output (`ST.ACC`) is stored in the
format's accumulator type, four bytes per element: in an 8-bit format one of
its vectors spans four rows.

## How FP32 and INT32 compute

- **Multiply and accumulate.** `FP32` uses IEEE float arithmetic. `INT32`
  multiplies and adds with 32-bit two's-complement wrap, and `AGG.*` reduce
  as 32-bit signed integers.
- **Masks.** The mask-and-shift on `MULT_RES` is the same in every format:
  each of the 128 mask bits gates one element.
- **`ACTIVATE.QUANTIZE`** writes the activated elements to `POST_AAQ_REG`
  without quantizing them, unless the state was built with
  `AaqOptions(quantize_output=True)` (from `ipu_emu.activations`). `R_ACC` keeps the full results; `ST.ACC`
  stores them. The AaQ stage is a placeholder: see OQ-20, OQ-22 and OQ-34 in
  [Open questions](../requirements/open-questions.md#emulator-and-isa).

## Arithmetic events

An accumulator overflow or a NaN result is an arithmetic event, not a fault:
see [Arithmetic events](../requirements/faults.md#arithmetic-events). What the
emulator does on an `FP32` overflow is OQ-18 in
[Open questions](../requirements/open-questions.md#emulator-and-isa).
