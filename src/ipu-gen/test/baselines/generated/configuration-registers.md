# Configuration registers

This page is **generated** from `ipu_isa.constants` and `ipu_isa.cr_halves`. The host
writes the CRs before a kernel runs; a kernel only reads them.

## The CRs

| Register | Value | Notes |
|----------|-------|-------|
| `CR0` | `0` | Read-only constant. As a MULT constant it reads 0.0. |
| `CR1` | `1` | Read-only constant. As a MULT constant it reads 1.0 in both modes (see [CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `CR2`–`CR15` | set by the host | Base addresses, strides, loop bounds, MULT constants, DS and QF halves. By convention `CR15.L` holds the default DS, but any CR half is a valid operand. |

Each LR and CR holds a 32-bit scalar. A negative constant is stored as its
32-bit two's complement (`value & LR_CR_SCALAR_VALUE_MASK`). A CR also holds two
16-bit halves, `CRn.L` (bits 15:0) and `CRn.H` (bits 31:16); an LR or COND operand that names a
half reads it sign-extended. The run mode (NX8 or FP32) is not a CR: see
[Element formats](../../explanation/element-formats.md).

## DS half

A DS half describes the active elements of a 128-element vector. The instructions with a `ds`
operand read it: `ACTIVATE.EXACT`, `ACTIVATE.QUANTIZE`, `AGG.MAX`, `AGG.MAX.FIRST`, `AGG.SUM`, `AGG.SUM.FIRST`, `MULT.EE`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.RC.VV`, `MULT.VE`.

| Field | Bits | Values |
|-------|------|--------|
| `valid_elements` | `[7:0]` | active elements per partition (`0`–`128`) |
| `num_partitions` | `[9:8]` | `0`: 1, `1`: 2, `2`: 4, `3`: 8 partitions (groups of elements) |
| `mask_type` | `[11:10]` | the fill of a masked-out element (below) |
| `valid_partitions` | `[15:12]` | how many partitions are active (`1`–`num_partitions`) |

The emulator's initial `CR15.L` is `0x1080`: every element active, one partition.
Any other CR starts at 0, which is no valid DS (`valid_partitions` 0): an instruction that
reads it is refused.

### `mask_type`

| Type | Code | Fill value |
|------|------|------------|
| `ZERO` | `0` | `0` |
| `POS_INF` | `1` | `+inf` |
| `NEG_INF` | `2` | `-inf` |
| `NAN` | `3` | refused (`CrContentError`) |

## QF half

A QF half names the NX8 type a row is read or quantized in. The instructions with a `qf`
operand read it in NX8 mode: `ACTIVATE.QUANTIZE`, `MULT.EE`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.RC.VV`, `MULT.VE`.

| Field | Bits | Values |
|-------|------|--------|
| `num_bits` | `[3:0]` | element width (`1`–`8`) |
| `exp_format` | `[7:4]` | exponent bits `fe` (`1`–`7`) |
| `sign` | `[11:8]` | `1`: signed |
| `dynamic` | `[15:12]` | `1`: dynamic exponent (signed 8-bit only) |

## MULT constant

A MULT constant operand reads a whole CR as one NX8 element with its own scale and type
(FP32 mode: its IEEE binary32 bits).

| Field | Bits | Values |
|-------|------|--------|
| `value` | `[7:0]` | the NX8 element code |
| `scale` | `[15:8]` | its scale (2<sup>scale − 127</sup>) |
| `exp_format` | `[19:16]` | its exponent bits `fe` |
| `sign` | `[23:20]` | `1`: signed |

```asm
AGG.SUM LR0 CR15.L;;
ACTIVATE.QUANTIZE relu CR15.L CR14.L CR3 LR0;;
MULT.RC.VV LR2 R0 0 LR4 CR15.L CR14.L;;
```

## From the emulator

```python
from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK
from ipu_isa.cr_halves import ds, qf
from ipu_isa.nx8 import Nx8Format

state.regfile.set_cr(2, 0x40000)                           # a base address
state.regfile.set_cr(9, (-128) & LR_CR_SCALAR_VALUE_MASK)  # a negative constant
state.set_cr_half(3, False, ds(valid_elements=64))         # CR3.L: a DS
state.set_cr_half(14, False, qf(Nx8Format(sign=1, exp_format=4, num_bits=8)))  # CR14.L: E4
```
