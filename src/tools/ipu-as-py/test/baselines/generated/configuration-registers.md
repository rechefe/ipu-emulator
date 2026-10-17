# Configuration registers

This page is **generated** from `ipu_isa.constants`. The host writes the CRs before a
kernel runs; a kernel only reads them.

## The CRs

| Register | Value | Notes |
|----------|-------|-------|
| `CR0` | `0` | Read-only constant. |
| `CR1` | `1` | Read-only constant. |
| `CR2`–`CR15` | set by the host | Base addresses, strides, loop bounds, scalar constants and dstructure words. By convention `CR15` holds the dstructure word, but any CR is a valid operand. |

Each LR and CR holds a 32-bit scalar. A negative constant is stored as its
32-bit two's complement (`value & LR_CR_SCALAR_VALUE_MASK`). The element format a
kernel computes in is not a CR: see [Element formats](../../explanation/element-formats.md).

## Dstructure word

A dstructure word in a CR describes the active elements of a vector. The instructions that
name a dstructure CR operand (`DstructureCrIdx`) read it: `ACTIVATE.QUANTIZE`, `AGG.MAX`, `AGG.MAX.FIRST`, `AGG.SUM`, `AGG.SUM.FIRST`, `MULT.EE`, `MULT.RC.VE`, `MULT.RC.VS`, `MULT.RC.VV`, `MULT.VE`.

| Field | Bits | Values | Default |
|-------|------|--------|---------|
| `valid_elements` | `[7:0]` | `0`–`128`: how many elements are active | `128` |
| `partition` | `[12:8]` | `0`, `2`, `4`, `8`, `16`: groups of elements the MULT mask shifts within (`0` is one group) | `0` |
| `pad_mode` | `[14:13]` | see below | `ZERO` |

The defaults are the emulator's initial `CR15`. Any other CR starts at 0, which
decodes to `valid_elements = 0`.

`AGG.*` and `ACTIVATE.QUANTIZE` read `valid_elements`; the MULT masking instructions read
`partition` and `pad_mode`.

### `pad_mode`

The value a MULT masking instruction writes into the elements its mask deactivates.

| Mode | Code | Fill value |
|------|------|------------|
| `ZERO` | `0` | `0` |
| `POS_INF` | `1` | `+inf` |
| `NEG_INF` | `2` | `-inf` |

`POS_INF` and `NEG_INF` exist only in float element formats; they are the identity
values of `min`- and `max`-style reductions (`ACC.MAX`).

```asm
AGG.SUM LR0 CR15;;
ACTIVATE.QUANTIZE relu CR3;;
MULT.RC.VV LR2 R0 0 LR4 CR15;;
```

## From the emulator

```python
from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK, PadMode

state.regfile.set_cr(2, 0x40000)                       # a base address
state.regfile.set_cr(9, (-128) & LR_CR_SCALAR_VALUE_MASK)  # a negative constant
state.set_cr_dstructure(valid_elements=64, partition=0, pad_mode=PadMode.NEG_INF, cr_idx=3)
```

`set_cr_dstructure` writes `CR15` when `cr_idx` is omitted.
