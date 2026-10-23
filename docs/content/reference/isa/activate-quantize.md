# `ACTIVATE.QUANTIZE`, `ACTIVATE.EXACT` and `LD.LUT` {#activations-emulator}

The AaQ stage activates the accumulator, quantizes the result and stores it:
one instruction does all three, and no register stages the result
([ADR 0007](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0007-operational-aaq-and-nx8.md)).
The emulator follows the [AaQ architecture spec](../../specs/ARCH-stage-aaq-str.md);
where the [microarchitecture spec](../../specs/MAS-stage-aaq-str.md) disagrees, the
difference is an open question
([OQ-42](../../requirements/open-questions.md#emulator-and-isa)).

| Instruction | Slot | Tag | Modes | Stores |
|---|---|---|---|---|
| `ACTIVATE.QUANTIZE` | `AAQ` | HARDWARE | NX8 and FP32 | one row: 128 NX8 bytes and the row's metadata word, or 128 FP32 values |
| `LD.LUT` | `AAQ` | HARDWARE | both | nothing: loads one LUT segment from XMEM |
| `ACTIVATE.EXACT` | `AAQ_EXACT` | EMULATOR | FP32 only | one row of 128 FP32 values |

```asm
    LD.LUT 0 CR4 LR0 ;;                             # LUT segment 0 <- row CR4 + LR0
    ACTIVATE.QUANTIZE relu CR15.L CR14.L CR2 LR1 ;; # relu, quantize, store row CR2 + LR1
    ACTIVATE.EXACT silu CR15.L CR3 LR1 ;;           # FP32 mode only: exact silu, store
```

The instruction reference has each instruction's operands, operation and
register reads: [`ACTIVATE.QUANTIZE`](instructions.md#activatequantize-activate-quantize-and-store),
[`LD.LUT`](instructions.md#ldlut-load-lut-segment) and
[`ACTIVATE.EXACT`](instructions.md#activateexact-activate-exactly-and-store-fp32).

## `ACTIVATE.QUANTIZE`

**Syntax:** `ACTIVATE.QUANTIZE activation_fn ds qf table_id table_offset`.

- `activation_fn` is the 3-bit function of the architecture spec: `identity`,
  `relu`, `relu6`, `reciprocal`, `rsqrt`, `exp2` or `generic` (encodings 0–6;
  7 is reserved). The keyword order is `AAQ_ACTIVATION_FN_NAMES` in
  `src/ipu-isa/src/ipu_isa/activations.py`, the operand type `AaqActivationFn`.
- `ds` is the DS half (`CR15.L` by convention) naming the active elements;
  `qf` the QF half naming the NX8 output format
  ([Configuration registers](configuration-registers.md)). Both always name a
  half: a bare `CR15` is an assembler error.
- `table_id` (a CR) and `table_offset` (an LR) give the row,
  `CR[table_id] + LR[table_offset]`
  ([OQ-8](../../requirements/open-questions.md#conflicts-between-existing-documents)).

**Operation.** The instruction reads the forwarded `R_ACC` (this word's ACC
result, [timing model](../../explanation/timing-model.md#rule-3)) and leaves it
unchanged.

1. **Activate** each active element with the architecture spec's special
   cases (§4.3.1): `reciprocal` and `rsqrt` of a subnormal or zero input add
   the smallest epsilon, 2^-127, with the input's sign; of ±inf they are 0;
   `rsqrt` of a negative input is NaN; `exp2(-inf)` is 0.
2. **Quantize** (NX8 mode) the active elements to one row in the QF half's
   format (§4.3.2): the row's `scale` comes from the largest magnitude, each
   element is rounded to nearest with ties half up, and a dynamic format picks
   the exponent width of the row by the §4.3.2.3 rule
   (`ipu_isa.nx8.encode_row`).
3. **Store** the 128 bytes and the row's metadata word (`scale[7:0]`,
   `dynamic_exponent[8]`) at the row. In the FP32 mode the activated row is
   stored as 128 FP32 values (512 bytes) instead, with no metadata, and `qf`
   is ignored.

Inactive elements are written as 0 in both modes: the architecture spec writes
NaN, which NX8 cannot hold
([OQ-36](../../requirements/open-questions.md#emulator-and-isa)).

**What is an error.** ±inf saturates to the largest magnitude and is left out
of the scale; a saturation sets the `QUANT_SATURATION`
[arithmetic event](../../requirements/faults.md#arithmetic-events). An invalid DS or
QF half is a configuration error, raised in the word's check phase: the word
leaves the state as a fault does. A row outside XMEM is fault F-6, found in
the same phase. A NaN to quantize in NX8 (for example `rsqrt` of a negative
input), `exp2` of a positive input, and `generic` with no declared function
depend on `R_ACC`, which the same word's ACC forwards. They are emulator
errors raised when `ACTIVATE.QUANTIZE` executes, after the word's LR, LOAD,
MULT and ACC writes have landed.

**The LUT and `generic`.** The hardware evaluates `generic` through the LUT
that five `LD.LUT` loads fill. The 17-bit entry format and the interpolation
are not published yet
([OQ-38](../../requirements/open-questions.md#emulator-and-isa)), so every
function, `generic` included, is evaluated with exact math. `generic`
computes the function the test declares as loaded:

```python
from ipu_emu.activations import AaqOptions
from ipu_emu.ipu_state import IpuState

state = IpuState(aaq=AaqOptions(generic=lambda x: x / (1 + abs(x))))
```

## `LD.LUT`

**Syntax:** `LD.LUT lut_addr table_id table_offset`.

`LD.LUT` copies the first 128 bytes (1024 bits) of row
`CR[table_id] + LR[table_offset]` into LUT segment `lut_addr` (0–4; 5–7 are
reserved). Segments 0–3 hold table bits; segment 4 holds the last 256 table
bits and the 46-bit LUT metadata (`max_exp_num`, `neg_c`, `pos_c`,
`range_mode`, `neg_mode`, `pos_mode`). It reads no row metadata and scales
nothing, in both modes.

Both specs load the segment from `R_ACC`; the hardware reads XMEM, and so does
the emulator ([OQ-39](../../requirements/open-questions.md#emulator-and-isa)).
`LD.LUT` and an `LD.*` of the LOAD slot share the XMEM read port, so a word
that holds both is an assembler error.

`ipu_isa.lut.lut_image(fn, range_mode, range_exp)` builds the five rows that
load a function. The entry bits it writes are provisional until the entry
format is published (`ipu_isa/lut.py`):

```python
from ipu_isa.lut import lut_image

rows = lut_image(lambda x: x / (1 + abs(x)), range_mode=0, range_exp=3)
assert len(rows) == 5 and all(len(row) == 128 for row in rows)
```

## `ACTIVATE.EXACT` (emulator only)

**Syntax:** `ACTIVATE.EXACT activation_fn ds table_id table_offset`.

`ACTIVATE.EXACT` applies the exact Python activation (`apply_activation` in
`src/ipu-emu/src/ipu_emu/activations.py`) to the active elements of the
forwarded `R_ACC` and stores the row as 128 FP32 values at
`CR[table_id] + LR[table_offset]`, inactive lanes 0. It has the 13 functions
of the `ActivationFn` operand (`ACTIVATION_FN_NAMES`): `identity`, `relu`,
`relu6`, `sigmoid`, `tanh`, `gelu`, `softplus`, `elu`, `exp2`, `reciprocal`,
`rsqrt`, `silu` and `window`. Unlike `ACTIVATE.QUANTIZE`, `rsqrt` of an input
at or below 0 is 0, and `exp2` takes any input.

It is in its own emulator-only slot, `AAQ_EXACT`, which runs after ACC.
It is for FP32 kernels that need a function the hardware list lacks: the
emulator rejects it in an IPU in the NX8 mode, the harness rejects it for an
NX8 case, and `assemble --target hardware` rejects it. A word may not hold
both `ACTIVATE.EXACT` and `ACTIVATE.QUANTIZE`.

### α and the window bounds (`elu`, `window`)

The ISA has no operand for `elu`'s α or for the bounds of `window`. The
emulator reads them from the state's `activation` field, an
`ActivationParams` (`elu_alpha`, `window_a`, `window_b`). Unset, they take the
defaults in `ipu_emu/activations.py` (`DEFAULT_ELU_ALPHA`, `DEFAULT_WINDOW_A`,
`DEFAULT_WINDOW_B`).

`window` is the half-open indicator: `window(x) = 1` for `a <= x < b`, else 0.
The defaults are placeholders until calibration: `a = 0.0`, `b = 0.1`.

They are set on the state, like the run mode, not through a CR:

```python
from ipu_emu.activations import ActivationParams
from ipu_emu.ipu_state import IpuState

state = IpuState(activation=ActivationParams(elu_alpha=1.0, window_a=0.0, window_b=0.1))
# or, after construction:
state.set_activation_alphas(elu_alpha=1.0, window_a=0.0, window_b=0.1)
```

A state keeps its values from run to run; change them with
`set_activation_alphas`, or build a new `IpuState`.
