# `ACTIVATE.QUANTIZE` in the emulator {#activations-emulator}

The [AaQ stage spec](../../specs/stage-aaq-str.md) describes how the hardware
applies an activation: a function id (for example from `act_cr_idx` and a `CR`
read) and α-like parameters that are not VLIW immediates. They come from
implementation-defined configuration (constants, fuses, side-band registers).

The Python emulator adds an AAQ-slot instruction, `ACTIVATE.QUANTIZE`, that
applies one of thirteen activation shapes to the elements of `R_ACC` and writes
the results to `POST_AAQ_REG`, without modelling the `act_cr_idx` path. It is a
placeholder until the AaQ stage is specified
([OQ-20](../../requirements/open-questions.md#emulator-and-isa)).

```asm
ACTIVATE.QUANTIZE relu CR15;;
```

- **Syntax:** `ACTIVATE.QUANTIZE activation_fn cr_idx`. `activation_fn` is a
  keyword: `identity`, `relu`, `relu6`, `sigmoid`, `tanh`, `gelu`,
  `softplus`, `elu`, `exp2`, `reciprocal`, `rsqrt`, `silu` or `window`. The
  number of active elements is `cr_idx`'s `valid_elements`, the same
  dstructure field the `AGG.*` instructions read
  ([Configuration registers](configuration-registers.md)). `cr_idx` is
  required: any of `CR0`–`CR15`, with no default.
- **Where it is defined:** the keyword order is the ISA's `ActivationFn`
  operand, `ACTIVATION_FN_NAMES` in `src/ipu-isa/src/ipu_isa/activations.py`.
  The math is the emulator's `apply_activation` in
  `src/ipu-emu/src/ipu_emu/activations.py`.

## `R_ACC`, `POST_AAQ_REG` and `ST.AAQ`

- Accumulation stays in `R_ACC`: 512 bytes, 128 32-bit elements. `AGG.SUM`,
  `AGG.SUM.FIRST`, `AGG.MAX` and `AGG.MAX.FIRST` reduce `R_ACC` in place and
  write one `R_ACC` element, chosen by an LR.
- `ACTIVATE.QUANTIZE` reads `R_ACC`, applies the activation to each active
  element and writes the result to `POST_AAQ_REG`. `R_ACC` is unchanged.
- `POST_AAQ_REG` is, for now, a 512-byte staging register with the same
  element layout as `R_ACC`.
- Activation and quantization are one instruction; there is no
  activate-only or quantize-only opcode. In INT8 the activated elements are
  clamped to `[-128, 127]` and written as 128 bytes at the start of
  `POST_AAQ_REG`; the rest is zeroed. In FP32 and INT32 the activated 32-bit
  elements are written whole, and quantized to that 128-byte prefix only when
  `AaqOptions.quantize_output` is set
  (`IpuState(aaq=AaqOptions(quantize_output=True))`). The 8-bit float formats
  refuse it.
- `ST.AAQ` stores all 512 bytes of `POST_AAQ_REG` to XMEM, in whatever layout
  it holds when the store issues.

## α and the window bounds (`elu`, `window`)

The ISA has no operand for `elu`'s α or for the bounds of `window`. The
emulator reads them from the state's `activation` field, an
`ActivationParams` (`elu_alpha`, `window_a`, `window_b`). Unset, they take the
defaults in `ipu_emu/activations.py` (`DEFAULT_ELU_ALPHA`, `DEFAULT_WINDOW_A`,
`DEFAULT_WINDOW_B`).

`window` is the half-open indicator: `window(x) = 1` for `a <= x < b`, else 0.
The defaults are placeholders until calibration: `a = 0.0`, `b = 0.1`. On the
hardware, `window` belongs to the LUT-backed `activation` group of the
[AaQ spec](../../specs/stage-aaq-str.md), so `a` and `b` are part of the loaded
LUT; the emulator keeps them on the state instead.

They are set on the state, like the element format, not through a CR:

```python
from ipu_emu.activations import ActivationParams
from ipu_emu.ipu_state import IpuState

state = IpuState(activation=ActivationParams(elu_alpha=1.0, window_a=0.0, window_b=0.1))
# or, after construction:
state.set_activation_alphas(elu_alpha=1.0, window_a=0.0, window_b=0.1)
```

A state keeps its values from run to run; change them with
`set_activation_alphas`, or build a new `IpuState`.
