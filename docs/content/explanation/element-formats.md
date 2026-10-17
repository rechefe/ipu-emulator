# Element formats

!!! note "Placeholder"
    E09-19 writes this page: the element formats (INT8, E1–E7, and the
    emulator-only FP32 and INT32), how each sets the element width, the XMEM row
    width and the accumulator type, and where each is checked. The
    [instruction reference](../reference/isa/instructions.md#element-formats)
    lists them. Until then this page holds how a kernel selects one in the
    emulator, moved here from the former IPU configuration page.

## Selecting the element format in the emulator

The element format is emulator configuration, not a CR value. Choose it when
constructing the state, or reset the state into another format:

```python
from ipu_emu.ipu_state import IpuState
from ipu_isa.formats import ElementFormat

state = IpuState(fmt=ElementFormat.INT8)

# Or return an existing state to reset, in another format.
state.reset(ElementFormat.E4)
```

`IpuState.fmt` sets the element and row bytes, the accumulator type and the
vector registers. `INT8` and `E1`-`E7` are the hardware formats; `FP32` and
`INT32` are emulator-only four-byte formats (see
[the emulator-only formats](../wide-vector-debug-mode.md)). `state.fmt` cannot
be assigned, and an unknown format raises `ConfigurationError` when the state
is built, never while a program runs.

For `IpuApp` subclasses, declare the format in the kernel's `SPEC` with
`ExecutionConfig(fmt=...)`; `app.make_state()` builds the state in it.

`PadMode.POS_INF` and `PadMode.NEG_INF` (see
[Configuration registers](../reference/isa/configuration-registers.md#pad_mode))
have no integer representation: using them in INT8 or INT32 raises
`EmulatorError`.
