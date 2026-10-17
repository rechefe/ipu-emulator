# Element formats

!!! note "Placeholder"
    E09-19 writes this page: the element formats (INT8, E1–E7, and the
    emulator-only FP32 and INT32), how each sets the element width, the XMEM row
    width and the accumulator type, and where each is checked. The
    [instruction reference](../reference/isa/instructions.md#element-formats)
    lists them. Until then this page holds how a kernel selects one in the
    emulator, moved here from the former IPU configuration page.

## Selecting the element format in the emulator

The element format (`dtype`) is emulator state, not a CR value. Set it when
constructing the state or in your app setup hook:

```python
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState

state = IpuState(dtype=DType.INT8)

# Or update an existing state before execution.
state.dtype = DType.E4
```

For `IpuApp` subclasses, store the selected element format on the app and copy it into
the state in `setup()`:

```python
class MyApp(IpuApp):
    def __init__(self, *, dtype: DType = DType.INT8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype

    def setup(self, state: IpuState) -> None:
        state.dtype = self.dtype
```

`PadMode.POS_INF` and `PadMode.NEG_INF` (see
[Configuration registers](../reference/isa/configuration-registers.md#pad_mode))
have no INT8 representation: using them while `state.dtype == DType.INT8`
raises `EmulatorError`.
