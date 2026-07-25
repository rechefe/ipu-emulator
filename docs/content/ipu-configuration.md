# IPU Configuration

IPU programs read configuration from the state that the host prepares before execution.
This page describes the Python emulator configuration surface used by application
harnesses and tests.

## Configuration locations

| Location | Purpose | How to set it in Python |
| --- | --- | --- |
| `IpuState.dtype` | Arithmetic data type used by emulator math paths. It is not stored in a `CR` register. | `state.dtype = DType.INT8` or `IpuState(dtype=DType.INT8)` |
| `CR0` | Read-only constant zero. | Already initialized; writes are ignored. |
| `CR1` | Read-only constant one. | Already initialized; writes are ignored. |
| `CR2`-`CR14` | Application configuration such as base addresses, strides, loop bounds, and scalar constants. | `state.regfile.set_cr(index, value)` |
| `CR15` | Dstructure register. Bits `[7:0]` hold `valid_elements`; bits `[11:8]` hold `partition`; bits `[13:12]` hold `pad_mode`. | `state.set_cr_dstructure(valid_elements=128, partition=0, pad_mode=PadMode.ZERO)` |

`LR` and `CR` registers store 20-bit scalar values. Use
`LR_CR_SCALAR_VALUE_MASK` when encoding negative or wrapped constants for those
registers.

## Selecting dtype

`dtype` is emulator-only state, not a `CR` value. Set it when constructing the
state or in your app setup hook:

```python
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState

state = IpuState(dtype=DType.INT8)

# Or update an existing state before execution.
state.dtype = DType.E4
```

For `IpuApp` subclasses, store the selected dtype on the app and copy it into
the state in `setup()`:

```python
class MyApp(IpuApp):
    def __init__(self, *, dtype: DType = DType.INT8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype

    def setup(self, state: IpuState) -> None:
        state.dtype = self.dtype
```

## Selecting lane count and partition

The `AGG.*` aggregation instructions, `ACTIVATE`, and `AAQ` do not take a
`valid_elements` assembly operand directly. Instead, they take a mandatory
`cr_idx` operand naming the CR register that supplies `valid_elements` — there
is no implicit default, every instruction must name a CR register explicitly
(any `CR0`-`CR15`). Configure the chosen register from Python before running
the program:

```python
state.set_cr_dstructure(valid_elements=64, partition=0)

config = state.get_cr_dstructure()
valid_elements = config.valid_elements
partition = config.partition
```

The emulator defaults `CR15` to `valid_elements=128` and `partition=0`.
Activation clamps the active lane count to the available 128 lanes at execution
time.

The ACC-slot aggregation instructions (`AGG.SUM`, `AGG.SUM.FIRST`, `AGG.MAX`,
`AGG.MAX.FIRST`) take a mandatory `cr_idx` operand: the active lane count is
read from that register's `valid_elements` at runtime. The destination slot in
`R_ACC` is given by an LR register:

```asm
AGG.SUM LR0, CR15;;
AGG.MAX.FIRST LR1, CR15;;
ACTIVATE relu, CR15;;

AGG.SUM LR0, CR3;;
AGG.MAX.FIRST LR1, CR3;;
ACTIVATE relu, CR3;;
```

The MULT-slot lane-masking instructions (`MULT.RC.VV`, `MULT.RC.VE`,
`MULT.RC.VS`, `MULT.VE`, `MULT.EE`) likewise take a mandatory CR-index operand
(`cr_idx` on `MULT.RC.*`, `dstructure_cr_idx` on `MULT.VE`/`MULT.EE`, since
those two already use `cr_idx` for the scalar multiplier). The named register's
`partition` field drives the mask-and-shift partition vectors — there is no
implicit fallback to `CR15`:

```asm
MULT.RC.VV LR2, R0, 0, LR4, CR15;;
MULT.VE    LR0, CR3, 0, LR2, CR15;;
```

## Padding masked-out MULT_RES lanes

Any lane deactivated by the mask-and-shift logic above (see
`_mult_mask_and_shift`) is filled with a value chosen by the named register's
`pad_mode` field instead of always being zeroed:

| `pad_mode` | Fill value | Notes |
| --- | --- | --- |
| `PadMode.ZERO` (default) | `0` | Matches historical behavior; valid for INT8 and float dtypes. |
| `PadMode.POS_INF` | `+inf` | Float dtypes only — identity value for `min`-style reductions. |
| `PadMode.NEG_INF` | `-inf` | Float dtypes only — identity value for `max`-style reductions (e.g. `ACC.MAX`). |

```python
from ipu_emu.ipu_config import PadMode

state.set_cr_dstructure(valid_elements=128, partition=0, pad_mode=PadMode.NEG_INF)
```

`POS_INF`/`NEG_INF` have no INT8 representation — using them while
`state.dtype == DType.INT8` raises `EmulatorError`.

## Setting CR application constants

Use `CR2`-`CR14` for app-specific constants. `CR0` and `CR1` are reserved
read-only constants, and `CR15` is the dstructure register.

```python
from ipu_emu.ipu_config import LR_CR_SCALAR_VALUE_MASK

OUTPUT_BASE_ADDR = 0x40000
WEIGHTS_BASE_ADDR = 0x20000

state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
state.regfile.set_cr(3, 128)  # stride
state.regfile.set_cr(13, WEIGHTS_BASE_ADDR)

# Encode signed or wrapped LR/CR constants with the 20-bit scalar mask.
state.regfile.set_cr(9, (-128) & LR_CR_SCALAR_VALUE_MASK)
```

A common app setup combines all configuration in one place:

```python
def setup(self, state: IpuState) -> None:
    state.dtype = self.dtype
    state.set_cr_dstructure(valid_elements=128, partition=0)

    state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
    state.regfile.set_cr(3, 128)
    state.regfile.set_cr(4, 1)
    state.regfile.set_cr(13, WEIGHTS_BASE_ADDR)
```
