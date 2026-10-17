# Wide-vector debug mode (emulator only)

The Python emulator can run in an optional **wide-vector debug mode** so multiply-stage vectors are treated as **128 elements of 32-bit values** (float or integer) instead of 128 bytes of INT8/FP8. This is intended for debugging and analysis without 8-bit quantization on the multiply path.

Hardware behaviour is unchanged; this mode exists only in `ipu_emu`.

## When to use it

- Compare a model or kernel against a **full-precision** reference (FP32 elements) or a **wider integer** path (INT32 elements with 32-bit wrap on multiply/add).
- Keep the **same assembly** and **same XMEM byte addresses**; only how loads are sized and how mult/acc interpret data changes.

See [GitHub issue #33](https://github.com/rechefe/ipu-emulator/issues/33) for the original requirements.

## Enabling the mode

Wide-vector mode is the emulator-only element formats **`FP32`** and **`INT32`**.
Construct [`IpuState`](https://github.com/rechefe/ipu-emulator/blob/master/src/tools/ipu-emu-py/src/ipu_emu/ipu_state.py) in one of them:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fmt` | `ElementFormat.INT8` | `ElementFormat.FP32` or `ElementFormat.INT32` turns wide-vector mode on, with that element arithmetic. Read-only after construction; `state.reset(fmt)` changes it. |
| `aaq` | `AaqOptions()` | `AaqOptions(quantize_output=True)` makes `ACTIVATE.QUANTIZE` also quantize the **wide elements it just wrote to `POST_AAQ_REG`** into the **leading 128 bytes** of that register (use **`ACTIVATE.QUANTIZE` … `identity`** to quantize `r_acc` unchanged). Off by default: it stops after writing the wide activated elements. |

```python
from ipu_emu.ipu_state import IpuState
from ipu_emu.emulator import load_program, run_until_complete
from ipu_isa.formats import ElementFormat

state = IpuState(fmt=ElementFormat.FP32)
# load program, set CR15 / XMEM as usual, then:
run_until_complete(state)
```

`AaqOptions` is in `ipu_emu.activations`; the package root exports `IpuState`:

```python
from ipu_emu import IpuState
from ipu_emu.activations import AaqOptions
```

The high-level helper [`run_test`](https://github.com/rechefe/ipu-emulator/blob/master/src/tools/ipu-emu-py/src/ipu_emu/emulator.py) builds a default `IpuState()` unless you pass `state=`. To use wide-vector mode with your own harness, create the state as above and pass it, or use `load_program_from_binary` / `load_program` and `run_until_complete` or `run_with_debug` the same way tests do. A kernel harness declares its format in `SPEC` with `ExecutionConfig(fmt=ElementFormat.FP32)`.

## XMEM layout in wide mode

While **addresses are still byte addresses**, wide mode changes how much data some loads consume per instruction:

- **`LD.MULT`** reads **128 elements** (512 bytes, 128×FP32 or 128×INT32, depending on the format) from XMEM into internal staging for **`R0` or `R1` only**. The architectural 128-element `r` register in the regfile is not the source for mult operands in this mode.
- **`LD.MULT.CYCLIC`** reads **128 elements** (512 bytes) into one slot of `r_cyclic` at the given **`index`**. **`index` must be one of the four slot boundaries** — `0`, `128`, `256`, `384`; any other value raises an error. (Same boundaries as normal mode; in wide mode each slot is 512 bytes instead of 128 bytes.)

Prepare XMEM accordingly (e.g. raw `float32` or `int32` little-endian blobs).

## Alignment rules

Wide mode unpacks `r_cyclic` as 128 consecutive 32-bit elements starting at a **byte offset**:

- **`rc_idx`** (and any LR-encoded `src`/`ra_idx` used to further index into Ra) passed to mult instructions must be **4-byte aligned**. Unaligned values raise `EmulatorError` so you do not silently mis-read element boundaries.

## Semantics that differ from normal mode

- **Multiply masks** (`mask_offset` immediate slot 0–7 / `mask_shift` LR): mask-and-shift on `mult_res` is **mode-blind** — it runs identically in wide mode as in narrow mode (each of the 128 mask bits gates the corresponding FP32/INT32 element one-for-one, the same lane mapping as narrow mode's per-element gating).
- **`ACTIVATE.QUANTIZE`**: unless `AaqOptions.quantize_output` is set, it **does not quantize** in wide mode — it writes the activated wide elements to **`POST_AAQ_REG`** and the full element results stay in **`R_ACC`**. Use the existing debug-only **`ST.ACC`** instruction (or read `R_ACC` in Python) to dump 128 elements (512 bytes) of accumulator data.
- **LR and CR** are **not** widened; scalars such as **`MULT.RC.VE`**'s CR-encoded `src` still use the **low byte** of a CR as a signed value in the wide path.

## INT32 vs FP32

- **`ElementFormat.FP32`**: element multiply and accumulate-add use IEEE float; good for spotting FP8/INT8 quantization effects.
- **`ElementFormat.INT32`**: element multiply uses 32-bit signed wrap; add matches INT8-mode wrap semantics per element. The **`AGG.*`** aggregation instructions reduce elements as 32-bit signed integers (wrap semantics) when the element format is INT32.

## Related documentation

- [Debugging IPU Programs](debugging.md) — interactive **`BREAK`** / `debug_prompt` workflow (orthogonal to wide-vector mode; you can combine them by passing a state created with `fmt=ElementFormat.FP32` into `run_with_debug`).
