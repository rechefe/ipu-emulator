# Wide-vector debug mode (emulator only)

The Python emulator can run in an optional **wide-vector debug mode** so multiply-stage vectors are treated as **128 lanes of 32-bit values** (float or integer) instead of 128 bytes of INT8/FP8. This is intended for debugging and analysis without 8-bit quantization on the multiply path.

Hardware behaviour is unchanged; this mode exists only in `ipu_emu`.

## When to use it

- Compare a model or kernel against a **full-precision** reference (FP32 lanes) or a **wider integer** path (INT32 lanes with 32-bit wrap on multiply/add).
- Keep the **same assembly** and **same XMEM byte addresses**; only how loads are sized and how mult/acc interpret data changes.

See [GitHub issue #33](https://github.com/rechefe/ipu-emulator/issues/33) for the original requirements.

## Enabling the mode

Construct [`IpuState`](https://github.com/rechefe/ipu-emulator/blob/master/src/tools/ipu-emu-py/src/ipu_emu/ipu_state.py) with keyword arguments:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `wide_vector_debug` | `False` | Turn wide-vector mode on. |
| `wide_vector_arithmetic` | `WideVectorArithmetic.FP32` | `FP32` or `INT32` lane arithmetic. |
| `wide_vector_quantize_output` | `False` | If `True`, `AAQ` quantizes the **wide lanes in `POST_AAQ_REG`** (typically filled with **`ACTIVATE`** from `r_acc`, e.g. **`ACTIVATE` … `identity`** to copy) into the **leading 128 bytes** of that register. If `False`, `AAQ` does nothing in wide mode (wide lanes stay in `R_ACC` until you stage them). |

```python
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_as.lark_tree import assemble

state = IpuState(
    wide_vector_debug=True,
    wide_vector_arithmetic=WideVectorArithmetic.FP32,
    wide_vector_quantize_output=False,
)
# load program, set CR15 / XMEM as usual, then:
run_until_complete(state)
```

`WideVectorArithmetic` is re-exported from `ipu_emu` for convenience:

```python
from ipu_emu import IpuState, WideVectorArithmetic
```

The high-level helper [`run_test`](https://github.com/rechefe/ipu-emulator/blob/master/src/tools/ipu-emu-py/src/ipu_emu/emulator.py) always builds a default `IpuState()`. To use wide-vector mode with your own harness, create the state as above, then use `load_program_from_binary` / `load_program` and `run_until_complete` or `run_with_debug` the same way tests do.

## XMEM layout in wide mode

While **addresses are still byte addresses**, wide mode changes how much data some loads consume per instruction:

- **`LDR_MULT_REG`** reads **512 elements** from XMEM (128×FP32 or 128×INT32, depending on `wide_vector_arithmetic`) into internal staging for **`R0` or `R1` only**. The architectural 128-element `r` register in the regfile is not the source for mult operands in this mode.
- **`LDR_CYCLIC_MULT_REG`** reads **512 elements** into `r_cyclic` at the given **`index`**, which must be **aligned to 512** (same rule as 128-element alignment in normal mode, scaled to the wide chunk).

Prepare XMEM accordingly (e.g. raw `float32` or `int32` little-endian blobs).

## Alignment rules

Wide mode unpacks `r_cyclic` as 128 consecutive 32-bit lanes starting at a **byte offset**:

- **`cyclic_offset`** and **`fixed_cyclic_idx`** passed to mult instructions must be **4-byte aligned**. Unaligned values raise `EmulatorError` so you do not silently mis-read lane boundaries.

## Semantics that differ from normal mode

- **Multiply masks** (`mask_offset` immediate slot 0–7 / `mask_shift` LR): mask-and-shift on `mult_res` is **disabled** in wide mode, because the 128-bit mask layout does not map to 128 FP32/INT32 lanes.
- **`AAQ`**: unless `wide_vector_quantize_output=True`, **`AAQ` is a no-op** in wide mode; full lane results remain in **`R_ACC`**. Use the existing debug-only **`STR_ACC_REG`** instruction (or read `R_ACC` in Python) to dump 512 elements of accumulator data.
- **LR and CR** are **not** widened; scalars such as **`MULT.VE.CR`** still use the **low byte** of a CR as a signed value in the wide path.

## INT32 vs FP32

- **`WideVectorArithmetic.FP32`**: lane multiply and accumulate-add use IEEE float; good for spotting FP8/INT8 quantization effects.
- **`WideVectorArithmetic.INT32`**: lane multiply uses 32-bit signed wrap; add matches INT8-mode wrap semantics per lane. **`AGG`** post-functions that need integer results use integer-friendly paths when the lane format is INT32.

## Related documentation

- [Debugging IPU Programs](debugging.md) — interactive **`BREAK`** / `debug_prompt` workflow (orthogonal to wide-vector mode; you can combine them by passing a state created with `wide_vector_debug=True` into `run_with_debug`).
