# Wide-vector debug mode (emulator only)

The Python emulator can run in an optional **wide-vector debug mode** so multiply-stage vectors are treated as **128 elements of 32-bit values** (float or integer) instead of 128 bytes of INT8/FP8. This is intended for debugging and analysis without 8-bit quantization on the multiply path.

Hardware behaviour is unchanged; this mode exists only in `ipu_emu`.

## When to use it

- Compare a model or kernel against a **full-precision** reference (FP32 elements) or a **wider integer** path (INT32 elements with 32-bit wrap on multiply/add).
- Keep the **same assembly** and **same XMEM byte addresses**; only how loads are sized and how mult/acc interpret data changes.

See [GitHub issue #33](https://github.com/rechefe/ipu-emulator/issues/33) for the original requirements.

## Enabling the mode

Construct [`IpuState`](https://github.com/rechefe/ipu-emulator/blob/master/src/tools/ipu-emu-py/src/ipu_emu/ipu_state.py) with keyword arguments:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `wide_vector_debug` | `False` | Turn wide-vector mode on. |
| `wide_vector_arithmetic` | `WideVectorArithmetic.FP32` | `FP32` or `INT32` element arithmetic. |
| `wide_vector_quantize_output` | `False` | If `True`, `ACTIVATE.QUANTIZE` also quantizes the **wide elements it just wrote to `POST_AAQ_REG`** into the **leading 128 bytes** of that register (use **`ACTIVATE.QUANTIZE` … `identity`** to quantize `r_acc` unchanged). If `False`, it stops after writing the wide activated elements. |

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

- **`LDR_MULT_REG`** reads **128 elements** (512 bytes, 128×FP32 or 128×INT32, depending on `wide_vector_arithmetic`) from XMEM into internal staging for **`R0` or `R1` only**. The architectural 128-element `r` register in the regfile is not the source for mult operands in this mode.
- **`LDR_CYCLIC_MULT_REG`** reads **128 elements** (512 bytes) into one slot of `r_cyclic` at the given **`index`**. **`index` must be one of the four slot boundaries** — `0`, `128`, `256`, `384`; any other value raises an error. (Same boundaries as normal mode; in wide mode each slot is 512 bytes instead of 128 bytes.)

Prepare XMEM accordingly (e.g. raw `float32` or `int32` little-endian blobs).

## Alignment rules

Wide mode unpacks `r_cyclic` as 128 consecutive 32-bit elements starting at a **byte offset**:

- **`rc_idx`** (and any LR-encoded `src`/`ra_idx` used to further index into Ra) passed to mult instructions must be **4-byte aligned**. Unaligned values raise `EmulatorError` so you do not silently mis-read element boundaries.

## Semantics that differ from normal mode

- **Multiply masks** (`mask_offset` immediate slot 0–7 / `mask_shift` LR): mask-and-shift on `mult_res` is **disabled** in wide mode, because the 128-bit mask layout does not map to 128 FP32/INT32 elements.
- **`ACTIVATE.QUANTIZE`**: unless `wide_vector_quantize_output=True`, it **does not quantize** in wide mode — it writes the activated wide elements to **`POST_AAQ_REG`** and the full element results stay in **`R_ACC`**. Use the existing debug-only **`STR_ACC_REG`** instruction (or read `R_ACC` in Python) to dump 128 elements (512 bytes) of accumulator data.
- **LR and CR** are **not** widened; scalars such as **`MULT.RC.VE`**'s CR-encoded `src` still use the **low byte** of a CR as a signed value in the wide path.

## INT32 vs FP32

- **`WideVectorArithmetic.FP32`**: element multiply and accumulate-add use IEEE float; good for spotting FP8/INT8 quantization effects.
- **`WideVectorArithmetic.INT32`**: element multiply uses 32-bit signed wrap; add matches INT8-mode wrap semantics per element. The **`AGG.*`** aggregation instructions reduce elements as 32-bit signed integers (wrap semantics) when the element format is INT32.

## Related documentation

- [Debugging IPU Programs](debugging.md) — interactive **`BREAK`** / `debug_prompt` workflow (orthogonal to wide-vector mode; you can combine them by passing a state created with `wide_vector_debug=True` into `run_with_debug`).
