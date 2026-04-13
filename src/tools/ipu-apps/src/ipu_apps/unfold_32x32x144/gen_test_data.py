"""Generate test data for the unfold_32x32x144 app.

Creates NHCW-striped input and golden FP32 output binaries for each dtype,
matching the IPU's acc.stride + mult.ev arithmetic exactly via ipu_math.

Usage::

    uv run python src/ipu_apps/unfold_32x32x144/gen_test_data.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, fp32_to_fp8_bytes

# -- Dimensions -------------------------------------------------------------

H         = 32    # spatial height
W         = 32    # spatial width
C         = 144   # channels
N_STRIPES = 8     # stripes of 4 rows each
N_STREAMS = 4     # TL, TR, BL, BR
N_TG      = 2     # token groups per channel (tg=0: stripes 0..3, tg=1: stripes 4..7)
N_TOK     = 128   # tokens per (stream, channel, tg) = 16×8 even-row/col pairs

# dtype 1.0 byte encodings (see __init__.py for derivation)
_ONES_BYTE: dict[DType, int] = {
    DType.INT8:     0x01,
    DType.E4: 0x38,
    DType.E5: 0x3C,
}

# Stream configs: (h_stride, v_stride)
#   h=enabled  → even cols (0, 2, ..., 30)
#   h=inverted → odd  cols (1, 3, ..., 31)
#   v=enabled  → even rows within each stripe (rows 0, 2)
#   v=inverted → odd  rows within each stripe (rows 1, 3)
_STREAMS = [
    ("enabled",  "enabled"),   # TL: even cols, even rows
    ("inverted", "enabled"),   # TR: odd  cols, even rows
    ("enabled",  "inverted"),  # BL: even cols, odd  rows
    ("inverted", "inverted"),  # BR: odd  cols, odd  rows
]


def _apply_stride(mult_result: list, h_stride: str, v_stride: str) -> list:
    """Decimate 128 MULT-result elements with acc.stride semantics.

    Models acc.stride with elements_in_row=32:
      128 elements = 4 rows × 32 elements/row  (one NHCW stripe's worth)
      v_stride: select rows 0,2 (enabled) or rows 1,3 (inverted) → 2 rows
      h_stride: select even (enabled) or odd (inverted) cols → 16 per row
      Output: 32 elements in row-major order within selected rows × cols.
    """
    rows = [mult_result[i * 32 : (i + 1) * 32] for i in range(4)]
    sel_rows = rows[::2] if v_stride == "enabled" else rows[1::2]
    result = []
    for row in sel_rows:
        result.extend(row[::2] if h_stride == "enabled" else row[1::2])
    return result   # 32 elements


def _reference_unfold(input_bytes: bytes, dtype: DType) -> bytes:
    """Compute expected IPU output for the unfold operation.

    Simulates the assembly loop exactly:
      For each (ch, stream, tg): process 4 stripes with acc.stride,
      each contributing 32 elements to a 128-element accumulator slot.

    Output layout (interleaved channel-major, matching teardown dump order):
      (N_STREAMS × N_TG × C) rows of 512 bytes (128 FP32/INT32 words).
      Row index = stream × 288 + ch × 2 + tg.
    """
    ones_byte = _ONES_BYTE[dtype]
    fmt = "<i" if dtype == DType.INT8 else "<f"
    n_rows = N_STREAMS * C * N_TG                # 1152 rows
    output = bytearray(n_rows * 512)

    for ch in range(C):
        for stream_idx, (h_stride, v_stride) in enumerate(_STREAMS):
            for tg in range(N_TG):
                # acc.stride writes 4 stripes → 128 accumulator elements.
                # Each stripe writes into a separate 32-element slot (stripe_offset * 32).
                # acc.stride writes mult_res values directly (no accumulation/addition).
                acc_128: list[int | float] = []
                for stripe_offset in range(4):
                    stripe   = tg * 4 + stripe_offset
                    row_num  = stripe * C + ch
                    row_data = input_bytes[row_num * 128 : (row_num + 1) * 128]

                    # mult.ev: broadcast ones_byte × each of 128 input bytes
                    mult_res = [ipu_mult(b, ones_byte, dtype) for b in row_data]

                    # acc.stride: decimate mult_res and write directly to acc slot
                    acc_128.extend(_apply_stride(mult_res, h_stride, v_stride))

                # Write 128 FP32/INT32 words to the output buffer
                out_row = stream_idx * C * N_TG + ch * N_TG + tg
                for i, val in enumerate(acc_128):
                    struct.pack_into(fmt, output, out_row * 512 + i * 4, val)

    return bytes(output)


def _spatial_to_nhcw_striped(spatial: np.ndarray) -> bytes:
    """Convert spatial (H, W, C) uint8 array to NHCW-striped binary.

    NHCW striped layout: 8 stripes × 144 channels, each row 128 bytes.
    Row (stripe, ch): spatial[stripe*4 : (stripe+1)*4, :, ch].flatten()
    """
    # spatial: (32, 32, 144) → (8 stripes, 4 rows, 32 cols, 144 ch)
    s = spatial.reshape(N_STRIPES, H // N_STRIPES, W, C)
    # transpose to (stripes, channels, rows, cols) → row-major flatten
    return s.transpose(0, 3, 1, 2).reshape(-1).tobytes()


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    spatial_shape = (H, W, C)

    if dtype == DType.INT8:
        spatial = rng.randint(-128, 128, size=spatial_shape, dtype=np.int8)
        input_bytes = _spatial_to_nhcw_striped(spatial.view(np.uint8))
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        spatial_fp32 = rng.uniform(-1.0, 1.0, size=spatial_shape).astype(np.float32)
        # encode each element as FP8
        spatial_fp8 = fp32_to_fp8_bytes(spatial_fp32.reshape(-1), dtype)
        spatial_uint8 = np.frombuffer(spatial_fp8, dtype=np.uint8).reshape(spatial_shape)
        input_bytes = _spatial_to_nhcw_striped(spatial_uint8)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    golden = _reference_unfold(input_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)

    print(
        f"  [{dtype_name}] input={len(input_bytes)}B  golden={len(golden)}B"
    )


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating unfold_32x32x144 test data...")
    _generate_for_dtype(out_dir, DType.INT8,     "int8")
    _generate_for_dtype(out_dir, DType.E4, "fp8_e4")
    _generate_for_dtype(out_dir, DType.E5, "fp8_e5")
    print("Done.")


if __name__ == "__main__":
    main()
