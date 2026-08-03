"""Generate test data for the unfold_16x16x192 app (L4).

Creates NHCW-striped input and golden FP32 output binaries for each dtype,
matching the IPU's acc.stride + MULT.RC.VV arithmetic exactly via ipu_math.

Differs from the L3 generator (unfold_32x32x144) in geometry:
  L3: 8 stripes × (4 rows × 32 cols), 4 acc.stride slots → 128 tokens, 2 tg
  L4: 2 stripes × (8 rows × 16 cols), 2 acc.stride slots →  64 tokens, 1 row

Because a stream fills only 2 of the 4 r_acc slots, the upper 64 lanes of each
output row are STALE — they retain whatever the previous store left behind.
The golden models this explicitly (see ``_reference_unfold``).

Usage::

    uv run python src/ipu_apps/unfold_16x16x192/gen_test_data.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, fp32_to_fp8_bytes

# -- Dimensions -------------------------------------------------------------

H         = 16    # spatial height
W         = 16    # spatial width
C         = 192   # channels
N_STRIPES = 2     # stripes of 8 rows each (128 B row = 8 rows × 16 cols)
N_STREAMS = 4     # TL, TR, BL, BR
N_TOK     = 64    # valid tokens per (stream, channel) = 8×8 sub-grid
ROWS_PER_STRIPE = 8   # spatial rows packed into one 128-byte source row

# dtype 1.0 byte encodings (see __init__.py for derivation)
_ONES_BYTE: dict[DType, int] = {
    DType.INT8: 0x01,
    DType.E4:   0x38,
    DType.E5:   0x3C,
}

# Stream configs: (h_stride, v_stride)
#   h=enabled  → even cols (0, 2, ..., 14)
#   h=inverted → odd  cols (1, 3, ..., 15)
#   v=enabled  → even rows within each stripe (rows 0, 2, 4, 6)
#   v=inverted → odd  rows within each stripe (rows 1, 3, 5, 7)
_STREAMS = [
    ("enabled",  "enabled"),   # TL: even cols, even rows
    ("inverted", "enabled"),   # TR: odd  cols, even rows
    ("enabled",  "inverted"),  # BL: even cols, odd  rows
    ("inverted", "inverted"),  # BR: odd  cols, odd  rows
]


def _apply_stride(mult_result: list, h_stride: str, v_stride: str) -> list:
    """Decimate 128 MULT-result elements with acc.stride semantics.

    Models acc.stride with elements_in_row=16:
      128 elements = 8 rows × 16 elements/row  (one NHCW stripe's worth)
      v_stride: select rows 0,2,4,6 (enabled) or 1,3,5,7 (inverted) → 4 rows
      h_stride: select even (enabled) or odd (inverted) cols → 8 per row
      Output: 32 elements in row-major order within selected rows × cols.
    """
    rows = [mult_result[i * W : (i + 1) * W] for i in range(ROWS_PER_STRIPE)]
    sel_rows = rows[::2] if v_stride == "enabled" else rows[1::2]
    result = []
    for row in sel_rows:
        result.extend(row[::2] if h_stride == "enabled" else row[1::2])
    return result   # 4 rows × 8 cols = 32 elements


def _reference_unfold(input_bytes: bytes, dtype: DType) -> bytes:
    """Compute expected IPU output for the unfold operation.

    Simulates the assembly loop exactly, including r_acc persistence:
      For each channel, the four streams are stored in order TL, TR, BL, BR.
      Each stream writes 2 stripes × 32 elements into r_acc slots 0 and 1
      (lanes 0..63); lanes 64..127 are never written by this kernel, so every
      store emits whatever those lanes held from before. r_acc is zero at reset
      and slots 2/3 are never touched, so the stale half stays zero throughout —
      but it is modelled here rather than assumed, so the golden stays correct
      if that ever changes.

    Output layout (per-stream channel-major, matching teardown dump order):
      (N_STREAMS × C) rows of 512 bytes (128 FP32/INT32 words).
      Row index = stream × C + ch.  Only the first 64 words are meaningful.
    """
    ones_byte = _ONES_BYTE[dtype]
    fmt = "<i" if dtype == DType.INT8 else "<f"
    zero = 0 if dtype == DType.INT8 else 0.0
    n_rows = N_STREAMS * C
    output = bytearray(n_rows * 512)

    # Persistent model of r_acc lanes 64..127 (never written by this kernel).
    stale_upper: list[int | float] = [zero] * 64

    for ch in range(C):
        for stream_idx, (h_stride, v_stride) in enumerate(_STREAMS):
            # acc.stride writes 2 stripes → r_acc slots 0,1 = lanes 0..63.
            # It writes mult_res values directly (no accumulation/addition).
            acc_lower: list[int | float] = []
            for stripe in range(N_STRIPES):
                row_num  = stripe * C + ch
                row_data = input_bytes[row_num * 128 : (row_num + 1) * 128]

                # MULT.RC.VV: each input byte × 1.0 → identity, widened
                mult_res = [ipu_mult(b, ones_byte, dtype) for b in row_data]

                # acc.stride: decimate mult_res, write directly to acc slot
                acc_lower.extend(_apply_stride(mult_res, h_stride, v_stride))

            assert len(acc_lower) == N_TOK
            acc_128 = acc_lower + stale_upper

            out_row = stream_idx * C + ch
            for i, val in enumerate(acc_128):
                struct.pack_into(fmt, output, out_row * 512 + i * 4, val)

    return bytes(output)


def _spatial_to_nhcw_striped(spatial: np.ndarray) -> bytes:
    """Convert spatial (H, W, C) uint8 array to NHCW-striped binary.

    NHCW striped layout: 2 stripes × 192 channels, each row 128 bytes.
    Row (stripe, ch): spatial[stripe*8 : (stripe+1)*8, :, ch].flatten()
    """
    # spatial: (16, 16, 192) → (2 stripes, 8 rows, 16 cols, 192 ch)
    s = spatial.reshape(N_STRIPES, ROWS_PER_STRIPE, W, C)
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
        spatial_fp8 = fp32_to_fp8_bytes(spatial_fp32.reshape(-1), dtype)
        spatial_uint8 = np.frombuffer(spatial_fp8, dtype=np.uint8).reshape(spatial_shape)
        input_bytes = _spatial_to_nhcw_striped(spatial_uint8)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    golden = _reference_unfold(input_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)

    print(f"  [{dtype_name}] input={len(input_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating unfold_16x16x192 test data...")
    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.E4,   "fp8_e4")
    _generate_for_dtype(out_dir, DType.E5,   "fp8_e5")
    print("Done.")


if __name__ == "__main__":
    main()
