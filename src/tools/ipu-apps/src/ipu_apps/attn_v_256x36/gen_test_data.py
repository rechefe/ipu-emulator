"""Generate test data for the attn_v_256x36 app (attn@V AGG kernel).

O[i, t] = sum_s P[i, s] * V[s, t]   per head h in [0,4), i,s in [0,256), t in [0,36).

P query-major  byte offset (1 B/elem): h*65536 + i*256 + s
V channel-major byte offset (1 B/elem): (h*36 + t)*256 + s
O golden (FP32/INT32, 1024 B/channel): (h*36 + t)*1024 + (i//128)*512 + (i%128)*4

Reproducible with numpy RandomState(42); FP8 uses ipu_math reference arithmetic.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

N_TOK  = 256
D      = 36
N_HEAD = 4

P_HEAD_STRIDE = 65536    # 256 * 256
O_CHAN_BYTES  = 1024     # 2 groups * 512 (FP32)


def _reference(p_bytes: bytes, v_bytes: bytes, dtype: DType) -> bytes:
    """Compute O matching the kernel's AGG datapath exactly.

    The kernel reduces each 128-key chunk with AGG.SUM over mult_res (a float64
    left-fold of float32 products, rounded to float32 on write to R_ACC), then
    AGG.SUM of chunk 1 adds (float64) to the chunk-0 float32 partial and rounds
    to float32. INT8 uses int32 accumulation via ipu_add. Mirroring the
    per-chunk reduction (not a per-key fold) is required for E5M2 to match.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    out = bytearray(N_HEAD * D * O_CHAN_BYTES)
    for h in range(N_HEAD):
        for t in range(D):
            chan = h * D + t
            for i in range(N_TOK):
                if dtype == DType.INT8:
                    acc: int = 0
                    for s in range(N_TOK):
                        p = p_bytes[h * P_HEAD_STRIDE + i * N_TOK + s]
                        v = v_bytes[chan * N_TOK + s]
                        acc = ipu_add(acc, ipu_mult(p, v, dtype), dtype)
                    res: int | float = acc
                else:
                    # Per-chunk float64 sum of float32 products -> float32 partial,
                    # then float64 add of the two float32 chunk partials -> float32.
                    racc = np.float32(0.0)
                    for c in range(2):
                        partial = 0.0  # float64 left-fold (matches _agg_sum_lanes)
                        for s in range(c * 128, (c + 1) * 128):
                            p = p_bytes[h * P_HEAD_STRIDE + i * N_TOK + s]
                            v = v_bytes[chan * N_TOK + s]
                            partial += float(np.float32(ipu_mult(p, v, dtype)))
                        if c == 0:
                            racc = np.float32(partial)          # AGG.SUM.FIRST
                        else:
                            racc = np.float32(partial + float(racc))  # AGG.SUM
                    res = float(racc)
                off = chan * O_CHAN_BYTES + (i // 128) * 512 + (i % 128) * 4
                struct.pack_into(fmt, out, off, res)
    return bytes(out)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    p_shape = (N_HEAD, N_TOK, N_TOK)   # head-major, query-major rows
    v_shape = (N_HEAD * D, N_TOK)      # channel-major rows

    if dtype == DType.INT8:
        p_arr = rng.randint(-128, 128, size=p_shape, dtype=np.int8)
        v_arr = rng.randint(-128, 128, size=v_shape, dtype=np.int8)
        p_bytes = p_arr.reshape(-1).view(np.uint8).tobytes()
        v_bytes = v_arr.reshape(-1).view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        p_fp32 = rng.uniform(-1.0, 1.0, size=p_shape).astype(np.float32)
        v_fp32 = rng.uniform(-1.0, 1.0, size=v_shape).astype(np.float32)
        p_bytes = fp32_to_fp8_bytes(p_fp32.reshape(-1), dtype)
        v_bytes = fp32_to_fp8_bytes(v_fp32.reshape(-1), dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"p_{dtype_name}.bin").write_bytes(p_bytes)
    (dtype_dir / f"v_{dtype_name}.bin").write_bytes(v_bytes)

    golden = _reference(p_bytes, v_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)
    print(f"  [{dtype_name}] P={len(p_bytes)}B  V={len(v_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating attn_v_256x36 test data...")
    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.E4, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.E5, "fp8_e5m2")
    print("Done.")


if __name__ == "__main__":
    main()
