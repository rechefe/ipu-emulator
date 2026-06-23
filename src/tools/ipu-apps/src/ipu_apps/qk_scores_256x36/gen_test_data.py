"""Generate test data for the qk_scores_256x36 app (one attention head).

Computes S[i, s] = sum_{c=0..35} Q[i, c] * K[s, c], query-major output.

Input layout (channel-major, what the harness reads):
    Q, K element [token t, channel c] at byte (c*N + t)*elem.

Output / golden layout (matches teardown dump):
    512 rows of 512 bytes (128 FP32 lanes each). Row r = i*N_TG + g holds
    S[i, g*128 : g*128+128] in the leading 128 lanes, zero-padded to 128.
    (Each R_ACC store is one 128-key group of query i.)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

N    = 256   # tokens (queries = keys)
D    = 36    # head_dim (contraction)
N_TG = 2     # key groups
N_TPG = 128  # keys per group


# --------------------------------------------------------------------------- #
# Quantized (INT8 / FP8) reference                                            #
# --------------------------------------------------------------------------- #
def _reference_scores(q_bytes: bytes, k_bytes: bytes, dtype: DType) -> bytes:
    """Channel-major byte inputs → query-major group-padded golden.

    Q[i,c] at byte c*N + i ; K[s,c] at byte c*N + s.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    out = bytearray(N * N_TG * 512)
    for i in range(N):
        for s in range(N):
            acc: int | float = 0
            for c in range(D):
                qv = q_bytes[c * N + i]
                kv = k_bytes[c * N + s]
                prod = ipu_mult(qv, kv, dtype)
                acc = ipu_add(acc, prod, dtype)
                if dtype != DType.INT8:
                    acc = float(np.float32(acc))
            g = s // N_TPG
            lane = s % N_TPG
            row = i * N_TG + g
            struct.pack_into(fmt, out, (row * 128 + lane) * 4, acc)
    return bytes(out)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    if dtype == DType.INT8:
        q_arr = rng.randint(-128, 128, size=(D, N), dtype=np.int8)   # [c, t]
        k_arr = rng.randint(-128, 128, size=(D, N), dtype=np.int8)
        q_bytes = q_arr.reshape(-1).view(np.uint8).tobytes()
        k_bytes = k_arr.reshape(-1).view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        q_fp32 = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)
        k_fp32 = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)
        q_bytes = fp32_to_fp8_bytes(q_fp32.reshape(-1), dtype)
        k_bytes = fp32_to_fp8_bytes(k_fp32.reshape(-1), dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"query_{dtype_name}.bin").write_bytes(q_bytes)
    (dtype_dir / f"key_{dtype_name}.bin").write_bytes(k_bytes)

    golden = _reference_scores(q_bytes, k_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)
    print(f"  [{dtype_name}] q={len(q_bytes)}B  k={len(k_bytes)}B  golden={len(golden)}B")


# --------------------------------------------------------------------------- #
# Wide-vector FP32 reference (primary correctness check)                       #
# --------------------------------------------------------------------------- #
def _pack_channel_major_fp32(arr: np.ndarray) -> bytes:
    """arr is [D, N] float32 (channel-major); pack as raw c-major bytes."""
    return arr.astype(np.float32).reshape(-1).tobytes()


def _generate_wide_fp32(out_dir: Path) -> None:
    rng = np.random.RandomState(42)
    wide_dir = out_dir / "wide_fp32"
    wide_dir.mkdir(parents=True, exist_ok=True)

    q = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)   # [c, t]
    k = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)

    (wide_dir / "query_fp32.bin").write_bytes(_pack_channel_major_fp32(q))
    (wide_dir / "key_fp32.bin").write_bytes(_pack_channel_major_fp32(k))

    # S[i,s] = sum_c Q[i,c]*K[s,c] = (Q^T @ K)[i,s] with Q,K as [c, t].
    scores = (q.T @ k).astype(np.float32)   # [N queries, N keys]

    out = bytearray(N * N_TG * 512)
    for i in range(N):
        for g in range(N_TG):
            row = i * N_TG + g
            seg = scores[i, g * N_TPG : (g + 1) * N_TPG]
            struct.pack_into(f"<{N_TPG}f", out, row * 512, *seg)
    (wide_dir / "out_fp32.bin").write_bytes(bytes(out))
    print(f"  [wide_fp32] q={q.size*4}B  k={k.size*4}B  golden={len(out)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating qk_scores_256x36 test data...")
    _generate_wide_fp32(out_dir)
    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.E4, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.E5, "fp8_e5m2")
    print("Done.")


if __name__ == "__main__":
    main()
