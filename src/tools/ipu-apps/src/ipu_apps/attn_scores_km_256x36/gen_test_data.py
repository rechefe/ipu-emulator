"""Generate test data for attn_scores_km_256x36 (Agent D, key-major scores).

Canonical channel-major activation tensors Q, K of shape [N_TOK, N_HEADS*D]:
    element [token t, channel h*36+c] at byte (h*36 + c)*256 + t.

Reference (head h):  S[i, s] = sum_c Q[i, c] * K[s, c]   (= Q_h @ K_hᵀ)
stored key-major:    golden_word[s*256 + i] = S[i, s]    (int32 INT8 / fp32 FP8)
matching the kernel's store order (rows of 128 words, (s, g) = (key, query group)).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

N_TOK   = 256
D       = 36
N_HEADS = 4
N_CH    = N_HEADS * D      # 144 channels in the canonical file
HEAD    = 0                # head verified by the golden


def _reference_scores(q_bytes: bytes, k_bytes: bytes, dtype: DType, head: int) -> bytes:
    """S[i, s] = sum_c Q[i,c]·K[s,c] for one head, stored key-major (word s*256+i)."""
    fmt = "<i" if dtype == DType.INT8 else "<f"
    head_base = head * D * N_TOK
    output = bytearray(N_TOK * N_TOK * 4)
    for s in range(N_TOK):           # key (outer / contiguous in output)
        for i in range(N_TOK):       # query (lane)
            acc: int | float = 0
            for c in range(D):
                q = q_bytes[head_base + c * N_TOK + i]
                k = k_bytes[head_base + c * N_TOK + s]
                prod = ipu_mult(q, k, dtype)
                acc = ipu_add(acc, prod, dtype)
                if dtype != DType.INT8:
                    acc = float(np.float32(acc))
            struct.pack_into(fmt, output, (s * N_TOK + i) * 4, acc)
    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    # Canonical channel-major tensors: shape [N_CH, N_TOK] flattened row-major
    # (channel-major: channel c row of 256 tokens contiguous).
    if dtype == DType.INT8:
        q_arr = rng.randint(-128, 128, size=(N_CH, N_TOK), dtype=np.int8)
        k_arr = rng.randint(-128, 128, size=(N_CH, N_TOK), dtype=np.int8)
        q_bytes = q_arr.reshape(-1).view(np.uint8).tobytes()
        k_bytes = k_arr.reshape(-1).view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        q_fp32 = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
        k_fp32 = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
        q_bytes = fp32_to_fp8_bytes(q_fp32.reshape(-1), dtype)
        k_bytes = fp32_to_fp8_bytes(k_fp32.reshape(-1), dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(q_bytes)
    (dtype_dir / f"weights_{dtype_name}.bin").write_bytes(k_bytes)

    golden = _reference_scores(q_bytes, k_bytes, dtype, HEAD)
    (dtype_dir / golden_name).write_bytes(golden)
    print(f"  [{dtype_name}] Q={len(q_bytes)}B  K={len(k_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating attn_scores_km_256x36 test data...")
    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.E4,   "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.E5,   "fp8_e5m2")
    print("Done.")


if __name__ == "__main__":
    main()
