"""Generate test data for the layernorm_256x144 app (wide-vector FP32 mode).

Reference: output[ch, tg, i] = γ[ch] × (x[ch,tg,i] − μ[tg,i]) / σ[tg,i] + β[ch]
where μ and σ are computed over the channel axis for each token independently.
No epsilon — matches the rsqrt activation which handles numerical stability internally.

Data layout written to file: rows in order (ch*N_TG + tg) for ch=0..143, tg=0,1.
Each row: N_TPG float32 values zero-padded to 128 → 512 bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_CH  = 144
N_TG  = 2
N_TPG = 128


def reference_layernorm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute LayerNorm and return (output, mean, var, inv_std).

    x:     [N_CH, N_TG, N_TPG] float32
    gamma: [N_CH]               float32
    beta:  [N_CH]               float32
    Returns output [N_CH, N_TG, N_TPG] float32.
    """
    mean = x.mean(axis=0)                         # [N_TG, N_TPG]
    centered = x - mean                            # [N_CH, N_TG, N_TPG]
    var = (centered ** 2).mean(axis=0)             # [N_TG, N_TPG]
    inv_std = np.where(var > 0.0, 1.0 / np.sqrt(var), 0.0).astype(np.float32)
    normalized = centered * inv_std                # [N_CH, N_TG, N_TPG]
    output = gamma[:, None, None] * normalized + beta[:, None, None]
    return output.astype(np.float32), mean.astype(np.float32), var.astype(np.float32), inv_std


def _pack_fp32_row(arr: np.ndarray) -> bytes:
    """Pack 1-D float32 array into a 512-byte row (zero-padded to 128 lanes)."""
    assert arr.ndim == 1 and len(arr) <= 128
    padded = np.zeros(128, dtype=np.float32)
    padded[: len(arr)] = arr
    return padded.tobytes()


def main() -> None:
    rng = np.random.RandomState(42)
    out_dir = Path(__file__).parent / "test_data_format" / "wide_fp32"
    out_dir.mkdir(parents=True, exist_ok=True)

    x     = rng.randn(N_CH, N_TG, N_TPG).astype(np.float32)
    gamma = rng.randn(N_CH).astype(np.float32) * 0.5 + 1.0
    beta  = rng.randn(N_CH).astype(np.float32) * 0.1

    # Write input: N_CH × N_TG rows in (ch*N_TG + tg) order, each 512 bytes
    input_bytes = b"".join(
        _pack_fp32_row(x[ch, tg])
        for ch in range(N_CH)
        for tg in range(N_TG)
    )
    (out_dir / "input_x_fp32.bin").write_bytes(input_bytes)

    # Write γ and β as raw float32 arrays (N_CH values × 4 bytes = 576 B)
    (out_dir / "gamma_fp32.bin").write_bytes(gamma.astype(np.float32).tobytes())
    (out_dir / "beta_fp32.bin").write_bytes(beta.astype(np.float32).tobytes())

    # Compute and write golden output
    output, mean, var, inv_std = reference_layernorm(x, gamma, beta)
    golden_bytes = b"".join(
        _pack_fp32_row(output[ch, tg])
        for ch in range(N_CH)
        for tg in range(N_TG)
    )
    (out_dir / "output_fp32.bin").write_bytes(golden_bytes)

    print(f"Generated {N_CH} channels × {N_TG} tg × {N_TPG} tokens")
    print(f"  input:  {len(input_bytes)} B")
    print(f"  gamma:  {len(gamma.tobytes())} B")
    print(f"  beta:   {len(beta.tobytes())} B")
    print(f"  output: {len(golden_bytes)} B")
    print(f"  mean range:    [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  var range:     [{var.min():.4f}, {var.max():.4f}]")
    print(f"  inv_std range: [{inv_std.min():.4f}, {inv_std.max():.4f}]")


if __name__ == "__main__":
    main()
