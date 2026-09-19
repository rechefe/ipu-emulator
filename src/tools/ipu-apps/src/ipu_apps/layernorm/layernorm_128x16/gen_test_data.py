"""Generate test data for the layernorm_128x16 app.

Reference: output[ch, i] = γ[ch] × (x[ch,i] − μ[i]) / σ[i] + β[ch]
where μ[i] = mean over channels, σ[i] = std over channels (no epsilon —
matches the rsqrt activation which handles numerical stability internally).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

N_CH  = 16
N_TPG = 128


def reference_layernorm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute LayerNorm and return (output, mean, var, inv_std) for verification.

    x:     [N_CH, N_TPG] float32
    gamma: [N_CH]        float32
    beta:  [N_CH]        float32
    Returns output [N_CH, N_TPG] float32.
    """
    mean = x.mean(axis=0)                        # [N_TPG]
    centered = x - mean                           # [N_CH, N_TPG]
    var = (centered ** 2).mean(axis=0)            # [N_TPG]
    inv_std = np.where(var > 0.0, 1.0 / np.sqrt(var), 0.0).astype(np.float32)
    normalized = centered * inv_std               # [N_CH, N_TPG]
    output = gamma[:, None] * normalized + beta[:, None]
    return output.astype(np.float32), mean.astype(np.float32), var.astype(np.float32), inv_std


def _pack_fp32_row(arr: np.ndarray, n_lanes: int = 128) -> bytes:
    """Pack 1-D float32 array into a 512-byte row (zero-padded to 128 lanes)."""
    assert arr.ndim == 1 and len(arr) <= n_lanes
    padded = np.zeros(n_lanes, dtype=np.float32)
    padded[: len(arr)] = arr
    return padded.tobytes()


def main() -> None:
    rng = np.random.RandomState(42)
    out_dir = Path(__file__).parent / "test_data_format" / "wide_fp32"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Random input, γ, β in reasonable FP32 range
    x = rng.randn(N_CH, N_TPG).astype(np.float32)
    gamma = rng.randn(N_CH).astype(np.float32) * 0.5 + 1.0  # near 1
    beta  = rng.randn(N_CH).astype(np.float32) * 0.1

    # Write input: N_CH rows × 512 bytes
    input_bytes = b"".join(_pack_fp32_row(x[ch]) for ch in range(N_CH))
    (out_dir / "input_x_fp32.bin").write_bytes(input_bytes)

    # Write γ and β as single 512-byte rows
    (out_dir / "gamma_fp32.bin").write_bytes(_pack_fp32_row(gamma))
    (out_dir / "beta_fp32.bin").write_bytes(_pack_fp32_row(beta))

    # Compute reference output
    output, mean, var, inv_std = reference_layernorm(x, gamma, beta)

    # Write golden output: N_CH rows × 512 bytes
    golden_bytes = b"".join(_pack_fp32_row(output[ch]) for ch in range(N_CH))
    (out_dir / "output_fp32.bin").write_bytes(golden_bytes)

    print(f"Generated {N_CH} channels × {N_TPG} tokens")
    print(f"  input:  {len(input_bytes)} B")
    print(f"  gamma:  {len(_pack_fp32_row(gamma))} B")
    print(f"  beta:   {len(_pack_fp32_row(beta))} B")
    print(f"  output: {len(golden_bytes)} B")
    print(f"  mean range:    [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  var range:     [{var.min():.4f}, {var.max():.4f}]")
    print(f"  inv_std range: [{inv_std.min():.4f}, {inv_std.max():.4f}]")


if __name__ == "__main__":
    main()
