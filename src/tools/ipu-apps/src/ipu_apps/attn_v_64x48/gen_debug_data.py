"""FP32 debug-input generation for attn_v_64x48.

This kernel is wide-vector FP32 only, so ``__main__.py`` generates its inputs
here, using the same recipe as ``test/test_attn_v_64x48_wide.py`` so that
``python -m ipu_apps.attn_v_64x48`` exercises the kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.attn_v_64x48 import N_TOK, D, N_BLOCK, N_CHAN, LANES

_SEED = 0xA48


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    P = rng.uniform(-1.0, 1.0, size=(N_BLOCK, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    # P and V are staged in the kernel's own row layout: one whole row per
    # P query row / V channel, keys in the leading N_TOK lanes.
    p_buf = np.zeros((N_BLOCK * N_TOK, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        p_buf[b * N_TOK:(b + 1) * N_TOK, :N_TOK] = P[b]   # row i = all keys for query i

    v_buf = np.zeros((N_CHAN, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        v_buf[b * D:(b + 1) * D, :N_TOK] = V[b]           # row t = all keys for channel t

    return {
        "p_path": _write(out_dir / "p_fp32.bin", p_buf),
        "v_path": _write(out_dir / "v_fp32.bin", v_buf),
    }
