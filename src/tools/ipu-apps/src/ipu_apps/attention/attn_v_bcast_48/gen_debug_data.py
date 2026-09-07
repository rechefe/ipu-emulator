"""FP32 debug-input generation for attn_v_bcast_48.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_attn_v_bcast_48_wide.py`` so that
``python -m ipu_apps.attention.attn_v_bcast_48`` exercises the kernel the way its test
does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.attention.attn_v_bcast_48 import N_TOK, D, N_BLOCK, N_CHAN, LANES

_SEED = 0xB48


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    P = rng.uniform(-1.0, 1.0, size=(N_BLOCK, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    # P KEY-major: row (b*64 + s) holds key s's 64 query scores in the leading
    # lanes -- exactly what attn_scores_km_64x48 writes.
    p_buf = np.zeros((N_BLOCK * N_TOK, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        p_buf[b * N_TOK:(b + 1) * N_TOK, :N_TOK] = P[b].T

    # V channel-major: row (b*48 + t) holds channel (b,t)'s 64 keys.
    v_buf = np.zeros((N_CHAN, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        v_buf[b * D:(b + 1) * D, :N_TOK] = V[b]

    p_path = out_dir / "p_fp32.bin"
    v_path = out_dir / "v_fp32.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    return {"p_path": p_path, "v_path": v_path}
