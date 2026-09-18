"""FP32 debug-input generation for attn_v_bcast_60.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_attn_v_bcast_60_wide.py`` so that
``python -m ipu_apps.attention.attn_v_bcast_60`` exercises the kernel the way its test
does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.attention.attn_v_bcast_60 import N_TOK, D, N_HEAD, N_CHAN, LANES, PV_STRIDE_ROWS

_SEED = 0xB60


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # P is staged KEY-major here (row s = all 16 query scores for key s) -- this
    # is what distinguishes the broadcast kernel from the query-major
    # attn_v_16x60.  V and O keep attn_v_16x60's layouts exactly.
    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        p_buf[h, :, :N_TOK] = P[h].T            # row s = P[:, s] (queries in lanes)

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    return {
        "p_path": _write(out_dir / "p_fp32.bin", p_buf),
        "v_path": _write(out_dir / "v_fp32.bin", v_buf),
    }
