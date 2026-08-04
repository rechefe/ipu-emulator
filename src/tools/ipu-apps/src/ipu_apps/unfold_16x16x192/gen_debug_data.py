"""FP32 debug-input generation for unfold_16x16x192.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_unfold_16x16x192_wide.py`` so that ``python -m ipu_apps.unfold_16x16x192`` exercises the
kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.unfold_16x16x192 import H, W, C, N_STRIPES, LANES

_SEED = 0x016


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    # NHCW-striped: row (stripe, ch) holds that channel's stripe rows x W cols.
    stripe_h = H // N_STRIPES
    src = np.zeros((N_STRIPES * C, LANES), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * stripe_h
        for ch in range(C):
            src[stripe * C + ch, : stripe_h * W] = x[ch, r0 : r0 + stripe_h, :].reshape(-1)

    return {"input_path": _write(out_dir / "input_fp32.bin", src)}
