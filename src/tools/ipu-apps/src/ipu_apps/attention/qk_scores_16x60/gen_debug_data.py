"""FP32 debug-input generation for qk_scores_16x60.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_qk_scores_16x60_wide.py`` so that ``python -m ipu_apps.attention.qk_scores_16x60``
exercises the kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.attention.qk_scores_16x60 import N, D

_SEED = 0x5C0


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    # Channel-major: element [token t, channel c] at (c*N + t).
    Q = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)

    return {
        "query_path": _write(out_dir / "q_fp32.bin", Q),
        "key_path": _write(out_dir / "k_fp32.bin", K),
    }
