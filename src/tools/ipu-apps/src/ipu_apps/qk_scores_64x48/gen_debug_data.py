"""FP32 debug-input generation for qk_scores_64x48.

This kernel is wide-vector FP32 only, so ``__main__.py`` generates its inputs
here, using the same recipe as ``test/test_qk_scores_64x48_wide.py`` so that
``python -m ipu_apps.qk_scores_64x48`` exercises the kernel the way its test
does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.qk_scores_64x48 import N, D, N_BLOCK

_SEED = 0x4C0


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    # Channel-major per (stream, head) block:
    # element [block b, token t, channel c] at (b*D + c)*N + t.
    Q = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N))
    K = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N))
    return {
        "query_path": _write(out_dir / "q_fp32.bin", Q),
        "key_path": _write(out_dir / "k_fp32.bin", K),
    }
