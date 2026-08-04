"""FP32 debug-input generation for residual_add_64x192.

This kernel is wide-vector FP32 only, so it generates its inputs here rather
than reading a checked-in data directory. ``__main__.py`` uses this generator,
with the same recipe as ``test/test_residual_add_64x192_wide.py`` so that
``python -m ipu_apps.residual_add_64x192`` exercises the kernel the way its
test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.residual_add_64x192 import N_ROWS, N_TOK, LANES

_SEED = 0x64192


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs.

    Each of the N_ROWS channel rows is a full 128-lane XMEM row with only the
    first N_TOK lanes carrying tokens; the tail lanes stay zero.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    A = np.zeros((N_ROWS, LANES), dtype=np.float32)
    B = np.zeros((N_ROWS, LANES), dtype=np.float32)
    A[:, :N_TOK] = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_TOK))
    B[:, :N_TOK] = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_TOK))
    return {
        "input_a_path": _write(out_dir / "a_fp32.bin", A),
        "input_b_path": _write(out_dir / "b_fp32.bin", B),
    }
