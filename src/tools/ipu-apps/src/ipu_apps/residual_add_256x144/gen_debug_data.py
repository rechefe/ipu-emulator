"""FP32 debug-input generation for residual_add_256x144.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_residual_add_256x144_wide.py`` so that ``python -m ipu_apps.residual_add_256x144`` exercises the
kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.residual_add_256x144 import N_ROWS, LANES

_SEED = 0xADD


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    A = rng.uniform(-1.0, 1.0, size=(N_ROWS, LANES))
    B = rng.uniform(-1.0, 1.0, size=(N_ROWS, LANES))
    return {
        "input_a_path": _write(out_dir / "a_fp32.bin", A),
        "input_b_path": _write(out_dir / "b_fp32.bin", B),
    }
