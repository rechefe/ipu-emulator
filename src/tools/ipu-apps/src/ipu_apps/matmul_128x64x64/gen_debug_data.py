"""FP32 debug-input generation for matmul_128x64x64.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_matmul_128x64x64_wide.py`` so that ``python -m ipu_apps.matmul_128x64x64`` exercises the
kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.matmul_128x64x64 import M, K, N

_SEED = 0x646464


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    # A is [M, K]; W is [N, K]. The kernel contracts A @ W.T.
    A = rng.uniform(-1.0, 1.0, size=(M, K))
    W = rng.uniform(-1.0, 1.0, size=(N, K))
    return {
        "input_path": _write(out_dir / "input_fp32.bin", A),
        "weights_path": _write(out_dir / "weights_fp32.bin", W),
    }
