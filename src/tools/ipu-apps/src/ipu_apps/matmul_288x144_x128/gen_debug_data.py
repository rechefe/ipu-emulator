"""FP32 debug-input generation for matmul_288x144_x128.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_matmul_288x144_x128_wide.py`` so that ``python -m ipu_apps.matmul_288x144_x128`` exercises the
kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.matmul_288x144_x128 import K, N_OUT, N_TG, N_TOK

_SEED = 0xC0FFEE


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    # D is channel-major [K, N_TG*N_TOK]; W is output-major [N_OUT, K].
    D = rng.uniform(-1.0, 1.0, size=(K, N_TG * N_TOK))
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K))
    return {
        "input_path": _write(out_dir / "input_fp32.bin", D),
        "weights_path": _write(out_dir / "weights_fp32.bin", W),
    }
