"""FP32 debug-input generation for layernorm_128x16.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_layernorm_128x16.py`` so that ``python -m ipu_apps.layernorm_128x16`` exercises the
kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.layernorm_128x16 import N_CH, N_TPG, ROW_BYTES

_SEED = 0x1216


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    lanes = ROW_BYTES // 4
    # x rows are (channel, token-group) pairs, N_TPG valid lanes each.
    x = np.zeros((N_CH, lanes), dtype=np.float32)
    x[:, :N_TPG] = rng.uniform(-4.0, 4.0, size=(N_CH, N_TPG))
    gamma = rng.uniform(0.5, 1.5, size=N_CH)
    beta = rng.uniform(-0.5, 0.5, size=N_CH)
    return {
        "input_path": _write(out_dir / "input_x_fp32.bin", x),
        "gamma_path": _write(out_dir / "gamma_fp32.bin", gamma),
        "beta_path": _write(out_dir / "beta_fp32.bin", beta),
    }
