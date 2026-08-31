"""FP32 debug-input generation for layernorm_16x240.

This kernel is wide-vector FP32 only and generates its inputs here rather than
reading a checked-in data directory -- there is no ``<NAME>_DATA_DIR`` for this
kernel. ``__main__.py`` uses this generator, with the same recipe as
``test/test_layernorm_16x240_wide.py`` so that
``python -m ipu_apps.layernorm.layernorm_16x240`` exercises the kernel the way its test
does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.layernorm.layernorm_16x240 import N_CH, N_TOK, LANES

_SEED = 0x16240


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs.

    x is one WHOLE row per channel (N_TG=1): N_TOK valid lanes and the
    remaining LANES-N_TOK lanes zero. Rows are never shared between channels,
    so the padding is real storage, not a packing opportunity.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    x = np.zeros((N_CH, LANES), dtype=np.float32)
    x[:, :N_TOK] = rng.uniform(-4.0, 4.0, size=(N_CH, N_TOK))
    gamma = rng.uniform(0.5, 1.5, size=N_CH)
    beta = rng.uniform(-0.5, 0.5, size=N_CH)
    return {
        "input_path": _write(out_dir / "input_x_fp32.bin", x),
        "gamma_path": _write(out_dir / "gamma_fp32.bin", gamma),
        "beta_path": _write(out_dir / "beta_fp32.bin", beta),
    }
