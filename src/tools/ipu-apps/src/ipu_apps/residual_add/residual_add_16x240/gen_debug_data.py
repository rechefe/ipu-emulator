"""FP32 debug-input generation for residual_add_16x240.

Generates inputs with the same recipe as
``test/test_residual_add_16x240_wide.py``, so that
``python -m ipu_apps.residual_add.residual_add_16x240`` exercises the kernel the way its
test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator. Do NOT factor this into a shared
module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.residual_add.residual_add_16x240 import N_CH, N_TOK, LANES

_SEED = 0x5ADD


def pack_rows(values: np.ndarray) -> np.ndarray:
    """Place [N_CH, N_TOK] tokens into [N_CH, LANES] rows, one channel per row.

    The 112 lanes past the token prefix are zero padding -- a channel owns a
    whole row and never shares it.
    """
    rows = np.zeros((N_CH, LANES), dtype=np.float32)
    rows[:, :N_TOK] = values
    return rows


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    a = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)

    a_path = out_dir / "a_fp32.bin"
    b_path = out_dir / "b_fp32.bin"
    a_path.write_bytes(pack_rows(a).tobytes())
    b_path.write_bytes(pack_rows(b).tobytes())
    return {"input_a_path": a_path, "input_b_path": b_path}
