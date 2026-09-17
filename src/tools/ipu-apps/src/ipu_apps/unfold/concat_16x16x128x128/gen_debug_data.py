"""FP32 debug-input generation for concat_16x16x128x128.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator (same convention
residual_add_16x240's own generator documents). Each input channel is one
opaque row -- this generator does not encode any particular spatial packing
inside a row, since the kernel itself never interprets one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.unfold.concat_16x16x128x128 import C_A, C_B, LANES

_SEED = 0xC047

def pack_rows(n_ch: int, rng: np.random.RandomState) -> np.ndarray:
    """[n_ch, LANES] random FP32 rows, one channel per row."""
    return rng.uniform(-1.0, 1.0, size=(n_ch, LANES)).astype(np.float32)


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    a = pack_rows(C_A, rng)
    b = pack_rows(C_B, rng)

    a_path = out_dir / "a_fp32.bin"
    b_path = out_dir / "b_fp32.bin"
    a_path.write_bytes(a.tobytes())
    b_path.write_bytes(b.tobytes())
    return {"input_a_path": a_path, "input_b_path": b_path}
