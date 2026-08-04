"""FP32 debug-input generation for unfold_8x8x240.

This kernel is wide-vector FP32 only, so it generates its inputs here using the
same recipe as ``test/test_unfold_8x8x240_wide.py``, so that
``python -m ipu_apps.unfold_8x8x240`` exercises the kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without dragging
in any other kernel's generator. Do NOT factor this into a shared module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.unfold_8x8x240 import H, W, C, LANES, _ROW_PACK_ORDER

_SEED = 0x058


def pack_stripe(x: np.ndarray) -> np.ndarray:
    """Pack a [C, H, W] spatial tensor into [C, LANES] XMEM rows.

    Row ch holds channel ch's 8x8 grid in the first H*W = 64 lanes, with spatial
    rows ordered by ``_ROW_PACK_ORDER`` so that ACC.STRIDE's even/odd view-row
    split matches the even/odd spatial-row split. Lanes 64..127 are zeroed so
    the stale tail of every output row is deterministic.
    """
    src = np.zeros((C, LANES), dtype=np.float32)
    for slot, spatial_row in enumerate(_ROW_PACK_ORDER):
        src[:, slot * W : (slot + 1) * W] = x[:, spatial_row, :]
    return src


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    src = pack_stripe(x)

    path = out_dir / "input_fp32.bin"
    path.write_bytes(np.ascontiguousarray(src, dtype=np.float32).tobytes())
    return {"input_path": path}
