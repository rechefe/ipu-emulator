"""FP32 debug-input generation for fully_connected.

This kernel is wide-vector FP32 only, so the old checked-in INT8/FP8
``test_data_format/`` inputs no longer describe a valid input. ``__main__.py``
generates its inputs here instead, using the same recipe as
``test/test_fully_connected_wide.py`` so that ``python -m ipu_apps.fully_connected`` exercises the
kernel the way its test does.

Self-contained on purpose: this kernel can be merged on its own without
dragging in any other kernel's generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.fully_connected import SAMPLES_NUM, INPUT_NEURONS, OUTPUT_NEURONS

_SEED = 0xFC0


def _write(path: Path, arr: np.ndarray) -> Path:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return path


def generate(out_dir: Path) -> dict[str, Path]:
    """Write FP32 inputs into ``out_dir``; return app-constructor kwargs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(_SEED)

    inputs = rng.uniform(-1.0, 1.0, size=(SAMPLES_NUM, INPUT_NEURONS))
    weights = rng.uniform(-1.0, 1.0, size=(OUTPUT_NEURONS, INPUT_NEURONS))
    return {
        "inputs_path": _write(out_dir / "inputs_fp32.bin", inputs),
        "weights_path": _write(out_dir / "weights_fp32.bin", weights),
    }
