"""Standalone runner + config list for softmax_columns.

Shared by the pytest wrapper in ipu-apps/test/. Computes a numpy column-softmax
reference over rows x width FP32 logits (softmax taken DOWN each column) and
compares. ``width`` is the real (unpadded) width; the harness pads it up to the
next power of two internally.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.softmax.softmax_columns import SoftmaxColumnsApp

ASM_PATH = Path(__file__).resolve().parent / "softmax_columns.asm"

# (rows, width, scale, seed)
TEST_CONFIGS = [
    (8, 128, 3.0, 0),      # single full chunk, few rows
    (64, 128, 4.0, 1),     # square 64x128
    (1, 128, 5.0, 2),      # single row -> softmax is all 1.0
    (16, 130, 3.0, 3),     # width 130 -> padded to 256 (2 chunks)
    (32, 200, 4.0, 4),     # width 200 -> padded to 256
    (128, 256, 3.0, 5),    # full 2-chunk width, many rows
    (64, 192, 50.0, 6),    # large |x| (stability), 192 -> 256
    (10, 129, 0.01, 7),    # near-uniform, 129 -> 256
    (256, 256, 3.0, 8),    # max width, 256 rows (no row-group cap)
]


def _reference(x: np.ndarray) -> np.ndarray:
    # Softmax DOWN columns (axis=0).
    z = np.exp(x - x.max(axis=0, keepdims=True))
    return z / z.sum(axis=0, keepdims=True)


def run_one(inst_file: Path, rows: int, width: int, scale: float, seed: int):
    rng = np.random.RandomState(seed)
    x = (rng.randn(rows, width) * scale).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        input_file = Path(tmp) / "in.bin"
        output_file = Path(tmp) / "out.bin"
        input_file.write_bytes(x.tobytes())
        app = SoftmaxColumnsApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=output_file,
            rows=rows,
            width=width,
        )
        _, cycles = app.run(max_cycles=8_000_000)
        out = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(rows, width)
    ref = _reference(x)
    max_abs = float(np.abs(out - ref).max())
    col_sums = out.sum(axis=0)
    return cycles, max_abs, col_sums, out, ref


if __name__ == "__main__":
    from ipu_as.lark_tree import assemble_to_bin_file

    with tempfile.TemporaryDirectory() as tmp:
        inst = Path(tmp) / "softmax_columns.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
        for rows, width, scale, seed in TEST_CONFIGS:
            cycles, max_abs, sums, _, _ = run_one(inst, rows, width, scale, seed)
            ok = max_abs < 1e-4 and np.allclose(sums, 1.0, atol=1e-5)
            print(f"rows={rows:3d} w={width:3d} scale={scale:<5} "
                  f"max_abs={max_abs:.2e} sum~1={np.allclose(sums,1.0,atol=1e-5)} "
                  f"cyc={cycles} {'OK' if ok else 'FAIL'}")
