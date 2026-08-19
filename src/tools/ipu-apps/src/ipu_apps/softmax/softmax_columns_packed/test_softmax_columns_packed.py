"""Standalone runner + config list for softmax_columns_packed.

Shared by the pytest wrapper in ipu-apps/test/. Computes a numpy column-softmax
reference (softmax DOWN each column) over rows x width FP32 logits with width
<= 64, and compares against the packed-element IPU output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp

ASM_PATH = Path(__file__).resolve().parent / "softmax_columns_packed.asm"

# (rows, width, scale, seed)
TEST_CONFIGS = [
    (16, 64, 3.0, 0),     # 2 rows/vec, clean width
    (64, 32, 4.0, 1),     # 4 rows/vec
    (100, 16, 3.0, 2),    # 8 rows/vec, rows not a multiple of rpv (tail)
    (33, 16, 5.0, 3),     # 8 rows/vec, 33 rows -> 5 vectors (tail of 1)
    (16, 33, 3.0, 4),     # width 33 -> pad to 64 (intra-group padding)
    (32, 20, 4.0, 5),     # width 20 -> pad to 32
    (16, 15, 50.0, 6),    # width 15 -> pad to 16, large |x| (stability)
    (8, 10, 0.01, 7),     # width 10 -> pad to 16, near-uniform
    (1, 64, 3.0, 8),      # single row -> softmax all 1.0
]


def _reference(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=0, keepdims=True))
    return z / z.sum(axis=0, keepdims=True)


def run_one(inst_file: Path, rows: int, width: int, scale: float, seed: int):
    rng = np.random.RandomState(seed)
    x = (rng.randn(rows, width) * scale).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        input_file = Path(tmp) / "in.bin"
        output_file = Path(tmp) / "out.bin"
        input_file.write_bytes(x.tobytes())
        app = SoftmaxColumnsPackedApp(
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
        inst = Path(tmp) / "softmax_columns_packed.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
        for rows, width, scale, seed in TEST_CONFIGS:
            cycles, max_abs, sums, _, _ = run_one(inst, rows, width, scale, seed)
            ok = max_abs < 1e-4 and np.allclose(sums, 1.0, atol=1e-5)
            print(f"rows={rows:3d} w={width:3d} scale={scale:<5} "
                  f"max_abs={max_abs:.2e} sum~1={np.allclose(sums,1.0,atol=1e-5)} "
                  f"cyc={cycles} {'OK' if ok else 'FAIL'}")
